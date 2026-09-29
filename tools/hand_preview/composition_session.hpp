#pragma once
#include <kfcore/gesture_interaction/composition_esn.hpp>
#include "gesture_status.hpp"
#include <kfcore/gesture_interaction/action_gate.hpp>
#include <kfcore/gesture_interaction/experiment.hpp>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace preview {
using namespace kfcore::gesture_interaction;
inline int normalize_key(int key) {
    return key >= 'A' && key <= 'Z' ? key + ('a' - 'A') : key;
}
// UI thread owns all data. Labels are explicit key presses, never inferred from
// motion. Raw clips are authoritative; the ESN states are derived training caches.
class CompositionSession {
public:
    std::string status = "ESN：未训练｜录制完整手势后按 T";
    std::string feedback = "分别录制各类样本，再按 T 训练";
    std::string prediction = "ESN：未训练";
    explicit CompositionSession(CompositionOptions options, ActionGateOptions gate = {}) : model_(options), gate_options_(gate) {
        (void)ActionGate(gate);
        clip_.reserve(options.maximum_samples);
        heldout_.reserve(options.maximum_clips);
        training_.reserve(options.maximum_clips);
    }
    CompositionTask task() const noexcept { return model_.options().task; }
    int class_count() const { return model_.class_count(); }
    const char* label_name(int index) const {
        if (task() == CompositionTask::Motion && index == 0) return "非挥手";
        if (task() == CompositionTask::Interaction && index == 0) return "非动作";
        return composition_display_name(composition_labels(task()).at(index));
    }
    bool recording() const noexcept { return collecting_; }
    bool dirty() const noexcept { return dirty_; }
    const CompositionEsn& model() const noexcept { return model_; }
    ActionGateOptions gate_options() const noexcept { return gate_options_; }
    std::uint64_t revision() const noexcept { return revision_; }
    void mark_saved() noexcept { dirty_ = false; }
    SessionArchive archive() const {
        if (collecting_) throw std::runtime_error("请先完成或取消录制，再保存");
        return {model_.options(), gate_options_, model_.weights(), model_.trained(), training_, heldout_};
    }
    static std::unique_ptr<CompositionSession> restore(const SessionArchive& saved) {
        auto session = std::make_unique<CompositionSession>(saved.options, saved.gate);
        session->model_ = std::move(*restore_model(saved));
        session->training_ = saved.training; session->heldout_ = saved.heldout;
        session->reset_window("已加载，实时历史已清空");
        session->feedback = "实验已加载，样本和模型已恢复";
        return session;
    }
    std::string class_names() const {
        if (task() == CompositionTask::Motion) return "非挥手/挥手";
        return task() == CompositionTask::Legacy ? "无组合/张握张/握张握" :
            std::string("非动作/") + label_name(1) + "/" + label_name(2);
    }
    void reset_window(const char* reason) {
        if (collecting_) {
            const int retry_key = composition_index(task(), label_) + (test_ ? kCompositionClasses : 0);
            feedback = std::string("录制未保存：") + reason + "；按 " + std::to_string(retry_key) + " 重试";
        }
        clip_.clear(); collecting_ = false; record_start_.reset();
        last_side_.reset();
        prediction = model_.trained() ? "ESN：正在积累帧历史" : "ESN：未训练";
        status = std::string("ESN：") + reason;
    }
    void key(int key) {
        key = normalize_key(key);
        try {
            if (key >= '0' && key <= '5') {
                if (collecting_) { feedback = "正在录制，请等待完成或按 C 取消"; return; }
                const int index = key - '0';
                if (index % kCompositionClasses >= class_count()) {
                    feedback = "挥手模式：0/1 训练非挥手/挥手；3/4 测试；2/5 不使用";
                    return;
                }
                label_ = composition_labels(task())[index % kCompositionClasses]; test_ = index >= kCompositionClasses;
                clip_.clear(); collecting_ = true; record_start_.reset(); last_side_.reset();
                status = std::string(test_ ? "测试录制：" : "训练录制：") + label_name(composition_index(task(), label_));
                feedback = status;
                if (task() == CompositionTask::Interaction) {
                    if (label_ == Composition::Grasp) feedback += "｜任意手型→握拳→保持";
                    else if (label_ == Composition::Release) feedback += "｜任意手型→握拳→张掌";
                    else feedback += "｜无完整抓取或放开序列";
                }
            } else if (key == 't') {
                if (collecting_) { feedback = "请等待录制完成或按 C 取消后再训练"; return; }
                clip_.clear(); model_.train_readout(); ++revision_;
                dirty_ = true;
                feedback = task() == CompositionTask::Legacy ? "ESN 已训练；实时按不重叠时间窗预测" : "训练完成；请用新录制片段测试滑动窗事件";
                prediction = "ESN：正在积累帧历史";
            } else if (key == 'v') evaluate();
            else if (key == 'c') reset_window("用户取消录制");
        } catch (const std::exception& e) { feedback = std::string("ESN: ") + e.what(); }
    }
    std::string summary() const {
        const auto& counts = model_.counts();
        std::array<int, kCompositionClasses> tests{};
        for (const auto& clip : heldout_) ++tests[composition_index(task(), clip.label)];
        std::string train_counts, test_counts;
        for (int i = 0; i < class_count(); ++i) {
            if (i) { train_counts += "/"; test_counts += "/"; }
            train_counts += std::to_string(counts[i]); test_counts += std::to_string(tests[i]);
        }
        return std::string(model_.trained() ? "已训练" : "未训练") +
            "｜训练 " + class_names() + " " + train_counts + "（每类需 " +
            std::to_string(model_.options().minimum_clips_per_class) + " 段）｜测试 " + test_counts;
    }
    void update(const kfcore::mediapipe::GestureFrame& frame, double seconds, int width = 0, int height = 0) {
        try {
            const auto sample = composition_sample(frame, seconds, width, height);
            if (!sample) {
                reset_window("检测到多只手，请仅露出一只手");
                return;
            }
            if (task() == CompositionTask::Motion && !frame.gestures.empty() && !sample->wrist)
                throw std::invalid_argument("动作识别需要来源图像尺寸");
            if (!clip_.empty()) {
                if (seconds <= clip_.back().seconds) {
                    reset_window("相机时间戳倒退或重复，请检查采集时间"); return;
                }
                if (seconds-clip_.back().seconds > model_.options().maximum_gap_seconds) {
                    reset_window("帧间隔过长，请检查相机与推理速度"); return;
                }
            }
            if (frame.gestures.empty()) {
                status = std::string(kNoHandStatus) + "，请在充足光线下露出整只手";
                if (clip_.empty()) { record_start_.reset(); return; }
                if (seconds-last_seen_ > model_.options().maximum_gap_seconds) {
                    reset_window("手部丢失过久，请露出整只手并改善照明"); return;
                }
                status += "｜短暂丢失已记为缺失帧";
            } else {
                const auto side = frame.landmarks.hands.front().handedness;
                if (task() != CompositionTask::Legacy && last_side_ && *last_side_ != side) {
                    reset_window("左右手发生切换，请用同一只手录制");
                    last_side_ = side; return;
                }
                last_side_ = side;
                last_seen_ = seconds;
                status = observation(frame.landmarks.hands.front(), frame.gestures.front());
            }
            if (!collecting_ && !model_.trained()) return;
            if (collecting_ && task() != CompositionTask::Legacy) {
                if (frame.gestures.empty() && clip_.empty()) { record_start_.reset(); return; }
                if (!record_start_) record_start_ = seconds + model_.options().record_countdown_seconds;
                if (seconds < *record_start_) {
                    status += "｜准备 " + std::string(label_name(composition_index(task(), label_))) + "，倒计时 " +
                        std::to_string(int(std::ceil(*record_start_-seconds))) + " 秒；保持起始姿势";
                    return;
                }
            }
            if (clip_.size() >= model_.options().maximum_samples) {
                reset_window("样本数达到上限，请缩短配置的时间窗"); return;
            }
            clip_.push_back(*sample);
            if (collecting_) {
                constexpr int kPercent = 100;
                const int progress = int(std::min(double(kPercent), kPercent * (seconds-clip_.front().seconds) / model_.options().duration_seconds));
                status += "｜录制 " + std::string(label_name(composition_index(task(), label_))) + " " + std::to_string(std::min(progress, kPercent)) + "%";
            }
            if (seconds-clip_.front().seconds < model_.options().duration_seconds) return;
            if (collecting_) {
                if (test_) {
                    if (heldout_.size() >= model_.options().maximum_clips) throw std::runtime_error("测试片段达到容量上限");
                    (void)model_.encode(clip_);
                    heldout_.push_back({clip_, label_, batch_, unix_ms()});
                } else {
                    if (training_.size() >= model_.options().maximum_clips) throw std::runtime_error("训练片段达到容量上限");
                    training_.push_back({clip_, label_, batch_, unix_ms()});
                    try { model_.add_training(clip_, label_); }
                    catch (...) { training_.pop_back(); throw; }
                }
                dirty_ = true;
                if (!test_) { prediction = "ESN：未训练"; ++revision_; }
                feedback = test_ ? "测试片段已暂存内存" : "训练片段已暂存内存；按 T 训练输出层";
                status = "ESN：录制完成";
                collecting_ = false;
            } else {
                const auto result = model_.predict(clip_);
                prediction = std::string(task() == CompositionTask::Legacy ? "上一时间窗：" : "时间窗候选：") +
                    label_name(composition_index(task(), result.label));
                if (task() != CompositionTask::Legacy) {
                    // Keep overlapping evidence, but never reuse samples as training labels.
                    const double cutoff = clip_.front().seconds + model_.options().live_stride_seconds;
                    auto end = std::lower_bound(clip_.begin(), clip_.end(), cutoff,
                        [](const CompositionSample& s, double t) { return s.seconds < t; });
                    clip_.erase(clip_.begin(), end);
                    return;
                }
            }
            clip_.clear();
        } catch (const std::exception& e) { reset_window(e.what()); }
    }
private:
    static std::uint64_t unix_ms() {
        return std::uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    }
    static std::string observation(const kfcore::hand_models::HandResult& hand, const kfcore::mediapipe::GesturePrediction& gesture) {
        using kfcore::hand_models::Handedness;
        const char* side = hand.handedness == Handedness::Left ? "左手" :
            hand.handedness == Handedness::Right ? "右手" : "未知手";
        std::ostringstream message;
        message << "检测到 1 只" << side << "｜" << gesture_display_name(gesture.label)
                << " " << std::fixed << std::setprecision(2) << gesture.scores[std::size_t(gesture.label)];
        return message.str();
    }
    void evaluate() {
        if (!model_.trained()) throw std::runtime_error("请先训练，再评估");
        std::array<int, kCompositionClasses> counts{};
        for (const auto& clip : heldout_) ++counts[composition_index(task(), clip.label)];
        for (int i = 0; i < class_count(); ++i) if (counts[i] < model_.options().minimum_clips_per_class)
            throw std::runtime_error(task() == CompositionTask::Motion ?
                "需要非挥手与挥手测试片段（按 3/4）" : "每一类都需要测试片段（按 3/4/5）");
        int correct = 0;
        std::array<std::array<int, kCompositionClasses>, kCompositionClasses> confusion{};
        for (const auto& clip : heldout_) {
            const auto prediction = model_.predict(clip.samples);
            ++confusion[composition_index(task(), clip.label)][composition_index(task(), prediction.label)];
            if (prediction.label == clip.label) ++correct;
        }
        std::cout << "测试混淆矩阵：行=真实，列=预测 [" << class_names() << "]\n";
        for (int row = 0; row < class_count(); ++row) {
            for (int column = 0; column < class_count(); ++column)
                std::cout << (column ? " " : "") << confusion[row][column];
            std::cout << '\n';
        }
        feedback = "测试正确 " + std::to_string(correct) + "/" + std::to_string(heldout_.size()) + "；混淆矩阵见控制台";
        std::cout << feedback << '\n';
    }
    CompositionEsn model_;
    ActionGateOptions gate_options_;
    std::uint64_t revision_ = 0;
    std::optional<kfcore::hand_models::Handedness> last_side_;
    std::optional<double> record_start_;
    CompositionClip clip_;
    std::vector<RecordedClip> training_, heldout_;
    std::uint64_t batch_ = unix_ms();
    bool dirty_ = false;
    double last_seen_ = 0;
    Composition label_ = Composition::None;
    bool collecting_ = false, test_ = false;
};
} // namespace preview
