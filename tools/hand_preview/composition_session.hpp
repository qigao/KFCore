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
    std::string status = "ESN: UNTRAINED | record whole gesture sequences, then T";
    std::string feedback = "Record each class, then press T to train";
    std::string prediction = "ESN: UNTRAINED";
    explicit CompositionSession(CompositionOptions options, ActionGateOptions gate = {}) : model_(options), gate_options_(gate) {
        (void)ActionGate(gate);
        clip_.reserve(options.maximum_samples);
        heldout_.reserve(options.maximum_clips);
        training_.reserve(options.maximum_clips);
    }
    CompositionTask task() const noexcept { return model_.options().task; }
    int class_count() const { return model_.class_count(); }
    const char* label_name(int index) const { return composition_name(composition_labels(task()).at(index)); }
    bool recording() const noexcept { return collecting_; }
    bool dirty() const noexcept { return dirty_; }
    const CompositionEsn& model() const noexcept { return model_; }
    ActionGateOptions gate_options() const noexcept { return gate_options_; }
    std::uint64_t revision() const noexcept { return revision_; }
    void mark_saved() noexcept { dirty_ = false; }
    SessionArchive archive() const {
        if (collecting_) throw std::runtime_error("finish or cancel recording before saving");
        return {model_.options(), gate_options_, model_.weights(), model_.trained(), training_, heldout_};
    }
    static std::unique_ptr<CompositionSession> restore(const SessionArchive& saved) {
        auto session = std::make_unique<CompositionSession>(saved.options, saved.gate);
        session->model_ = std::move(*restore_model(saved));
        session->training_ = saved.training; session->heldout_ = saved.heldout;
        session->reset_window("loaded; fresh live history");
        session->feedback = "Experiment loaded; samples and model restored";
        return session;
    }
    std::string class_names() const {
        if (task() == CompositionTask::Motion) return "Neutral/Wave";
        return task() == CompositionTask::Legacy ? "No-combo/OCO/COC" :
            std::string("Neutral/") + label_name(1) + "/" + label_name(2);
    }
    void reset_window(const char* reason) {
        if (collecting_) {
            const int retry_key = composition_index(task(), label_) + (test_ ? kCompositionClasses : 0);
            feedback = std::string("Not saved: ") + reason + "; press " + std::to_string(retry_key) + " to retry";
        }
        clip_.clear(); collecting_ = false; record_start_.reset();
        last_side_.reset();
        prediction = model_.trained() ? "ESN: collecting history" : "ESN: UNTRAINED";
        status = std::string("ESN: ") + reason;
    }
    void key(int key) {
        key = normalize_key(key);
        try {
            if (key >= '0' && key <= '5') {
                if (collecting_) { feedback = "Recording in progress; finish or press C to cancel"; return; }
                const int index = key - '0';
                if (index % kCompositionClasses >= class_count()) {
                    feedback = "Wave mode: 0/1 train Neutral/Wave; 3/4 test; 2/5 unused";
                    return;
                }
                label_ = composition_labels(task())[index % kCompositionClasses]; test_ = index >= kCompositionClasses;
                clip_.clear(); collecting_ = true; record_start_.reset();
                status = std::string(test_ ? "TEST recording: " : "TRAIN recording: ") + composition_name(label_);
                feedback = status;
            } else if (key == 't') {
                if (collecting_) { feedback = "Finish recording or press C before training"; return; }
                clip_.clear(); model_.train_readout(); ++revision_;
                dirty_ = true;
                feedback = task() == CompositionTask::Legacy ? "ESN: trained; live non-overlapping windows (not events)" : "Trained: live sliding-window events; test with NEW clips";
                prediction = "ESN: collecting history";
            } else if (key == 'v') evaluate();
            else if (key == 'c') reset_window("cancelled by user");
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
        return std::string(model_.trained() ? "TRAINED" : "UNTRAINED") +
            " | train " + class_names() + " " + train_counts + " (need " +
            std::to_string(model_.options().minimum_clips_per_class) + " each) | test " + test_counts;
    }
    void update(const kfcore::mediapipe::GestureFrame& frame, double seconds, int width = 0, int height = 0) {
        try {
            const auto sample = composition_sample(frame, seconds, width, height);
            if (!sample) {
                reset_window("multiple detections - keep only one hand visible");
                return;
            }
            if (task() == CompositionTask::Motion && !frame.gestures.empty() && !sample->wrist)
                throw std::invalid_argument("motion requires source image dimensions");
            if (!clip_.empty()) {
                if (seconds <= clip_.back().seconds) {
                    reset_window("camera time reversed/repeated - check capture timestamps"); return;
                }
                if (seconds-clip_.back().seconds > model_.options().maximum_gap_seconds) {
                    reset_window("frame gap too long - check camera/inference speed"); return;
                }
            }
            if (frame.gestures.empty()) {
                status = std::string(kNoHandStatus) + " - show full hand in good light";
                if (clip_.empty()) { record_start_.reset(); return; }
                if (seconds-last_seen_ > model_.options().maximum_gap_seconds) {
                    reset_window("hand lost too long - show full hand, improve lighting"); return;
                }
                status += " | short gap encoded as missing";
            } else {
                const auto side = frame.landmarks.hands.front().handedness;
                if (task() != CompositionTask::Legacy && last_side_ && *last_side_ != side) {
                    reset_window("hand side changed - record with the same hand");
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
                    status += " | READY " + std::string(composition_name(label_)) + " in " +
                        std::to_string(int(std::ceil(*record_start_-seconds))) + "s; hold START pose";
                    return;
                }
            }
            if (clip_.size() >= model_.options().maximum_samples) {
                reset_window("sample limit reached - shorten configured window"); return;
            }
            clip_.push_back(*sample);
            if (collecting_) {
                constexpr int kPercent = 100;
                const int progress = int(std::min(double(kPercent), kPercent * (seconds-clip_.front().seconds) / model_.options().duration_seconds));
                status += " | REC " + std::string(composition_name(label_)) + " " + std::to_string(std::min(progress, kPercent)) + "%";
            }
            if (seconds-clip_.front().seconds < model_.options().duration_seconds) return;
            if (collecting_) {
                if (test_) {
                    if (heldout_.size() >= model_.options().maximum_clips) throw std::runtime_error("test clip capacity reached");
                    (void)model_.encode(clip_);
                    heldout_.push_back({clip_, label_, batch_, unix_ms()});
                } else {
                    if (training_.size() >= model_.options().maximum_clips) throw std::runtime_error("training clip capacity reached");
                    training_.push_back({clip_, label_, batch_, unix_ms()});
                    try { model_.add_training(clip_, label_); }
                    catch (...) { training_.pop_back(); throw; }
                }
                dirty_ = true;
                if (!test_) { prediction = "ESN: UNTRAINED"; ++revision_; }
                feedback = test_ ? "Test clip saved in RAM" : "Training clip saved in RAM; press T to train readout";
                status = "ESN: window complete";
                collecting_ = false;
            } else {
                const auto result = model_.predict(clip_);
                prediction = std::string(task() == CompositionTask::Legacy ? "Last window: " : "Window candidate: ") + composition_name(result.label);
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
        const char* side = hand.handedness == Handedness::Left ? "Left" :
            hand.handedness == Handedness::Right ? "Right" : "Unknown";
        std::ostringstream message;
        message << "1 hand " << side << " | " << gesture_display_name(gesture.label)
                << " " << std::fixed << std::setprecision(2) << gesture.scores[std::size_t(gesture.label)];
        return message.str();
    }
    void evaluate() {
        if (!model_.trained()) throw std::runtime_error("train before evaluating");
        std::array<int, kCompositionClasses> counts{};
        for (const auto& clip : heldout_) ++counts[composition_index(task(), clip.label)];
        for (int i = 0; i < class_count(); ++i) if (counts[i] < model_.options().minimum_clips_per_class)
            throw std::runtime_error(task() == CompositionTask::Motion ?
                "need held-out Neutral and Wave clips (keys 3/4)" : "need held-out clips for every class (keys 3/4/5)");
        int correct = 0;
        std::array<std::array<int, kCompositionClasses>, kCompositionClasses> confusion{};
        for (const auto& clip : heldout_) {
            const auto prediction = model_.predict(clip.samples);
            ++confusion[composition_index(task(), clip.label)][composition_index(task(), prediction.label)];
            if (prediction.label == clip.label) ++correct;
        }
        std::cout << "Held-out confusion rows=true columns=predicted [" << class_names() << "]\n";
        for (int row = 0; row < class_count(); ++row) {
            for (int column = 0; column < class_count(); ++column)
                std::cout << (column ? " " : "") << confusion[row][column];
            std::cout << '\n';
        }
        feedback = "Held-out correct=" + std::to_string(correct) + "/" + std::to_string(heldout_.size()) + "; matrix in console";
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
