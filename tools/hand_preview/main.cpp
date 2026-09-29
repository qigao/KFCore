#include <kfcore/mediapipe/hand_landmarker.hpp>
#include <kfcore/mediapipe/gesture_recognizer.hpp>
#include <kfcore/gesture_interaction/interaction.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#define NOMINMAX
#include <windows.h>
#include <gdiplus.h>
#include "camera.hpp"
#include "composition_session.hpp"
#include "experiment_dialog.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
constexpr char kWindow[] = "KFCore | MediaPipe 手势训练";
constexpr int kCanvasWidth = 1100, kCanvasHeight = 960;
constexpr int kMargin = 24, kImageTop = 130, kImageHeight = 542;
constexpr int kLineHeight = 24, kBoneThickness = 2, kPointRadius = 3;
constexpr double kTextScale = 0.55, kTitleScale = 0.8, kMilliseconds = 1000.0;
constexpr int kEscape = 27, kMaxCamera = 64;
constexpr std::size_t kChannels = 3;
const cv::Scalar kBackground(24, 24, 24), kText(235, 235, 235), kMuted(180, 180, 180);
constexpr std::array<std::array<int, 2>, 21> kBones{{
    {0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},
    {5,9},{9,10},{10,11},{11,12},{9,13},{13,14},{14,15},{15,16},
    {13,17},{0,17},{17,18},{18,19},{19,20}}};

struct Options {
    std::string backend, palm, landmark, image, snapshot, embedder, classifier;
    int camera = -1, frames = 0, mode = 0, timeout_seconds = 10;
    std::size_t max_bytes = preview::kMaxFrameBytes;
    bool headless = false;
    bool esn = false;
    bool actions = false;
    kfcore::gesture_interaction::CompositionOptions composition;
};

int number(const std::string& value) {
    int result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < 0)
        throw std::invalid_argument("需要非负整数：" + value);
    return result;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--headless") { o.headless = true; continue; }
        if (key == "--esn") { o.esn = true; continue; }
        if (key == "--actions") { o.esn = true; o.actions = true; continue; }
        if (i + 1 == argc) throw std::invalid_argument("缺少参数值：" + key);
        const std::string value = argv[++i];
        if (key == "--backend") o.backend = value;
        else if (key == "--palm") o.palm = value;
        else if (key == "--landmark") o.landmark = value;
        else if (key == "--gesture-embedder") o.embedder = value;
        else if (key == "--gesture-classifier") o.classifier = value;
        else if (key == "--image") o.image = value;
        else if (key == "--camera") o.camera = number(value);
        else if (key == "--frames") o.frames = number(value);
        else if (key == "--mode") o.mode = number(value);
        else if (key == "--timeout") o.timeout_seconds = number(value);
        else if (key == "--max-frame-bytes") o.max_bytes = number(value);
        else if (key == "--esn-window-ms") o.composition.duration_seconds = number(value) / kMilliseconds;
        else if (key == "--esn-gap-ms") o.composition.maximum_gap_seconds = number(value) / kMilliseconds;
        else if (key == "--esn-min-clips") o.composition.minimum_clips_per_class = number(value);
        else if (key == "--snapshot") o.snapshot = value;
        else throw std::invalid_argument("未知参数：" + key);
    }
    if (o.backend.empty() || o.palm.empty() || o.landmark.empty())
        throw std::invalid_argument("必须提供 --backend、--palm 和 --landmark；请查看 --help");
    if ((o.camera >= 0) == !o.image.empty() || o.camera > kMaxCamera)
        throw std::invalid_argument("必须且只能指定 --image 路径或 --camera 编号（0..64）");
    if (o.timeout_seconds == 0 || o.max_bytes == 0)
        throw std::invalid_argument("超时和最大帧字节数必须大于零");
    if (!o.image.empty() && o.frames > 1)
        throw std::invalid_argument("图片输入只有一帧");
    if (o.headless && o.frames == 0)
        throw std::invalid_argument("--headless 需要正数 --frames 上限");
    if (o.esn && (o.headless || o.camera < 0))
        throw std::invalid_argument("--esn 需要交互式相机会话");
    if (o.embedder.empty() != o.classifier.empty() || (o.esn && o.embedder.empty()))
        throw std::invalid_argument("手势模式需要 --gesture-embedder 和 --gesture-classifier；ESN 需要手势模式");
    if (!o.snapshot.empty() && (std::filesystem::exists(o.snapshot) ||
        std::filesystem::path(o.snapshot).extension() != ".png"))
        throw std::invalid_argument("--snapshot 必须指向新的 .png 文件（不可覆盖）");
    return o;
}

class ChineseTextRenderer {
public:
    ChineseTextRenderer() {
        Gdiplus::GdiplusStartupInput input;
        if (Gdiplus::GdiplusStartup(&token_, &input, nullptr) != Gdiplus::Ok)
            throw std::runtime_error("中文字体初始化失败");
    }
    ~ChineseTextRenderer() { Gdiplus::GdiplusShutdown(token_); }
    ChineseTextRenderer(const ChineseTextRenderer&) = delete;
    ChineseTextRenderer& operator=(const ChineseTextRenderer&) = delete;

    void draw(cv::Mat& canvas, const std::string& value, cv::Point at,
              const cv::Scalar& color, double scale) const {
        if (value.empty()) return;
        if (canvas.type() != CV_8UC3 || !canvas.isContinuous() ||
            canvas.step > std::size_t((std::numeric_limits<int>::max)()))
            throw std::runtime_error("文字画布格式无效");
        const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                               int(value.size()), nullptr, 0);
        if (count <= 0) throw std::runtime_error("界面文字不是 UTF-8");
        std::wstring wide(std::size_t(count), L'\0');
        if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                int(value.size()), wide.data(), count) != count)
            throw std::runtime_error("界面文字转换失败");
        Gdiplus::Bitmap bitmap(canvas.cols, canvas.rows, int(canvas.step),
                               PixelFormat24bppRGB, canvas.data);
        Gdiplus::Graphics graphics(&bitmap);
        const float height = float(18.0 * scale / kTextScale);
        Gdiplus::FontFamily family(L"Microsoft YaHei");
        if (!family.IsAvailable()) throw std::runtime_error("缺少微软雅黑字体");
        Gdiplus::Font font(&family, height, Gdiplus::FontStyleRegular, Gdiplus::UnitPixel);
        Gdiplus::SolidBrush brush(Gdiplus::Color(255, BYTE(color[2]), BYTE(color[1]), BYTE(color[0])));
        graphics.SetTextRenderingHint(Gdiplus::TextRenderingHintAntiAliasGridFit);
        if (bitmap.GetLastStatus() != Gdiplus::Ok || graphics.DrawString(
                wide.data(), count, &font, Gdiplus::PointF(float(at.x), float(at.y)-height), &brush) != Gdiplus::Ok)
            throw std::runtime_error("中文文字绘制失败");
    }
private:
    ULONG_PTR token_ = 0;
};

void text(cv::Mat& canvas, const std::string& value, cv::Point at,
          const cv::Scalar& color = kText, double scale = kTextScale) {
    static const ChineseTextRenderer renderer;
    renderer.draw(canvas, value, at, color, scale);
}

// Clip before integer conversion: out-of-frame model coordinates are valid,
// while non-finite values must never reach OpenCV's drawing primitives.
cv::Point point(float x, float y, const cv::Rect& area, double scale, bool mirror, int width) {
        if (!std::isfinite(x) || !std::isfinite(y)) throw std::runtime_error("手部关键点坐标不是有限数值");
    const double px = mirror ? width - 1.0 - x : x;
    return {static_cast<int>(std::clamp(area.x + px * scale, double(area.x), double(area.x + area.width - 1))),
            static_cast<int>(std::clamp(area.y + y * scale, double(area.y), double(area.y + area.height - 1)))};
}

cv::Mat render(const cv::Mat& source, const kfcore::hand_models::HandFrame& result,
               bool paused, bool ended, bool mirror, bool overlay, int frame_index,
               double position_ms, const std::string& source_name) {
    cv::Mat canvas(kCanvasHeight, kCanvasWidth, CV_8UC3, kBackground);
    text(canvas, "MEDIAPIPE / 手势训练", {kMargin, 36}, kText, kTitleScale);
    const std::string state = ended ? "图片" : paused ? "已暂停" : "运行中";
    text(canvas, state + "｜ONNX CPU｜手数：" + std::to_string(result.hands.size()) +
         "｜推理：" + cv::format("%.1f ms", result.timings.total_ms), {kMargin, 68});
    text(canvas, "来源：" + source_name.substr(0, 100), {kMargin, 92}, kMuted);
    text(canvas, "第 " + std::to_string(frame_index) + " 帧｜" + cv::format("%.2f 秒", position_ms / kMilliseconds) +
         "｜镜像：" + (mirror ? "开" : "关"), {kMargin, 116}, kMuted);
    const double scale = std::min(double(kCanvasWidth - 2 * kMargin) / source.cols,
                                  double(kImageHeight) / source.rows);
    const cv::Size size(std::max(1, int(source.cols * scale)), std::max(1, int(source.rows * scale)));
    const cv::Rect area((kCanvasWidth - size.width) / 2, kImageTop + (kImageHeight - size.height) / 2,
                        size.width, size.height);
    cv::Mat preview;
    if (mirror) cv::flip(source, preview, 1); else preview = source;
    cv::resize(preview, canvas(area), size);
    if (overlay) for (const auto& hand : result.hands) {
        const bool left = hand.handedness == kfcore::hand_models::Handedness::Left;
        const cv::Scalar color = left ? cv::Scalar(120, 235, 130) : cv::Scalar(240, 190, 110);
        std::array<cv::Point, kfcore::hand_models::kHandLandmarkCount> points;
        for (std::size_t i = 0; i < points.size(); ++i)
            points[i] = point(hand.landmarks[i].x, hand.landmarks[i].y, area, scale, mirror, source.cols);
        for (const auto& bone : kBones) cv::line(canvas, points[bone[0]], points[bone[1]], color, kBoneThickness, cv::LINE_AA);
        for (const auto& p : points) cv::circle(canvas, p, kPointRadius, kText, cv::FILLED, cv::LINE_AA);
        const auto& box = hand.palm.box;
        cv::rectangle(canvas, point(box.x, box.y, area, scale, mirror, source.cols),
            point(box.x + box.width, box.y + box.height, area, scale, mirror, source.cols), color, 1);
        const char* label = hand.handedness == kfcore::hand_models::Handedness::Unknown ? "未知手" : left ? "左手" : "右手";
        text(canvas, std::string(label) + cv::format(" %.2f", hand.landmark_confidence), points[0], color);
    }
    constexpr int kControlsTop = 704;
    text(canvas, "空格：暂停/继续    N：单步取帧    ESC：退出", {kMargin, kControlsTop});
    text(canvas, "M：镜像    O：显示/隐藏标注｜镜像仅影响显示", {kMargin, kControlsTop + kLineHeight}, kMuted);
    return canvas;
}

class Window {
public:
    explicit Window(bool enabled) : enabled_(enabled) {
        if (enabled_) cv::namedWindow(kWindow, cv::WINDOW_AUTOSIZE);
    }
    ~Window() { if (enabled_) { try { cv::destroyWindow(kWindow); } catch (...) {} } }
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
private:
    bool enabled_;
};

// Mouse and keyboard dispatch the same commands on the UI thread.
struct TrainingControls {
    static constexpr int kPanelTop = 839, kPanelHeight = 111, kPanelGap = 16;
    static constexpr int kPanelWidth = (kCanvasWidth - 2*kMargin - kPanelGap)/2;
    static constexpr int kInset = 12, kButtonGap = 8, kButtonWidth = 159;
    static constexpr int kSampleTop = 867, kActionTop = 908, kButtonHeight = 34;
    static constexpr std::array<int, 14> kKeys{'0','1','2','3','4','5','t','v','c',' ','s','l','g','w'};
    static const cv::Scalar kTrainFill, kTestFill, kTrainAccent, kTestAccent;
    static const cv::Scalar kTrainButton, kTestButton, kButtonFill;
    int class_count = 0;
    bool actions = false;
    int pending = -1;
    static cv::Rect area(int index) {
        constexpr int kFileLeft = 740, kFileTop = 14, kFileWidth = 168;
        if (index >= 12) return {380+(index-12)*(kFileWidth+kButtonGap), kFileTop, kFileWidth, kButtonHeight};
        if (index >= 10) return {kFileLeft+(index-10)*(kFileWidth+kButtonGap), kFileTop, kFileWidth, kButtonHeight};
        const int left = index < 3 || index == 6 || index == 8 ? kMargin : kMargin+kPanelWidth+kPanelGap;
        const int content_left = left+kInset;
        if (index < 6) return {content_left+(index%3)*(kButtonWidth+kButtonGap),
            kSampleTop, kButtonWidth, kButtonHeight};
        return {content_left+(index == 8 || index == 9 ? 2*(kButtonWidth+kButtonGap) : 0),
            kActionTop, index == 8 || index == 9 ? kButtonWidth : 2*kButtonWidth+kButtonGap,
            kButtonHeight};
    }
    static void mouse(int event, int x, int y, int, void* user) {
        if (event != cv::EVENT_LBUTTONUP) return;
        auto& controls = *static_cast<TrainingControls*>(user);
        for (int i = 0; i < int(kKeys.size()); ++i)
            if ((i >= 6 || i%3 < controls.class_count) && (i < 12 || controls.actions) &&
                area(i).contains(cv::Point{x,y})) {
                controls.pending = kKeys[i]; break;
            }
    }
    static void button(cv::Mat& canvas, int index, const std::string& label,
                       const cv::Scalar& fill, const cv::Scalar& border) {
        const auto bounds = area(index);
        cv::rectangle(canvas, bounds, fill, cv::FILLED);
        cv::rectangle(canvas, bounds, border, 1);
        text(canvas, label, {bounds.x+9,bounds.y+22}, kText, 0.45);
    }
    void draw(cv::Mat& canvas, const preview::CompositionSession& session, bool motion_selected) const {
        const int test_left = kMargin+kPanelWidth+kPanelGap;
        const auto required = std::to_string(session.model().options().minimum_clips_per_class);
        cv::rectangle(canvas, {kMargin,kPanelTop,kPanelWidth,kPanelHeight}, kTrainFill, cv::FILLED);
        cv::rectangle(canvas, {test_left,kPanelTop,kPanelWidth,kPanelHeight}, kTestFill, cv::FILLED);
        text(canvas, "1 训练集", {kMargin+kInset,kPanelTop+20}, kTrainAccent, 0.53);
        text(canvas, "每类录 " + required + " 段，再点训练模型", {kMargin+145,kPanelTop+20}, kMuted, 0.42);
        text(canvas, "2 测试集", {test_left+kInset,kPanelTop+20}, kTestAccent, 0.53);
        text(canvas, "另录新片段，再点评估测试", {test_left+145,kPanelTop+20}, kMuted, 0.42);
        for (int i = 0; i < 6; ++i) {
            if (i%3 >= session.class_count()) continue;
            const char* name = session.label_name(i%3);
            if (session.task() == kfcore::gesture_interaction::CompositionTask::Legacy)
                name = i%3 == 0 ? "无组合" : i%3 == 1 ? "张握张" : "握张握";
            const auto label = std::to_string(i) + " 录" + name;
            button(canvas, i, label, kButtonFill, i < 3 ? kTrainAccent : kTestAccent);
        }
        button(canvas, 6, "T 训练模型", kTrainButton, kTrainAccent);
        button(canvas, 7, "V 评估测试", kTestButton, kTestAccent);
        button(canvas, 8, "C 取消录制", kButtonFill, kMuted);
        button(canvas, 9, "空格 暂停", kButtonFill, kMuted);
        button(canvas, 10, "S 另存为...", kButtonFill, kMuted);
        button(canvas, 11, "L 加载...", kButtonFill, kMuted);
        if (actions) {
            button(canvas, 12, "G 抓放模型", motion_selected ? kButtonFill : kTrainButton, kTrainAccent);
            button(canvas, 13, "W 挥手模型", motion_selected ? kTestButton : kButtonFill, kTestAccent);
        }
    }
};
const cv::Scalar TrainingControls::kTrainFill(46, 35, 27);
const cv::Scalar TrainingControls::kTestFill(34, 43, 30);
const cv::Scalar TrainingControls::kTrainAccent(210, 145, 75);
const cv::Scalar TrainingControls::kTestAccent(140, 195, 120);
const cv::Scalar TrainingControls::kTrainButton(145, 82, 35);
const cv::Scalar TrainingControls::kTestButton(65, 112, 45);
const cv::Scalar TrainingControls::kButtonFill(55, 55, 55);

std::string pipeline_identity(const Options& options) {
    using namespace kfcore::runtime;
    std::string identity = compute_model_artifact_sha256(options.backend);
    for (const auto& manifest : {options.palm, options.landmark, options.embedder, options.classifier}) {
        identity += compute_model_artifact_sha256(manifest);
        const auto package = ModelPackage::load(manifest);
        for (const auto& artifact : package.artifacts()) identity += compute_model_artifact_sha256(package.artifact_path(artifact));
    }
    if (options.actions) identity += "|actions-arbitrary-fist-terminal-v2";
    return identity;
}

void experiment_command(int key, const std::string& identity,
    std::unique_ptr<preview::CompositionSession>& interaction,
    std::unique_ptr<preview::CompositionSession>& motion,
    preview::CompositionSession*& selected) {
    try {
        if (interaction->recording() || (motion && motion->recording()))
            throw std::runtime_error("请先完成或取消录制，再保存或加载");
        const auto path = preview::choose_experiment_file(key == 's');
        if (!path) { selected->feedback = "文件操作已取消，实验未改变"; return; }
        if (key == 's') {
            kfcore::gesture_interaction::ExperimentArchive archive{identity, {interaction->archive()}};
            if (motion) archive.sessions.push_back(motion->archive());
            kfcore::gesture_interaction::save_experiment(*path, archive);
            interaction->mark_saved(); if (motion) motion->mark_saved();
            selected->feedback = "完整实验已保存：" + path->filename().u8string();
        } else {
            const auto archive = kfcore::gesture_interaction::load_experiment(*path, identity);
            if (archive.sessions.size() != (motion ? 2U : 1U))
                throw std::runtime_error("实验模式不匹配，请用相同的 --actions 或 --esn 重新启动");
            auto loaded_interaction = preview::CompositionSession::restore(archive.sessions[0]);
            auto loaded_motion = motion ? preview::CompositionSession::restore(archive.sessions[1]) : nullptr;
            if ((interaction->dirty() || (motion && motion->dirty())) && !preview::confirm_discard_experiment()) {
                selected->feedback = "加载已取消，当前实验保留"; return;
            }
            const bool selected_motion = selected == motion.get();
            // Both candidates are validated before this noexcept ownership commit.
            interaction.swap(loaded_interaction); motion.swap(loaded_motion);
            selected = selected_motion ? motion.get() : interaction.get();
            selected->feedback = "完整实验已加载：" + path->filename().u8string();
        }
        std::cout << "Experiment " << (key == 's' ? "saved: " : "loaded: ") << path->u8string() << '\n';
    } catch (const std::exception& error) {
        const std::string reason = error.what();
        selected->feedback = motion && reason == "MediaPipe pipeline identity mismatch" ?
            "实验操作失败：模型资源或抓放训练规则不匹配；旧抓放实验不能直接加载" :
            "实验操作失败：" + reason;
    }
}

int run(const Options& o) {
    using namespace kfcore;
    runtime::Runtime runtime;
    const auto backend = runtime.load_backend(o.backend);
    std::unique_ptr<mediapipe::HandLandmarker> detector;
    std::unique_ptr<mediapipe::GestureRecognizer> recognizer;
    const auto policy = runtime::ExecutionPolicy::exact("onnxruntime", "cpu");
    if (o.embedder.empty())
        detector = mediapipe::HandLandmarker::load(runtime, runtime::ModelPackage::load(o.palm),
            runtime::ModelPackage::load(o.landmark), policy);
    else
        recognizer = mediapipe::GestureRecognizer::load(runtime, runtime::ModelPackage::load(o.palm),
            runtime::ModelPackage::load(o.landmark), runtime::ModelPackage::load(o.embedder),
            runtime::ModelPackage::load(o.classifier), policy);
    std::unique_ptr<preview::Camera> camera;
    std::unique_ptr<preview::CompositionSession> composition;
    std::unique_ptr<preview::CompositionSession> motion;
    if (o.esn) {
        auto options = o.composition;
        if (o.actions) options.task = kfcore::gesture_interaction::CompositionTask::Interaction;
        composition = std::make_unique<preview::CompositionSession>(options);
        if (o.actions) {
            options.task = kfcore::gesture_interaction::CompositionTask::Motion;
            motion = std::make_unique<preview::CompositionSession>(options);
        }
    }
    auto* selected = composition.get();
    gesture_interaction::InteractionOptions event_options;
    event_options.maximum_gap_seconds = o.composition.maximum_gap_seconds;
    gesture_interaction::GestureInteraction interaction_events(event_options);
    std::string basic_event = "暂无基础手势事件", action_event = "暂无动作事件";
    const auto consume_events = [&](const std::vector<gesture_interaction::Event>& events) {
        for (const auto& event : events) {
            std::string message = preview::event_display_name(event.kind);
            if (event.gesture != mediapipe::CannedGesture::None)
                message += "：" + std::string(preview::gesture_display_name(event.gesture));
            message += "（" + std::string(preview::event_reason_display_name(event.reason)) + "）";
            if (event.kind == gesture_interaction::EventKind::GestureStarted ||
                event.kind == gesture_interaction::EventKind::GestureEnded ||
                event.kind == gesture_interaction::EventKind::GestureCancelled) basic_event = message;
            else action_event = message;
        }
    };
    const preview::CompositionSession* configured_interaction = nullptr;
    const preview::CompositionSession* configured_motion = nullptr;
    const gesture_interaction::CompositionEsn* configured_interaction_model = nullptr;
    const gesture_interaction::CompositionEsn* configured_motion_model = nullptr;
    std::uint64_t configured_interaction_revision = 0, configured_motion_revision = 0;
    double event_time = 0;
    const auto synchronize_models = [&] {
        if (!o.actions) return;
        const auto* interaction_model = composition->model().trained() && !composition->recording() ? &composition->model() : nullptr;
        const auto* motion_model = motion->model().trained() && !motion->recording() ? &motion->model() : nullptr;
        if (composition.get() != configured_interaction || interaction_model != configured_interaction_model ||
            composition->revision() != configured_interaction_revision || motion.get() != configured_motion ||
            motion_model != configured_motion_model || motion->revision() != configured_motion_revision) {
            consume_events(interaction_events.configure_sequences(
                {interaction_model, motion_model, composition->gate_options(), motion->gate_options()}, event_time));
            configured_interaction = composition.get(); configured_interaction_model = interaction_model;
            configured_interaction_revision = composition->revision();
            configured_motion = motion.get(); configured_motion_model = motion_model;
            configured_motion_revision = motion->revision();
        }
    };
    const std::string experiment_identity = o.esn ? pipeline_identity(o) : std::string{};
    if (o.camera >= 0) camera = std::make_unique<preview::Camera>(o.camera, o.mode, o.max_bytes);
    TrainingControls controls;
    controls.actions = o.actions;
    if (selected) controls.class_count = selected->class_count();
    Window window(!o.headless);
    if (composition) cv::setMouseCallback(kWindow, TrainingControls::mouse, &controls);
    cv::Mat source, canvas;
    hand_models::HandFrame result;
    std::vector<mediapipe::GesturePrediction> gestures;
    bool paused = false, ended = false, mirror = false, overlay = true, step = false;
    int processed = 0, with_hands = 0;
    double position_ms = 0;
    std::uint64_t timestamp = 0;
    auto last_frame = std::chrono::steady_clock::now();
    const std::string name = o.camera >= 0 ? "相机 " + std::to_string(o.camera) : std::filesystem::path(o.image).filename().u8string();
    while (true) {
        synchronize_models();
        if ((!paused || step) && !ended) {
            cv::Mat next;
            if (camera) camera->take(next, timestamp);
            else {
                next = cv::imread(o.image, cv::IMREAD_COLOR);
                if (next.empty()) throw std::runtime_error("无法读取图片：" + o.image);
                ended = true;
            }
            if (!next.empty()) {
                if (next.type() != CV_8UC3) throw std::runtime_error("来源图像必须解码为 BGR8");
                const image::ImageView view{next.data, (next.rows - 1) * next.step[0] + next.cols * kChannels,
                    next.cols, next.rows, next.step[0], image::PixelFormat::Bgr8, image::MemoryKind::Host};
                if (recognizer) {
                    auto recognized = recognizer->infer(view);
                    if (composition) composition->update(recognized, timestamp / (kMilliseconds*kMilliseconds), next.cols, next.rows);
                    if (motion) motion->update(recognized, timestamp / (kMilliseconds*kMilliseconds), next.cols, next.rows);
                    event_time = timestamp / (kMilliseconds*kMilliseconds);
                    // A single-step frame may be inspected, but cannot start a live interaction while paused.
                    if (!paused) consume_events(interaction_events.process(recognized, {event_time, next.cols, next.rows}));
                    gestures = std::move(recognized.gestures);
                    result = std::move(recognized.landmarks);
                } else result = detector->infer(view);
                source = std::move(next);
                ++processed;
                if (!result.hands.empty()) ++with_hands;
                position_ms = timestamp / kMilliseconds;
                last_frame = std::chrono::steady_clock::now();
                step = false;
            } else {
                const auto idle = std::chrono::steady_clock::now() - last_frame;
                if (std::chrono::duration<double>(idle).count() > event_options.maximum_gap_seconds)
                    consume_events(interaction_events.reset(event_time));
                if (idle > std::chrono::seconds(o.timeout_seconds))
                    throw std::runtime_error("相机取帧超时，请检查设备、模式和相机权限");
            }
        }
        if (!source.empty()) canvas = render(source, result, paused, ended, mirror, overlay, processed, position_ms, name);
        else {
            canvas = cv::Mat(kCanvasHeight, kCanvasWidth, CV_8UC3, kBackground);
            text(canvas, "等待相机画面……按 ESC 退出", {kMargin, 68});
        }
        if (recognizer && !composition && !source.empty()) {
            std::string labels = gestures.empty() ? preview::kNoHandStatus : "已检测到手｜手势：";
            for (const auto& gesture : gestures) labels += " " + std::string(preview::gesture_display_name(gesture.label));
            text(canvas, labels, {kMargin, 812});
        }
        if (recognizer) text(canvas, basic_event, {560, 68}, kText, 0.45);
        if (composition) {
            constexpr int kPredictionLeft = 560, kPredictionTop = 116;
            text(canvas, selected->prediction, {kPredictionLeft, kPredictionTop});
            if (motion) text(canvas, "事件：" + action_event,
                {kPredictionLeft, kPredictionTop-kLineHeight}, kText, 0.45);
            constexpr int kEsnTop = 758;
            text(canvas, selected->summary(), {kMargin, kEsnTop});
            text(canvas, selected->status, {kMargin, kEsnTop + kLineHeight}, kMuted);
            text(canvas, selected->feedback, {kMargin, kEsnTop + 2 * kLineHeight});
            const char* hint = !o.actions ? "先在左侧训练，再用右侧新片段测试；测试样本不参与训练" :
                selected == motion.get() ? "非挥手：有手但不做水平往返（可静止、单向或竖直移动）" :
                "抓取：任意手型→握拳→保持；放开：任意手型→握拳→张掌；非动作：无完整序列";
            text(canvas, hint, {kMargin, 827}, kMuted, 0.47);
            controls.draw(canvas, *selected, selected == motion.get());
        }
        if (!o.headless) cv::imshow(kWindow, canvas);
        if ((o.frames > 0 && processed >= o.frames) || (o.headless && ended)) break;
        constexpr int kPollMilliseconds = 10;
        if (o.headless) { std::this_thread::sleep_for(std::chrono::milliseconds(kPollMilliseconds)); continue; }
        const int keyboard = preview::normalize_key(cv::waitKeyEx(kPollMilliseconds));
        const int key = controls.pending >= 0 ? std::exchange(controls.pending, -1) : keyboard;
        if (key == kEscape || cv::getWindowProperty(kWindow, cv::WND_PROP_VISIBLE) < 1) break;
        if (key == ' ' && !ended) { paused = !paused; last_frame = std::chrono::steady_clock::now(); }
        if (key == 'n' && !ended) { paused = true; step = true; last_frame = std::chrono::steady_clock::now(); }
        if (key == 'm') mirror = !mirror;
        if (key == 'o') overlay = !overlay;
        if (key == ' ' || key == 'n' || key == 's' || key == 'l' || key == 'c')
            consume_events(interaction_events.reset(event_time));
        if (composition) {
            if (key == 's' || key == 'l') {
                experiment_command(key, experiment_identity, composition, motion, selected);
                // File dialogs block UI capture consumption; restart the timeout baseline.
                last_frame = std::chrono::steady_clock::now();
            } else if (key == ' ' || key == 'n') {
                composition->reset_window("画面暂停或单步，时间窗已重置");
                if (motion) motion->reset_window("画面暂停或单步，时间窗已重置");
            } else if (o.actions && (key == 'g' || key == 'w')) {
                if (composition->recording() || motion->recording())
                    selected->feedback = "请先完成或取消录制，再切换模型";
                else {
                    selected = key == 'g' ? composition.get() : motion.get();
                    controls.class_count = selected->class_count();
                }
            } else if (paused && key >= '0' && key <= '5') selected->reset_window("请先继续预览，再开始录制");
            else selected->key(key);
        }
    }
    consume_events(interaction_events.reset(event_time));
    if (!o.snapshot.empty() && !cv::imwrite(o.snapshot, canvas)) throw std::runtime_error("画面截图写入失败");
    std::cout << "frames=" << processed << " frames_with_hands=" << with_hands
              << " replaced=" << (camera ? camera->replaced() : 0)
              << " backend=onnxruntime device=cpu classifier=" << (recognizer ? "mediapipe-canned" : "not-loaded") << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "kfhand_preview --backend 插件 --palm 清单 --landmark 清单\n"
                     "  (--image 路径 | --camera 编号) [--mode 编号] [--frames 帧数]\n"
                     "  [--headless] [--snapshot 新文件.png] [--timeout 秒数] [--max-frame-bytes 字节数]\n"
                     "  --list-cameras | --list-modes 相机编号\n"
                     "  --gesture-embedder 清单 --gesture-classifier 清单（需要三维手部关键点模型）\n"
                     "  --esn [--esn-window-ms 3000] [--esn-gap-ms 250] [--esn-min-clips 3]\n"
                     "  --actions：G 抓放模型，W 挥手模型；两支分别训练、测试。\n"
                     "  抓放：0/1/2 训练非动作/抓取/放开；3/4/5 测试。\n"
                     "  挥手：0/1 训练非挥手/挥手；3/4 测试；2/5 不使用；C 取消。\n"
                     "  ESN：0/1/2 录训练片段；3/4/5 录测试片段；T 训练；V 评估。\n"
                     "  S 另存完整实验为新的 .kfesn；L 加载（恢复样本和模型，清空实时历史）。\n"
                     "空格暂停；N 单步；M 镜像；O 显示标注；ESC 退出。\n";
        return 0;
    }
    try {
        if (argc == 2 && std::string(argv[1]) == "--list-cameras") { preview::Camera::list(); return 0; }
        if (argc == 3 && std::string(argv[1]) == "--list-modes") { preview::Camera::list(number(argv[2])); return 0; }
        return run(parse(argc, argv));
    }
    catch (const std::exception& e) {
        std::cerr << "手势训练工具失败：" << e.what() << '\n';
        return 1;
    }
}
