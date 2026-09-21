#include <kfcore/mediapipe/hand_landmarker.hpp>
#include <kfcore/mediapipe/gesture_recognizer.hpp>
#include <kfcore/gesture_interaction/interaction.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
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
constexpr char kWindow[] = "KFCore | MediaPipe Hand Preview";
constexpr int kCanvasWidth = 1100, kCanvasHeight = 960;
constexpr int kMargin = 24, kImageTop = 130, kImageHeight = 542;
constexpr int kLineHeight = 24, kTextThickness = 1, kBoneThickness = 2, kPointRadius = 3;
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
        throw std::invalid_argument("expected a non-negative integer: " + value);
    return result;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--headless") { o.headless = true; continue; }
        if (key == "--esn") { o.esn = true; continue; }
        if (key == "--actions") { o.esn = true; o.actions = true; continue; }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
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
        else throw std::invalid_argument("unknown option: " + key);
    }
    if (o.backend.empty() || o.palm.empty() || o.landmark.empty())
        throw std::invalid_argument("--backend, --palm and --landmark are required; see --help");
    if ((o.camera >= 0) == !o.image.empty() || o.camera > kMaxCamera)
        throw std::invalid_argument("choose exactly one --image PATH or --camera INDEX (0..64)");
    if (o.timeout_seconds == 0 || o.max_bytes == 0)
        throw std::invalid_argument("timeout and max-frame-bytes must be positive");
    if (!o.image.empty() && o.frames > 1)
        throw std::invalid_argument("image input contains exactly one frame");
    if (o.headless && o.frames == 0)
        throw std::invalid_argument("--headless requires a positive --frames limit");
    if (o.esn && (o.headless || o.camera < 0))
        throw std::invalid_argument("--esn requires an interactive camera session");
    if (o.embedder.empty() != o.classifier.empty() || (o.esn && o.embedder.empty()))
        throw std::invalid_argument("gesture mode requires --gesture-embedder and --gesture-classifier; ESN requires gesture mode");
    if (!o.snapshot.empty() && (std::filesystem::exists(o.snapshot) ||
        std::filesystem::path(o.snapshot).extension() != ".png"))
        throw std::invalid_argument("--snapshot must be a new .png path (no overwrite)");
    return o;
}

void text(cv::Mat& canvas, const std::string& value, cv::Point at,
          const cv::Scalar& color = kText, double scale = kTextScale) {
    cv::putText(canvas, value, at, cv::FONT_HERSHEY_SIMPLEX, scale, color,
                kTextThickness, cv::LINE_AA);
}

// Clip before integer conversion: out-of-frame model coordinates are valid,
// while non-finite values must never reach OpenCV's drawing primitives.
cv::Point point(float x, float y, const cv::Rect& area, double scale, bool mirror, int width) {
    if (!std::isfinite(x) || !std::isfinite(y)) throw std::runtime_error("non-finite landmark");
    const double px = mirror ? width - 1.0 - x : x;
    return {static_cast<int>(std::clamp(area.x + px * scale, double(area.x), double(area.x + area.width - 1))),
            static_cast<int>(std::clamp(area.y + y * scale, double(area.y), double(area.y + area.height - 1)))};
}

cv::Mat render(const cv::Mat& source, const kfcore::hand_models::HandFrame& result,
               bool paused, bool ended, bool mirror, bool overlay, int frame_index,
               double position_ms, const std::string& source_name) {
    cv::Mat canvas(kCanvasHeight, kCanvasWidth, CV_8UC3, kBackground);
    text(canvas, "MEDIAPIPE / HAND PREVIEW", {kMargin, 36}, kText, kTitleScale);
    const std::string state = ended ? "IMAGE" : paused ? "PAUSED" : "RUNNING";
    text(canvas, state + "  |  ONNX CPU  |  Hands: " + std::to_string(result.hands.size()) +
         "  |  Infer: " + cv::format("%.1f ms", result.timings.total_ms), {kMargin, 68});
    text(canvas, "Source: " + source_name.substr(0, 100), {kMargin, 92}, kMuted);
    text(canvas, "Frame " + std::to_string(frame_index) + "  |  " + cv::format("%.2f s", position_ms / kMilliseconds) +
         "  |  Mirror: " + (mirror ? "ON" : "OFF"), {kMargin, 116}, kMuted);
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
        const char* label = hand.handedness == kfcore::hand_models::Handedness::Unknown ? "Unknown" : left ? "Left" : "Right";
        text(canvas, std::string(label) + cv::format(" %.2f", hand.landmark_confidence), points[0], color);
    }
    constexpr int kControlsTop = 704;
    text(canvas, "SPACE: pause/resume preview   N: next captured frame   ESC: exit", {kMargin, kControlsTop});
    text(canvas, "M: mirror preview   O: overlay on/off   |   Mirror affects display only", {kMargin, kControlsTop + kLineHeight}, kMuted);
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
    static constexpr int kTop = 866, kHeight = 34, kRowGap = 8, kWidth = 168, kGap = 8;
    static constexpr std::array<int, 14> kKeys{'0','1','2','3','4','5','g','w','t','v','c',' ','s','l'};
    int pending = -1;
    static cv::Rect area(int index) {
        constexpr int kFileButtons = 12, kFileLeft = 740, kFileTop = 14;
        if (index >= kFileButtons) return {kFileLeft+(index-kFileButtons)*(kWidth+kGap), kFileTop, kWidth, kHeight};
        constexpr int kColumns = 6;
        return {kMargin+(index%kColumns)*(kWidth+kGap), kTop+(index/kColumns)*(kHeight+kRowGap), kWidth, kHeight};
    }
    static void mouse(int event, int x, int y, int, void* user) {
        if (event != cv::EVENT_LBUTTONUP) return;
        auto& controls = *static_cast<TrainingControls*>(user);
        for (int i = 0; i < int(kKeys.size()); ++i)
            if (area(i).contains(cv::Point{x,y})) { controls.pending = kKeys[i]; break; }
    }
    void draw(cv::Mat& canvas, const preview::CompositionSession& session, bool actions) const {
        const std::array<std::string, 8> commands{
            actions ? "G: Grasp/Release" : "Legacy mode", actions ? "W: Wave" : "Legacy mode",
            "T: Train", "V: Evaluate", "C: Cancel", "SPACE: Pause", "S: Save as...", "L: Load..."};
        for (int i = 0; i < int(kKeys.size()); ++i) {
            if (i < 6 && i%3 >= session.class_count()) continue;
            const auto bounds = area(i);
            cv::rectangle(canvas, bounds, kMuted, 1);
            const std::string label = i < 6 ? std::string(i < 3 ? "Train " : "Test ") +
                (i%3 == 0 ? "Neutral" : session.label_name(i%3)) : commands[i-6];
            text(canvas, label, {bounds.x+8,bounds.y+22}, kText, 0.45);
        }
    }
};

std::string pipeline_identity(const Options& options) {
    using namespace kfcore::runtime;
    std::string identity = compute_model_artifact_sha256(options.backend);
    for (const auto& manifest : {options.palm, options.landmark, options.embedder, options.classifier}) {
        identity += compute_model_artifact_sha256(manifest);
        const auto package = ModelPackage::load(manifest);
        for (const auto& artifact : package.artifacts()) identity += compute_model_artifact_sha256(package.artifact_path(artifact));
    }
    return identity;
}

void experiment_command(int key, const std::string& identity,
    std::unique_ptr<preview::CompositionSession>& interaction,
    std::unique_ptr<preview::CompositionSession>& motion,
    preview::CompositionSession*& selected) {
    try {
        if (interaction->recording() || (motion && motion->recording()))
            throw std::runtime_error("Finish or cancel recording before saving/loading");
        const auto path = preview::choose_experiment_file(key == 's');
        if (!path) { selected->feedback = "File operation cancelled; experiment unchanged"; return; }
        if (key == 's') {
            kfcore::gesture_interaction::ExperimentArchive archive{identity, {interaction->archive()}};
            if (motion) archive.sessions.push_back(motion->archive());
            kfcore::gesture_interaction::save_experiment(*path, archive);
            interaction->mark_saved(); if (motion) motion->mark_saved();
            selected->feedback = "Saved complete experiment: " + path->filename().u8string();
        } else {
            const auto archive = kfcore::gesture_interaction::load_experiment(*path, identity);
            if (archive.sessions.size() != (motion ? 2U : 1U))
                throw std::runtime_error("Experiment mode differs; restart with matching --actions or --esn");
            auto loaded_interaction = preview::CompositionSession::restore(archive.sessions[0]);
            auto loaded_motion = motion ? preview::CompositionSession::restore(archive.sessions[1]) : nullptr;
            if ((interaction->dirty() || (motion && motion->dirty())) && !preview::confirm_discard_experiment()) {
                selected->feedback = "Load cancelled; current experiment retained"; return;
            }
            const bool selected_motion = selected == motion.get();
            // Both candidates are validated before this noexcept ownership commit.
            interaction.swap(loaded_interaction); motion.swap(loaded_motion);
            selected = selected_motion ? motion.get() : interaction.get();
            selected->feedback = "Loaded complete experiment: " + path->filename().u8string();
        }
        std::cout << "Experiment " << (key == 's' ? "saved: " : "loaded: ") << path->u8string() << '\n';
    } catch (const std::exception& error) { selected->feedback = std::string("Experiment: ")+error.what(); }
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
    std::string basic_event = "No basic gesture event", action_event = "No action event";
    const auto consume_events = [&](const std::vector<gesture_interaction::Event>& events) {
        for (const auto& event : events) {
            std::string message = gesture_interaction::event_name(event.kind);
            if (event.gesture != mediapipe::CannedGesture::None)
                message += ": " + std::string(mediapipe::gesture_name(event.gesture));
            message += " (" + std::string(gesture_interaction::event_reason_name(event.reason)) + ")";
            if (event.kind == gesture_interaction::EventKind::GestureStarted ||
                event.kind == gesture_interaction::EventKind::GestureEnded ||
                event.kind == gesture_interaction::EventKind::GestureCancelled) basic_event = message;
            else action_event = message;
        }
    };
    std::array<const preview::CompositionSession*, 2> configured_sessions{};
    std::array<const gesture_interaction::CompositionEsn*, 2> configured_models{};
    std::array<std::uint64_t, 2> configured_revisions{};
    double event_time = 0;
    const auto synchronize_models = [&] {
        if (!o.actions) return;
        const std::array<const preview::CompositionSession*, 2> sessions{composition.get(), motion.get()};
        std::array<const gesture_interaction::CompositionEsn*, 2> models{};
        std::array<std::uint64_t, 2> revisions{};
        for (std::size_t i = 0; i < sessions.size(); ++i) {
            revisions[i] = sessions[i]->revision();
            if (sessions[i]->model().trained() && !sessions[i]->recording()) models[i] = &sessions[i]->model();
        }
        if (sessions != configured_sessions || models != configured_models || revisions != configured_revisions) {
            consume_events(interaction_events.configure_sequences(
                {models[0], models[1], sessions[0]->gate_options(), sessions[1]->gate_options()}, event_time));
            configured_sessions = sessions; configured_models = models; configured_revisions = revisions;
        }
    };
    const std::string experiment_identity = o.esn ? pipeline_identity(o) : std::string{};
    if (o.camera >= 0) camera = std::make_unique<preview::Camera>(o.camera, o.mode, o.max_bytes);
    TrainingControls controls;
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
    const std::string name = o.camera >= 0 ? "Camera " + std::to_string(o.camera) : std::filesystem::path(o.image).filename().string();
    while (true) {
        synchronize_models();
        if ((!paused || step) && !ended) {
            cv::Mat next;
            if (camera) camera->take(next, timestamp);
            else {
                next = cv::imread(o.image, cv::IMREAD_COLOR);
                if (next.empty()) throw std::runtime_error("cannot read image: " + o.image);
                ended = true;
            }
            if (!next.empty()) {
                if (next.type() != CV_8UC3) throw std::runtime_error("source must decode as BGR8");
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
                    throw std::runtime_error("camera frame timeout; check device, mode and camera permissions");
            }
        }
        if (!source.empty()) canvas = render(source, result, paused, ended, mirror, overlay, processed, position_ms, name);
        else {
            canvas = cv::Mat(kCanvasHeight, kCanvasWidth, CV_8UC3, kBackground);
            text(canvas, "Waiting for camera frame... ESC to exit", {kMargin, 68});
        }
        if (recognizer && !composition && !source.empty()) {
            std::string labels = gestures.empty() ? preview::kNoHandStatus : "Hands detected | Gestures:";
            for (const auto& gesture : gestures) labels += " " + std::string(preview::gesture_display_name(gesture.label));
            text(canvas, labels, {kMargin, 812});
        }
        if (recognizer) text(canvas, basic_event, {560, 68}, kText, 0.45);
        if (composition) {
            constexpr int kPredictionLeft = 560, kPredictionTop = 116;
            text(canvas, selected->prediction, {kPredictionLeft, kPredictionTop});
            if (motion) text(canvas, "Events: " + action_event,
                {kPredictionLeft, kPredictionTop-kLineHeight}, kText, 0.45);
            constexpr int kEsnTop = 764;
            text(canvas, selected->summary(), {kMargin, kEsnTop});
            text(canvas, selected->status, {kMargin, kEsnTop + kLineHeight}, kMuted);
            text(canvas, selected->feedback, {kMargin, kEsnTop + 2 * kLineHeight});
            const char* hint = !o.actions ? "0/1/2 train OCO/COC; 3/4/5 test; S save / L load" :
                selected == motion.get() ? "Wave: 2+ horizontal cycles. Neutral: static, one-way sweeps. S: save complete experiment" :
                "Grasp: open -> fist; Release: fist -> open. Hold start/end. S: save complete experiment";
            text(canvas, hint, {kMargin, 836}, kMuted, 0.5);
            controls.draw(canvas, *selected, o.actions);
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
                composition->reset_window("preview paused/stepped; window reset");
                if (motion) motion->reset_window("preview paused/stepped; window reset");
            } else if (motion && (key == 'g' || key == 'w')) {
                if (selected->recording()) selected->feedback = "Finish recording or press C before switching branch";
                else selected = key == 'g' ? composition.get() : motion.get();
            } else if (paused && key >= '0' && key <= '5') selected->reset_window("resume preview before recording");
            else selected->key(key);
        }
    }
    consume_events(interaction_events.reset(event_time));
    if (!o.snapshot.empty() && !cv::imwrite(o.snapshot, canvas)) throw std::runtime_error("snapshot write failed");
    std::cout << "frames=" << processed << " frames_with_hands=" << with_hands
              << " replaced=" << (camera ? camera->replaced() : 0)
              << " backend=onnxruntime device=cpu classifier=" << (recognizer ? "mediapipe-canned" : "not-loaded") << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "kfhand_preview --backend PLUGIN --palm MANIFEST --landmark MANIFEST\n"
                     "  (--image PATH | --camera INDEX) [--mode INDEX] [--frames COUNT]\n"
                     "  [--headless] [--snapshot NEW.png] [--timeout SECONDS] [--max-frame-bytes COUNT]\n"
                     "  --list-cameras | --list-modes CAMERA\n"
                     "  --gesture-embedder MANIFEST --gesture-classifier MANIFEST (requires world landmark model)\n"
                     "  --esn [--esn-window-ms 3000] [--esn-gap-ms 250] [--esn-min-clips 3]\n"
                     "  --actions: parallel Grasp/Release and Wave ESNs. G/W select training branch.\n"
                     "  Wave: 0/1 train Neutral/Wave; 3/4 test. Keys 2/5 unused. C cancel.\n"
                     "  ESN: 0/1/2 train clips; 3/4/5 held-out clips; T fit; V evaluate.\n"
                     "  S save complete experiment as NEW .kfesn; L load (restores samples/model, resets live history).\n"
                     "SPACE pause preview; N step; M mirror; O overlay; ESC close.\n";
        return 0;
    }
    try {
        if (argc == 2 && std::string(argv[1]) == "--list-cameras") { preview::Camera::list(); return 0; }
        if (argc == 3 && std::string(argv[1]) == "--list-modes") { preview::Camera::list(number(argv[2])); return 0; }
        return run(parse(argc, argv));
    }
    catch (const std::exception& e) {
        std::cerr << "Hand preview failed: " << e.what() << '\n';
        return 1;
    }
}
