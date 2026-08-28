#include "hand_interaction_demo_capture.hpp"
#include "hand_interaction_demo_cli.hpp"
#include "hand_interaction_demo_frame.hpp"
#include "hand_interaction_demo_face.hpp"
#include "hand_interaction_demo_ui.hpp"

#include "kfcore/hand_interaction/hand_interaction.hpp"
#include "kfcore/vision_models/core.hpp"
#if defined(KFCORE_HAND_DEMO_HAS_CPU)
#include "kfcore/vision_models/cpu.hpp"
#endif
#if defined(KFCORE_HAND_DEMO_HAS_TENSORRT)
#include "kfcore/vision_models/tensorrt.hpp"
#endif

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

namespace demo = kfcore::hand_interaction::demo;
using kfcore::hand_interaction::GestureFrameContext;
using kfcore::hand_interaction::HandInteractionFrame;
using kfcore::hand_interaction::HandInteractionOptions;
using kfcore::hand_interaction::HandInteractionPipeline;
using kfcore::vision_models::HandFrame;
using kfcore::vision_models::HandInferenceBackend;
using kfcore::vision_models::HandPipeline;
using kfcore::vision_models::FaceMeshFrame;

constexpr char kWindowTitle[] = "KFCore THIG Hand Interaction + FaceMesh";
constexpr char kHandInteractionGraph[] = "hand_interaction_cycle";
constexpr char kWaveGraph[] = "wave_cycle";
constexpr char kScreenClickGraph[] = "screen_click_cycle";
constexpr std::chrono::milliseconds kFrameWait { 50 };
constexpr double kFpsSmoothingAlpha = 0.15;

demo::BackendAvailability backend_availability()
{
    demo::BackendAvailability result;
#if defined(KFCORE_HAND_DEMO_HAS_CPU)
    result.cpu = true;
#endif
#if defined(KFCORE_HAND_DEMO_HAS_TENSORRT)
    result.tensorrt = true;
#endif
    return result;
}

const char* format_name(int format)
{
    switch (format)
    {
    case TURBO_VIDEO_CAPTURE_FORMAT_I420:
        return "I420";
    case TURBO_VIDEO_CAPTURE_FORMAT_NV12:
        return "NV12";
    case TURBO_VIDEO_CAPTURE_FORMAT_RGB24:
        return "RGB24";
    case TURBO_VIDEO_CAPTURE_FORMAT_BGRA:
        return "BGRA";
    case TURBO_VIDEO_CAPTURE_FORMAT_MJPEG:
        return "MJPEG (unsupported by demo)";
    default:
        return "unknown";
    }
}

void print_cameras()
{
    const auto devices = demo::list_camera_devices();
    if (devices.empty())
    {
        std::cout << "No video capture devices found.\n";
        return;
    }
    for (std::size_t device_index = 0U; device_index < devices.size(); ++device_index)
    {
        const auto& device = devices[device_index];
        std::cout << "camera " << device_index << ": " << device.name
                  << "\n  id: " << device.id << '\n';
        const auto modes = demo::list_camera_modes(device.id);
        for (const auto& mode : modes)
        {
            std::cout << "  mode " << mode.mode_id << ": " << mode.width << 'x'
                      << mode.height << '@' << turbo_video_mode_fps(&mode) << " "
                      << format_name(mode.format) << " [" << mode.framerate_numerator
                      << '/' << mode.framerate_denominator << "]\n";
        }
    }
}

std::unique_ptr<HandInferenceBackend> make_backend(const demo::Arguments& arguments)
{
    if (arguments.backend == demo::Backend::Cpu)
    {
#if defined(KFCORE_HAND_DEMO_HAS_CPU)
        return kfcore::vision_models::CpuHandBackend::load(
            { arguments.palm_model, arguments.hand_model, arguments.classifier_model });
#else
        throw std::logic_error("CPU backend was not compiled into this demo");
#endif
    }

#if defined(KFCORE_HAND_DEMO_HAS_TENSORRT)
    return kfcore::vision_models::TensorRtHandBackend::load(
        { arguments.palm_model, arguments.hand_model, arguments.classifier_model });
#else
    throw std::logic_error("TensorRT backend was not compiled into this demo");
#endif
}

kfcore::image::ImageView image_view(const cv::Mat& image)
{
    if (image.empty() || image.type() != CV_8UC3 || !image.isContinuous())
    {
        throw std::invalid_argument("demo inference image must be continuous CV_8UC3");
    }
    return { image.data,
             image.total() * image.elemSize(),
             image.cols,
             image.rows,
             image.step,
             kfcore::image::PixelFormat::Bgr8,
             kfcore::image::MemoryKind::Host };
}

class DemoWindow final
{
public:
    DemoWindow(int width, int height)
    {
        cv::namedWindow(kWindowTitle, cv::WINDOW_NORMAL);
        cv::resizeWindow(kWindowTitle, width, height);
        cv::Mat waiting(height, width, CV_8UC3, cv::Scalar(24, 24, 24));
        cv::putText(waiting, "Waiting for camera frames...", cv::Point(24, 48),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(235, 235, 235), 1,
                    cv::LINE_AA);
        cv::imshow(kWindowTitle, waiting);
        (void)cv::waitKey(1);
    }

    ~DemoWindow()
    {
        try
        {
            cv::destroyWindow(kWindowTitle);
        }
        catch (...)
        {
        }
    }

    [[nodiscard]] bool is_open() const
    {
        return cv::getWindowProperty(kWindowTitle, cv::WND_PROP_VISIBLE) > 0.0;
    }
};

int run(const demo::Arguments& arguments)
{
    if (arguments.list_cameras)
    {
        print_cameras();
        return 0;
    }

    const auto devices = demo::list_camera_devices();
    if (arguments.capture.camera_index < 0 ||
        static_cast<std::size_t>(arguments.capture.camera_index) >= devices.size())
    {
        throw std::invalid_argument("selected camera index is not present; use --list-cameras");
    }
    const auto& device = devices[static_cast<std::size_t>(arguments.capture.camera_index)];
    const auto modes = demo::list_camera_modes(device.id);
    const auto& mode = modes.at(demo::select_mode(modes, arguments.capture));

    const auto model_load_started = std::chrono::steady_clock::now();
    auto       model_pipeline = HandPipeline::create(make_backend(arguments));
    auto       face_pipeline = demo::make_face_pipeline(arguments);
    const auto model_load_finished = std::chrono::steady_clock::now();
    const auto model_load_ms = std::chrono::duration<double, std::milli>(
                                   model_load_finished - model_load_started)
                                   .count();
    std::cout << std::fixed << std::setprecision(2) << "Model load: "
              << model_load_ms << " ms\n";
    HandInteractionOptions interaction_options;
    interaction_options.temporal.wave_require_horizontal_palm_axis = true;
    HandInteractionPipeline interaction_pipeline(interaction_options);
    demo::RecentActionHistory action_history;
    demo::LatestFrameMailbox mailbox(arguments.capture.max_frame_bytes);
    demo::CapturedFrame      captured = mailbox.make_consumer_frame();
    demo::CameraCapture      camera(device.id, mode, mailbox);
    DemoWindow               window(mode.width, mode.height);

    std::cout << "Capturing " << device.name << " mode " << mode.mode_id << " ("
              << mode.width << 'x' << mode.height << '@' << turbo_video_mode_fps(&mode)
              << ' ' << format_name(mode.format) << ")\n";
    camera.start();
    bool running = true;
    std::uint64_t processed_frames = 0U;
    std::optional<std::chrono::steady_clock::time_point> previous_frame_started;
    double displayed_fps = 0.0;
    while (running && window.is_open())
    {
        const demo::TakeStatus status = mailbox.take_latest(captured, kFrameWait);
        if (status == demo::TakeStatus::Closed)
        {
            throw std::runtime_error("camera capture entered the error state");
        }
        if (status == demo::TakeStatus::Timeout)
        {
            running = demo::action_from_key(cv::waitKey(1)) != demo::DemoAction::Quit;
            continue;
        }

        const auto frame_started = std::chrono::steady_clock::now();
        if (previous_frame_started.has_value())
        {
            const double interval_seconds =
                std::chrono::duration<double>(frame_started - *previous_frame_started)
                    .count();
            if (interval_seconds > 0.0)
            {
                const double sampled_fps = 1.0 / interval_seconds;
                displayed_fps = displayed_fps > 0.0
                                    ? kFpsSmoothingAlpha * sampled_fps +
                                          (1.0 - kFpsSmoothingAlpha) * displayed_fps
                                    : sampled_fps;
            }
        }
        previous_frame_started = frame_started;
        const cv::Mat bgr = demo::to_bgr(captured);
        const auto converted = std::chrono::steady_clock::now();
        const kfcore::image::ImageView frame_view = image_view(bgr);
        HandFrame hands = model_pipeline->process(frame_view);
        FaceMeshFrame face;
        if (face_pipeline)
        {
            face = face_pipeline->process(frame_view);
        }
        const auto thig_started = std::chrono::steady_clock::now();
        const GestureFrameContext context {
            captured.serial, thig_started, captured.width, captured.height
        };
        HandInteractionFrame interaction = interaction_pipeline.process(hands, context);
        const auto thig_finished = std::chrono::steady_clock::now();

        demo::DemoThigStatus thig_status;
        thig_status.recent_actions =
            action_history.update(interaction.actions, thig_finished);
        thig_status.hand_state = interaction_pipeline.graph_state(kHandInteractionGraph);
        thig_status.wave_state = interaction_pipeline.graph_state(kWaveGraph);
        thig_status.click_state = interaction_pipeline.graph_state(kScreenClickGraph);

        demo::DemoMetrics metrics;
        metrics.capture = mailbox.counters();
        metrics.model   = hands.timings;
        metrics.face    = face.timings;
        metrics.fps     = displayed_fps;
        metrics.convert_ms =
            std::chrono::duration<double, std::milli>(converted - frame_started).count();
        metrics.thig_ms =
            std::chrono::duration<double, std::milli>(thig_finished - thig_started).count();
        metrics.frame_ms =
            std::chrono::duration<double, std::milli>(thig_finished - frame_started).count();
        cv::imshow(kWindowTitle, demo::compose_overlay(
            bgr, hands, interaction, thig_status,
            face_pipeline ? &face : nullptr, metrics));

        const demo::DemoAction action = demo::action_from_key(cv::waitKey(1));
        if (action == demo::DemoAction::Reset)
        {
            model_pipeline->reset();
            interaction_pipeline.reset();
            action_history.reset();
        }
        else if (action == demo::DemoAction::Quit)
        {
            running = false;
        }
        ++processed_frames;
        if (arguments.max_frames.has_value() &&
            processed_frames >= *arguments.max_frames)
        {
            running = false;
        }
    }
    camera.stop();
    const auto counters = mailbox.counters();
    std::cout << "Capture summary: captured=" << counters.captured_frames
              << " consumed=" << counters.consumed_frames
              << " coalesced=" << counters.coalesced_frames
              << " rejected=" << counters.rejected_frames << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::vector<std::string> values;
        values.reserve(static_cast<std::size_t>(argc));
        for (int index = 0; index < argc; ++index)
        {
            values.emplace_back(argv[index] != nullptr ? argv[index] : "");
        }
        return run(demo::parse_arguments(values, backend_availability()));
    }
    catch (const std::exception& error)
    {
        std::cerr << "hand_interaction_demo: " << error.what() << '\n';
        return 1;
    }
}
