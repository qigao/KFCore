#include "hand_interaction_demo_capture.hpp"
#include "hand_interaction_demo_cli.hpp"
#include "hand_interaction_demo_frame.hpp"
#include "hand_interaction_demo_face.hpp"
#include "hand_interaction_demo_ui.hpp"

#include "kfcore/hand_interaction/hand_interaction.hpp"
#include "kfcore/face_models/core.hpp"
#include "kfcore/hand_models/core.hpp"
#include "kfcore/hand_models/cpu.hpp"
#include "kfcore/hand_models/tensorrt.hpp"

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
using kfcore::hand_models::HandFrame;
using kfcore::hand_models::HandInferenceBackend;
using kfcore::hand_models::HandPipeline;
using kfcore::hand_models::HandPipelineOptions;
using kfcore::face_models::FaceMeshFrame;

constexpr char kWindowTitle[] = "KFCore THIG Hand Interaction + FaceMesh";
constexpr char kHandInteractionGraph[] = "hand_interaction_cycle";
constexpr char kWaveGraph[] = "wave_cycle";
constexpr char kScreenClickGraph[] = "screen_click_cycle";
constexpr std::chrono::milliseconds kFrameWait { 50 };
constexpr double kFpsSmoothingAlpha = 0.15;

demo::BackendAvailability backend_availability()
{
    return { true, true };
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
        return kfcore::hand_models::CpuHandBackend::load(
            { arguments.palm_model, arguments.hand_model, arguments.classifier_model });
    }

    return kfcore::hand_models::TensorRtHandBackend::load(
        { arguments.palm_model, arguments.hand_model, arguments.classifier_model });
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
    HandPipelineOptions hand_pipeline_options;
    hand_pipeline_options.appearance.enabled = true;
    auto       model_pipeline = HandPipeline::create(
        make_backend(arguments), hand_pipeline_options);
    auto       face_pipeline = demo::make_face_pipeline(arguments);
    std::unique_ptr<kfcore::hand_models::TensorRtHandInput> tensor_rt_input;
    if (arguments.backend == demo::Backend::TensorRt)
    {
        tensor_rt_input =
            kfcore::hand_models::TensorRtHandInput::create();
    }
    const auto model_load_finished = std::chrono::steady_clock::now();
    const auto model_load_ms = std::chrono::duration<double, std::milli>(
                                   model_load_finished - model_load_started)
                                   .count();
    std::cout << std::fixed << std::setprecision(2) << "Model load: "
              << model_load_ms << " ms\n";
    HandInteractionOptions interaction_options;
    interaction_options.temporal.wave_require_horizontal_palm_axis =
        arguments.wave_require_horizontal_axis;
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
    std::vector<demo::HandIdentityDiagnostic> previous_identity_diagnostics;
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
        cv::Mat inference_bgr;
        kfcore::image::ImageView source_view;
        if (demo::inference_view_compatible(captured.format))
        {
            source_view = demo::inference_view(captured);
        }
        else
        {
            inference_bgr = demo::to_bgr(captured);
            source_view = image_view(inference_bgr);
        }
        auto frame_view =
            kfcore::image::FrameView::borrow(source_view);
        if (tensor_rt_input)
        {
            frame_view = tensor_rt_input->prepare(source_view);
        }
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
        const std::vector<demo::HandIdentityDiagnostic> identity_diagnostics =
            demo::make_identity_diagnostics(hands, interaction);
        if (identity_diagnostics != previous_identity_diagnostics)
        {
            std::cout << demo::format_identity_diagnostic_line(
                             captured.serial, hands, interaction)
                      << std::endl;
            previous_identity_diagnostics = identity_diagnostics;
        }

        demo::DemoThigStatus thig_status;
        thig_status.recent_actions =
            action_history.update(interaction.actions, thig_finished);
        thig_status.hand_state = interaction_pipeline.graph_state(kHandInteractionGraph);
        thig_status.wave_state = interaction_pipeline.graph_state(kWaveGraph);
        thig_status.click_state = interaction_pipeline.graph_state(kScreenClickGraph);

        const auto display_conversion_started = std::chrono::steady_clock::now();
        const cv::Mat bgr = inference_bgr.empty() ? demo::to_bgr(captured) : inference_bgr;
        const auto converted = std::chrono::steady_clock::now();

        demo::DemoMetrics metrics;
        metrics.capture = mailbox.counters();
        metrics.model   = hands.timings;
        metrics.face    = face.timings;
        metrics.fps     = displayed_fps;
        metrics.convert_ms =
            std::chrono::duration<double, std::milli>(
                converted - display_conversion_started).count();
        metrics.thig_ms =
            std::chrono::duration<double, std::milli>(thig_finished - thig_started).count();
        metrics.frame_ms =
            std::chrono::duration<double, std::milli>(converted - frame_started).count();
        cv::imshow(kWindowTitle, demo::compose_overlay(
            bgr, hands, interaction, thig_status,
            face_pipeline ? &face : nullptr, metrics));

        const demo::DemoAction action = demo::action_from_key(cv::waitKey(1));
        if (action == demo::DemoAction::Reset)
        {
            model_pipeline->reset();
            interaction_pipeline.reset();
            action_history.reset();
            previous_identity_diagnostics.clear();
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
        return run(demo::parse_arguments_from_environment(values,
                                                          backend_availability()));
    }
    catch (const std::exception& error)
    {
        std::cerr << "hand_interaction_demo: " << error.what() << '\n';
        return 1;
    }
}
