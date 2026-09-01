#include "yolo_domain_capture.hpp"
#include "yolo_domain_cli.hpp"
#include "yolo_domain_frame.hpp"
#include "yolo_domain_profile.hpp"
#include "yolo_domain_ui.hpp"
#include "yolo_domain_window.hpp"

#include "kfcore/yolo/tracking.hpp"

#include "kfcore/yolo/onnx.hpp"
#include "kfcore/yolo/tensorrt.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace
{

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using kfcore::yolo::DetectionFrame;
using kfcore::yolo::ImageView;
using kfcore::yolo::OnnxDetector;
using kfcore::yolo::OnnxDetectorOptions;
using kfcore::image::BgrImage;
using namespace kfcore::yolo::demo;

constexpr std::size_t kMaximumImagePaths = 100000U;
constexpr std::size_t kTimingWindowCapacity = 300U;
constexpr std::uint64_t kMetricsRefreshFrames = 30U;
constexpr auto kCaptureTimeout = std::chrono::milliseconds(5000);
constexpr std::size_t kMaximumDecodedImageBytes = 64U * 1024U * 1024U;
constexpr const char* kWindowName = "KFCore YOLOv8 Domain Applications";

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error(message);
}

double elapsed_ms(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<std::string> argument_values(int argc, char** argv)
{
    std::vector<std::string> result;
    result.reserve(static_cast<std::size_t>((std::max)(argc, 0)));
    for (int index = 0; index < argc; ++index)
    {
        result.emplace_back(argv[index] == nullptr ? "" : argv[index]);
    }
    return result;
}

BackendAvailability compiled_backends() noexcept
{
    return { true, true };
}

const char* format_name(int format) noexcept
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
        return "MJPEG (unsupported)";
    default:
        return "unknown";
    }
}

double mode_fps(const turbo_video_native_mode_t& mode) noexcept
{
    if (mode.framerate_denominator == 0U)
    {
        return 0.0;
    }
    return static_cast<double>(mode.framerate_numerator) /
           static_cast<double>(mode.framerate_denominator);
}

void print_cameras()
{
    const auto devices = list_camera_devices();
    if (devices.empty())
    {
        std::cout << "No video capture devices found.\n";
        return;
    }
    for (std::size_t index = 0U; index < devices.size(); ++index)
    {
        const auto& device = devices[index];
        std::cout << '[' << index << "] " << device.name
                  << (device.is_default != 0 ? " (default)" : "")
                  << "\n  id=" << device.id << '\n';
        const auto modes = list_camera_modes(device.id);
        for (const auto& mode : modes)
        {
            std::cout << "  mode=" << mode.mode_id << ' ' << mode.width << 'x'
                      << mode.height << '@' << std::fixed << std::setprecision(2)
                      << mode_fps(mode) << ' ' << format_name(mode.format) << '\n';
        }
    }
}

void require_regular_model(const fs::path& path)
{
    std::error_code error;
    if (!path.is_absolute() || !fs::is_regular_file(path, error) || error)
    {
        fail("configured model must be an existing absolute regular file: " +
             path.string());
    }
}

class Detector final
{
public:
    static Detector load(const Arguments& arguments, double& load_ms)
    {
        require_regular_model(arguments.model);
        const auto start = Clock::now();
        Detector result;
        if (*arguments.backend == Backend::Cpu)
        {
            OnnxDetectorOptions options;
            options.intra_op_threads = arguments.intra_op_threads;
            options.inter_op_threads = arguments.inter_op_threads;
            options.mirror_horizontal = arguments.mirror;
            result.cpu_ = OnnxDetector::load(arguments.model, options);
        }
        else
        {
            result.engine_ = kfcore::yolo::Engine::load(arguments.model);
            kfcore::yolo::DetectorOptions options;
            options.mirror_horizontal = arguments.mirror;
            result.tensorrt_ = result.engine_->create_detector(options);
        }
        load_ms = elapsed_ms(start, Clock::now());
        return result;
    }

    DetectionFrame detect(const ImageView& image)
    {
        if (cpu_ != nullptr)
        {
            return cpu_->detect(image);
        }
        if (tensorrt_ != nullptr)
        {
            return tensorrt_->detect(image);
        }
        fail("detector has no initialized backend");
    }

private:
    std::unique_ptr<OnnxDetector> cpu_;
    std::shared_ptr<const kfcore::yolo::Engine> engine_;
    std::unique_ptr<kfcore::yolo::TensorRtDetector> tensorrt_;
};

std::string lower_extension(const fs::path& path)
{
    std::string value = path.extension().string();
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character)
    {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool is_supported_image(const fs::path& path)
{
    static const std::vector<std::string> extensions {
        ".bmp", ".jpeg", ".jpg", ".png"
    };
    const std::string extension = lower_extension(path);
    return std::find(extensions.begin(), extensions.end(), extension) !=
           extensions.end();
}

void require_directory(const fs::path& path, const char* option)
{
    std::error_code error;
    if (!path.is_absolute() || !fs::is_directory(path, error) || error)
    {
        fail(std::string(option) + " must be an existing absolute directory: " +
             path.string());
    }
}

std::vector<fs::path> image_paths(const fs::path& directory,
                                  std::optional<std::uint64_t> max_frames)
{
    std::vector<fs::path> result;
    std::error_code error;
    fs::directory_iterator iterator(directory, error);
    if (error)
    {
        fail("cannot enumerate --images directory: " + directory.string());
    }
    for (const auto& entry : iterator)
    {
        const bool regular = entry.is_regular_file(error);
        if (error)
        {
            fail("cannot inspect image entry: " + entry.path().string());
        }
        if (regular && is_supported_image(entry.path()))
        {
            if (result.size() == kMaximumImagePaths)
            {
                fail("image directory exceeds the application path limit");
            }
            result.push_back(entry.path());
        }
    }
    std::sort(result.begin(), result.end(), [](const fs::path& left, const fs::path& right)
    {
        return left.generic_u8string() < right.generic_u8string();
    });
    if (result.empty())
    {
        fail("--images contains no supported image files: " + directory.string());
    }
    if (max_frames.has_value() && *max_frames < result.size())
    {
        result.resize(static_cast<std::size_t>(*max_frames));
    }
    return result;
}

void prepare_output_directory(const fs::path& input, const fs::path& output)
{
    if (!output.is_absolute())
    {
        fail("--output must be an absolute directory path");
    }
    std::error_code error;
    if (fs::exists(output, error))
    {
        if (error || !fs::is_directory(output, error) || error)
        {
            fail("--output must be a directory: " + output.string());
        }
    }
    else if (!fs::create_directories(output, error) || error)
    {
        fail("cannot create --output directory: " + output.string());
    }
    if (fs::equivalent(input, output, error) && !error)
    {
        fail("--images and --output must be different directories");
    }
    if (error)
    {
        fail("cannot compare --images and --output directories");
    }
}

struct ProcessingState
{
    Detector&                    detector;
    kfcore::yolo::ByteTrackSession tracker;
    const DomainProfile&        profile;
    const Arguments&            arguments;
    TimingWindow                timings { kTimingWindowCapacity };
    MetricsSnapshot             metrics;
    double                      model_load_ms = 0.0;
    std::uint64_t               frame_count = 0U;
};

struct ProcessedFrame
{
    kfcore::yolo::TrackFrame tracks;
    DomainSummary            summary;
    FrameTimings             timings;
};

ProcessedFrame infer_and_track(ProcessingState& state, const ImageView& image,
                               double capture_wait_ms, double convert_ms)
{
    ProcessedFrame result;
    result.timings.capture_wait_ms = capture_wait_ms;
    result.timings.convert_ms = convert_ms;

    const auto detect_start = Clock::now();
    DetectionFrame detections = state.detector.detect(image);
    detections = filter_detections(detections, state.profile,
                                   state.arguments.score_threshold);
    const auto track_start = Clock::now();
    result.timings.detect_ms = elapsed_ms(detect_start, track_start);
    result.tracks = state.tracker.update(detections);
    result.summary = summarize_tracks(result.tracks, state.profile);
    result.timings.track_ms = elapsed_ms(track_start, Clock::now());
    return result;
}

OverlayState overlay_state(const ProcessingState& state,
                           const ProcessedFrame& frame,
                           const CaptureCounters& capture)
{
    OverlayState overlay;
    overlay.profile = &state.profile;
    overlay.tracks = &frame.tracks;
    overlay.summary = &frame.summary;
    overlay.backend = backend_name(*state.arguments.backend);
    overlay.model_load_ms = state.model_load_ms;
    overlay.metrics = state.metrics;
    overlay.capture = capture;
    const double total = state.metrics.total.current_ms;
    overlay.fps = total > 0.0 ? 1000.0 / total : 0.0;
    return overlay;
}

void finish_timing(ProcessingState& state, ProcessedFrame& frame,
                   Clock::time_point render_start)
{
    const auto end = Clock::now();
    frame.timings.render_ms = elapsed_ms(render_start, end);
    frame.timings.total_ms = frame.timings.capture_wait_ms +
                             frame.timings.convert_ms + frame.timings.detect_ms +
                             frame.timings.track_ms + frame.timings.render_ms;
    state.timings.add(frame.timings);
    ++state.frame_count;
    if (state.frame_count == 1U ||
        state.frame_count % kMetricsRefreshFrames == 0U)
    {
        state.metrics = state.timings.snapshot();
    }
    else
    {
        update_current_metrics(state.metrics, frame.timings, state.timings.size());
    }
}

void print_final_metrics(const ProcessingState& state,
                         const CaptureCounters& capture)
{
    const MetricsSnapshot snapshot = state.timings.snapshot();
    std::cout << "application=" << state.profile.name
              << " backend=" << backend_name(*state.arguments.backend)
              << " frames=" << state.frame_count
              << " model_load=" << std::fixed << std::setprecision(2)
              << state.model_load_ms << "ms\n"
              << format_metrics(snapshot) << '\n'
              << "capture captured=" << capture.captured_frames
              << " consumed=" << capture.consumed_frames
              << " coalesced=" << capture.coalesced_frames
              << " rejected=" << capture.rejected_frames << '\n';
}

void run_images(const Arguments& arguments, ProcessingState& state)
{
    require_directory(*arguments.images, "--images");
    prepare_output_directory(*arguments.images, *arguments.output);
    const auto paths = image_paths(*arguments.images, arguments.max_frames);
    for (const auto& path : paths)
    {
        const auto convert_start = Clock::now();
        BgrImage image = load_bgr(path, kMaximumDecodedImageBytes,
                                  kMaximumDecodedImageBytes);
        const double convert_ms = elapsed_ms(convert_start, Clock::now());
        ProcessedFrame frame = infer_and_track(
            state, bgr_image_view(image), 0.0, convert_ms);
        if (arguments.mirror)
        {
            const auto mirror_start = Clock::now();
            mirror_bgr_horizontal(image);
            frame.timings.convert_ms += elapsed_ms(mirror_start, Clock::now());
        }
        const auto render_start = Clock::now();
        draw_overlay(image, overlay_state(state, frame, {}));
        const fs::path destination = *arguments.output / path.filename();
        save_bgr(destination, image, kMaximumDecodedImageBytes);
        finish_timing(state, frame, render_start);
    }
    print_final_metrics(state, {});
}

void run_camera(const Arguments& arguments, ProcessingState& state)
{
    const auto devices = list_camera_devices();
    const std::size_t device_index = static_cast<std::size_t>(arguments.capture.camera_index);
    if (device_index >= devices.size())
    {
        fail("--camera index is outside the enumerated device list");
    }
    const auto& device = devices[device_index];
    const auto modes = list_camera_modes(device.id);
    const auto& mode = modes[select_mode(modes, arguments.capture)];
    kfcore::yolo::ByteTrackOptions tracking_options;
    tracking_options.frame_rate = static_cast<float>(mode_fps(mode));
    state.tracker = kfcore::yolo::ByteTrackSession(tracking_options);
    std::cout << "camera=" << device.name << " mode=" << mode.mode_id << ' '
              << mode.width << 'x' << mode.height << '@' << std::fixed
              << std::setprecision(2) << mode_fps(mode) << ' '
              << format_name(mode.format) << '\n';

    LatestFrameMailbox mailbox(arguments.capture.max_frame_bytes);
    CapturedFrame captured = mailbox.make_consumer_frame();
    std::unique_ptr<NativeWindow> window;
    if (!arguments.headless)
    {
        window = std::make_unique<NativeWindow>(kWindowName, mode.width, mode.height);
    }
    CameraCapture camera(device.id, mode, mailbox);
    camera.start();
    while (!arguments.max_frames.has_value() ||
           state.frame_count < *arguments.max_frames)
    {
        const auto wait_start = Clock::now();
        const TakeStatus status = mailbox.take_latest(captured, kCaptureTimeout);
        const auto wait_end = Clock::now();
        if (status == TakeStatus::Timeout)
        {
            fail("camera produced no frame within the capture timeout");
        }
        if (status == TakeStatus::Closed)
        {
            fail("camera capture closed before the requested frame count");
        }
        std::optional<BgrImage> image;
        double convert_ms = 0.0;
        ImageView inference_view;
        if (captured.format == TURBO_VIDEO_CAPTURE_FORMAT_BGRA)
        {
            const auto convert_start = Clock::now();
            image = to_bgr(captured, kMaximumDecodedImageBytes);
            convert_ms = elapsed_ms(convert_start, Clock::now());
            inference_view = bgr_image_view(*image);
        }
        else
        {
            inference_view = capture_image_view(captured);
        }
        ProcessedFrame frame = infer_and_track(state, inference_view,
                                                elapsed_ms(wait_start, wait_end),
                                                convert_ms);
        if (!arguments.headless)
        {
            const auto convert_start = Clock::now();
            if (!image.has_value())
            {
                image = to_bgr(captured, kMaximumDecodedImageBytes);
            }
            if (arguments.mirror)
            {
                mirror_bgr_horizontal(*image);
            }
            frame.timings.convert_ms += elapsed_ms(convert_start, Clock::now());
        }
        const auto render_start = Clock::now();
        const CaptureCounters counters = mailbox.counters();
        KeyAction action = KeyAction::None;
        if (!arguments.headless)
        {
            draw_overlay(*image, overlay_state(state, frame, counters));
            window->present(*image);
            action = decode_key(window->poll_key());
        }
        finish_timing(state, frame, render_start);
        if (action == KeyAction::Reset)
        {
            state.tracker.reset();
        }
        else if (action == KeyAction::Quit)
        {
            break;
        }
    }
    camera.stop();
    mailbox.close();
    print_final_metrics(state, mailbox.counters());
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Arguments arguments =
            parse_arguments(argument_values(argc, argv), compiled_backends());
        if (arguments.help)
        {
            std::cout << usage_text() << '\n';
            return 0;
        }
        if (arguments.list_cameras)
        {
            print_cameras();
            return 0;
        }

        double model_load_ms = 0.0;
        Detector detector = Detector::load(arguments, model_load_ms);
        kfcore::yolo::ByteTrackOptions tracking_options;
        tracking_options.frame_rate = static_cast<float>(arguments.capture.fps);
        ProcessingState state { detector,
                                kfcore::yolo::ByteTrackSession(tracking_options),
                                domain_profile(*arguments.application), arguments };
        state.model_load_ms = model_load_ms;
        if (arguments.images.has_value())
        {
            run_images(arguments, state);
        }
        else
        {
            run_camera(arguments, state);
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "yolov8_domain_demo: " << error.what() << '\n';
        return 1;
    }
}
