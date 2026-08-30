#include "yolo_domain_cli.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace kfcore::yolo::demo
{
namespace
{

[[noreturn]] void fail(const std::string& message)
{
    throw std::invalid_argument(message);
}

template <typename Integer>
Integer parse_integer(const std::string& text, const std::string& option,
                      Integer minimum)
{
    static_assert(std::is_integral_v<Integer>);
    try
    {
        std::size_t consumed = 0U;
        if constexpr (std::is_signed_v<Integer>)
        {
            const long long value = std::stoll(text, &consumed, 10);
            if (consumed != text.size() || value < static_cast<long long>(minimum) ||
                value > static_cast<long long>((std::numeric_limits<Integer>::max)()))
            {
                fail(option + " is outside its valid range");
            }
            return static_cast<Integer>(value);
        }
        else
        {
            if (!text.empty() && text.front() == '-')
            {
                fail(option + " is outside its valid range");
            }
            const unsigned long long value = std::stoull(text, &consumed, 10);
            if (consumed != text.size() || value < static_cast<unsigned long long>(minimum) ||
                value > static_cast<unsigned long long>(
                            (std::numeric_limits<Integer>::max)()))
            {
                fail(option + " is outside its valid range");
            }
            return static_cast<Integer>(value);
        }
    }
    catch (const std::invalid_argument&)
    {
        fail(option + " must be an integer");
    }
    catch (const std::out_of_range&)
    {
        fail(option + " is outside its valid range");
    }
}

float parse_score(const std::string& text, const std::string& option)
{
    try
    {
        std::size_t consumed = 0U;
        const float value = std::stof(text, &consumed);
        if (consumed != text.size() || !std::isfinite(value) || value < 0.0F ||
            value > 1.0F)
        {
            fail(option + " must be finite within [0,1]");
        }
        return value;
    }
    catch (const std::invalid_argument&)
    {
        fail(option + " must be a floating-point value");
    }
    catch (const std::out_of_range&)
    {
        fail(option + " is outside its valid range");
    }
}

Backend parse_backend(const std::string& text)
{
    if (text == "cpu")
    {
        return Backend::Cpu;
    }
    if (text == "tensorrt")
    {
        return Backend::TensorRt;
    }
    fail("--backend must be cpu or tensorrt");
}

bool supported_format(int format) noexcept
{
    return format == TURBO_VIDEO_CAPTURE_FORMAT_NV12 ||
           format == TURBO_VIDEO_CAPTURE_FORMAT_I420 ||
           format == TURBO_VIDEO_CAPTURE_FORMAT_RGB24 ||
           format == TURBO_VIDEO_CAPTURE_FORMAT_BGRA;
}

int rounded_fps(const turbo_video_native_mode_t& mode)
{
    if (mode.framerate_denominator == 0U)
    {
        return 0;
    }
    return static_cast<int>(std::lround(
        static_cast<double>(mode.framerate_numerator) /
        static_cast<double>(mode.framerate_denominator)));
}

std::filesystem::path configured_model_path(
    DomainKind application, Backend backend, const std::filesystem::path& model_root,
    const std::string& tensorrt_profile)
{
    if (model_root.empty())
    {
        fail("--model is required when KFCORE_MODEL_ROOT is not set");
    }
    const std::string basename = "yolov8n-" + domain_profile(application).name;
    if (backend == Backend::Cpu)
    {
        return model_root / (basename + ".onnx");
    }
    if (tensorrt_profile.empty())
    {
        fail("--model is required when KFCORE_TENSORRT_ENGINE_PROFILE is not set");
    }
    return model_root / "tensorrt" / tensorrt_profile / (basename + ".engine");
}

std::string environment_value(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr ? value : "";
}

} // namespace

Arguments parse_arguments(const std::vector<std::string>& values,
                          BackendAvailability availability,
                          const std::filesystem::path& model_root,
                          const std::string& tensorrt_profile)
{
    if (values.empty())
    {
        fail("argument vector must include the executable name");
    }
    Arguments result;
    std::set<std::string> seen;
    bool camera_explicit = false;
    bool geometry_explicit = false;
    for (std::size_t index = 1U; index < values.size(); ++index)
    {
        const std::string& option = values[index];
        if (option == "--help" || option == "-h" || option == "--list-cameras" ||
            option == "--mirror" || option == "--headless")
        {
            if (!seen.insert(option).second)
            {
                fail("duplicate option: " + option);
            }
            if (option == "--help" || option == "-h") result.help = true;
            else if (option == "--list-cameras") result.list_cameras = true;
            else if (option == "--mirror") result.mirror = true;
            else result.headless = true;
            continue;
        }
        if (option.empty() || option.front() != '-')
        {
            fail("unexpected positional argument: " + option);
        }
        if (!seen.insert(option).second)
        {
            fail("duplicate option: " + option);
        }
        if (index + 1U >= values.size() || values[index + 1U].empty())
        {
            fail("missing value for " + option);
        }
        const std::string& value = values[++index];
        if (option == "--application")
            result.application = parse_domain_kind(value);
        else if (option == "--backend")
            result.backend = parse_backend(value);
        else if (option == "--model")
            result.model = value;
        else if (option == "--images")
            result.images = std::filesystem::path(value);
        else if (option == "--output")
            result.output = std::filesystem::path(value);
        else if (option == "--camera")
        {
            result.capture.camera_index = parse_integer<int>(value, option, 0);
            camera_explicit = true;
        }
        else if (option == "--mode")
            result.capture.mode_id = parse_integer<std::uint64_t>(value, option, 0U);
        else if (option == "--width")
        {
            result.capture.width = parse_integer<int>(value, option, 1);
            geometry_explicit = true;
        }
        else if (option == "--height")
        {
            result.capture.height = parse_integer<int>(value, option, 1);
            geometry_explicit = true;
        }
        else if (option == "--fps")
        {
            result.capture.fps = parse_integer<int>(value, option, 1);
            geometry_explicit = true;
        }
        else if (option == "--max-frame-bytes")
            result.capture.max_frame_bytes =
                parse_integer<std::size_t>(value, option, 1U);
        else if (option == "--max-frames")
            result.max_frames = parse_integer<std::uint64_t>(value, option, 1U);
        else if (option == "--score-threshold")
            result.score_threshold = parse_score(value, option);
        else if (option == "--intra-op-threads")
            result.intra_op_threads = parse_integer<int>(value, option, 0);
        else if (option == "--inter-op-threads")
            result.inter_op_threads = parse_integer<int>(value, option, 0);
        else
            fail("unknown option: " + option);
    }

    if (result.capture.mode_id.has_value() && geometry_explicit)
    {
        fail("--mode cannot be combined with --width, --height, or --fps");
    }
    if (result.help || result.list_cameras)
    {
        return result;
    }
    if (!result.application.has_value())
    {
        fail("--application is required");
    }
    if (!availability.cpu && !availability.tensorrt)
    {
        fail("no inference backend was compiled into the application");
    }
    if (!result.backend.has_value())
    {
        if (availability.cpu && availability.tensorrt)
        {
            fail("--backend is required when CPU and TensorRT are both compiled");
        }
        result.backend = availability.cpu ? Backend::Cpu : Backend::TensorRt;
    }
    if ((*result.backend == Backend::Cpu && !availability.cpu) ||
        (*result.backend == Backend::TensorRt && !availability.tensorrt))
    {
        fail(std::string(backend_name(*result.backend)) +
             " backend was not compiled into the application");
    }
    if (result.model.empty())
    {
        result.model = configured_model_path(*result.application, *result.backend,
                                             model_root, tensorrt_profile);
    }
    if (result.images.has_value())
    {
        if (!result.output.has_value())
        {
            fail("--output is required with --images");
        }
        if (camera_explicit || result.capture.mode_id.has_value() || geometry_explicit)
        {
            fail("camera selection options cannot be combined with --images");
        }
    }
    else if (result.output.has_value())
    {
        fail("--output requires --images");
    }
    if (!result.images.has_value() && result.headless && !result.max_frames.has_value())
    {
        fail("headless camera capture requires --max-frames");
    }
    return result;
}

Arguments parse_arguments_from_environment(const std::vector<std::string>& values,
                                           BackendAvailability availability)
{
    return parse_arguments(values, availability,
                           environment_value("KFCORE_MODEL_ROOT"),
                           environment_value("KFCORE_TENSORRT_ENGINE_PROFILE"));
}

std::size_t select_mode(const std::vector<turbo_video_native_mode_t>& modes,
                        const CaptureRequest& request)
{
    for (std::size_t index = 0U; index < modes.size(); ++index)
    {
        const turbo_video_native_mode_t& current = modes[index];
        if (request.mode_id.has_value())
        {
            if (current.mode_id == *request.mode_id && supported_format(current.format))
            {
                return index;
            }
        }
        else if (current.width == request.width && current.height == request.height &&
                 rounded_fps(current) == request.fps &&
                 current.format == TURBO_VIDEO_CAPTURE_FORMAT_NV12)
        {
            return index;
        }
    }
    fail(request.mode_id.has_value()
             ? "requested mode id is absent or uses an unsupported format"
             : "no exact NV12 camera mode matches width, height, and fps");
}

const char* backend_name(Backend backend) noexcept
{
    return backend == Backend::Cpu ? "cpu" : "tensorrt";
}

std::string usage_text()
{
    return
        "usage: yolov8_domain_demo --application <drone|football|parking> "
        "--backend <cpu|tensorrt> [--model <onnx|engine>] "
        "[--camera <index> [--mode <id> | --width 640 --height 480 --fps 30] | "
        "--images <dir> --output <dir>] [--score-threshold <0..1>] "
        "[--max-frames <count>] [--mirror] [--headless]\n"
        "       yolov8_domain_demo --list-cameras\n"
        "When --model is omitted, KFCORE_MODEL_ROOT and (for TensorRT) "
        "KFCORE_TENSORRT_ENGINE_PROFILE select the model.";
}

} // namespace kfcore::yolo::demo
