#include "hand_interaction_demo_cli.hpp"

#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace kfcore::hand_interaction::demo
{
namespace
{

[[noreturn]] void fail(const std::string& message)
{
    throw std::invalid_argument("hand_interaction_demo arguments: " + message);
}

template <typename Integer>
Integer parse_integer(const std::string& text, const std::string& option, Integer minimum)
{
    Integer value {};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc {} || parsed.ptr != text.data() + text.size() || value < minimum)
    {
        fail(option + " has an invalid numeric value: " + text);
    }
    return value;
}

float parse_score(const std::string& text, const std::string& option)
{
    float value = 0.0F;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc {} || parsed.ptr != text.data() + text.size() ||
        !std::isfinite(value) || value < 0.0F || value > 1.0F)
    {
        fail(option + " must be a finite value within [0,1]: " + text);
    }
    return value;
}

bool supported_capture_format(int format) noexcept
{
    switch (format)
    {
    case SALTS_VIDEO_CAPTURE_FORMAT_NV12:
    case SALTS_VIDEO_CAPTURE_FORMAT_I420:
    case SALTS_VIDEO_CAPTURE_FORMAT_BGRA:
    case SALTS_VIDEO_CAPTURE_FORMAT_RGB24:
        return true;
    default:
        return false;
    }
}

int rounded_fps(const salts_video_native_mode_t& mode)
{
    if (mode.framerate_denominator == 0U)
    {
        return 0;
    }
    const std::uint64_t rounded =
        (static_cast<std::uint64_t>(mode.framerate_numerator) +
         mode.framerate_denominator / 2U) /
        mode.framerate_denominator;
    return rounded <= static_cast<std::uint64_t>((std::numeric_limits<int>::max)())
               ? static_cast<int>(rounded)
               : 0;
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

} // namespace

Arguments parse_arguments(const std::vector<std::string>& values,
                          BackendAvailability availability)
{
    if (values.empty())
    {
        fail("program name is missing");
    }

    Arguments                      result;
    std::unordered_set<std::string> seen;
    std::optional<std::string>      backend_value;
    bool                            geometry_set = false;

    for (std::size_t index = 1U; index < values.size(); ++index)
    {
        const std::string& option = values[index];
        if (!seen.insert(option).second)
        {
            fail("duplicate option: " + option);
        }
        if (option == "--list-cameras")
        {
            result.list_cameras = true;
            continue;
        }
        if (option == "--wave-require-horizontal-axis")
        {
            result.wave_require_horizontal_axis = true;
            continue;
        }
        if (index + 1U >= values.size() || values[index + 1U].empty() ||
            values[index + 1U].rfind("--", 0U) == 0U)
        {
            fail("missing value for " + option);
        }
        const std::string& value = values[++index];
        if (option == "--backend")
        {
            backend_value = value;
        }
        else if (option == "--face-score")
        {
            result.face_detection_score_threshold = parse_score(value, option);
        }
        else if (option == "--facemesh-score")
        {
            result.face_landmark_score_threshold = parse_score(value, option);
        }
        else if (option == "--camera")
        {
            result.capture.camera_index = parse_integer<int>(value, option, 0);
        }
        else if (option == "--mode")
        {
            result.capture.mode_id = parse_integer<std::uint64_t>(value, option, 0U);
        }
        else if (option == "--width")
        {
            result.capture.width = parse_integer<int>(value, option, 1);
            geometry_set         = true;
        }
        else if (option == "--height")
        {
            result.capture.height = parse_integer<int>(value, option, 1);
            geometry_set          = true;
        }
        else if (option == "--fps")
        {
            result.capture.fps = parse_integer<int>(value, option, 1);
            geometry_set       = true;
        }
        else if (option == "--max-frame-bytes")
        {
            result.capture.max_frame_bytes =
                parse_integer<std::size_t>(value, option, 1U);
        }
        else if (option == "--max-frames")
        {
            result.max_frames = parse_integer<std::uint64_t>(value, option, 1U);
        }
        else
        {
            fail("unknown option: " + option);
        }
    }

    if (result.capture.mode_id.has_value() && geometry_set)
    {
        fail("--mode cannot be combined with --width, --height, or --fps");
    }
    if (result.list_cameras)
    {
        return result;
    }

    if (backend_value.has_value())
    {
        result.backend = parse_backend(*backend_value);
    }
    else if (availability.cpu != availability.tensorrt)
    {
        result.backend = availability.cpu ? Backend::Cpu : Backend::TensorRt;
    }
    else
    {
        fail(availability.cpu ? "--backend is required when both backends are available"
                              : "no hand inference backend was compiled");
    }

    if ((*result.backend == Backend::Cpu && !availability.cpu) ||
        (*result.backend == Backend::TensorRt && !availability.tensorrt))
    {
        fail("requested backend is not available in this build");
    }

    return result;
}

std::size_t select_mode(const std::vector<salts_video_native_mode_t>& modes,
                        const CaptureRequest& request)
{
    for (std::size_t index = 0U; index < modes.size(); ++index)
    {
        const auto& current = modes[index];
        const bool  identity_matches =
            request.mode_id.has_value() && current.mode_id == *request.mode_id;
        const bool geometry_matches =
            !request.mode_id.has_value() && current.width == request.width &&
            current.height == request.height && rounded_fps(current) == request.fps;
        if ((identity_matches || geometry_matches) &&
            supported_capture_format(current.format))
        {
            return index;
        }
    }
    fail(request.mode_id.has_value()
             ? "requested mode id is absent or uses an unsupported format"
             : "no exact supported camera mode matches width, height, and fps");
}

} // namespace kfcore::hand_interaction::demo
