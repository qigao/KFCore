#include "hand_interaction_demo_cli.hpp"

#include <turbo_fs.h>

#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace kfcore::hand_interaction::demo
{
namespace
{

constexpr std::string_view kPalmRelative =
    "hand_gesture_model/palm_detection/palm_detection_full_inf_post_192x192.onnx";
constexpr std::string_view kHandRelative =
    "hand_gesture_model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx";
constexpr std::string_view kClassifierRelative =
    "hand_gesture_model/keypoint_classifier/keypoint_classifier.onnx";
constexpr std::string_view kTensorRtRelative = "hand_gesture_model/tensorrt";

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

void require_readable_file(const std::filesystem::path& path, const char* role)
{
    const std::string native = path.string();
    turbo_fs_stat_t   status {};
    if (turbo_fs_stat(native.c_str(), &status) != 0 || !status.is_file ||
        turbo_fs_access(native.c_str(), TURBO_FS_ACCESS_READ) != 0)
    {
        fail(std::string(role) + " is not a readable file: " + native);
    }
}

int format_rank(int format)
{
    switch (format)
    {
    case TURBO_VIDEO_CAPTURE_FORMAT_NV12:
        return 0;
    case TURBO_VIDEO_CAPTURE_FORMAT_I420:
        return 1;
    case TURBO_VIDEO_CAPTURE_FORMAT_BGRA:
        return 2;
    case TURBO_VIDEO_CAPTURE_FORMAT_RGB24:
        return 3;
    default:
        return -1;
    }
}

int rounded_fps(const turbo_video_native_mode_t& mode)
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
        fail("program name is missing");
    }

    Arguments                      result;
    std::unordered_set<std::string> seen;
    std::optional<std::string>      backend_value;
    std::optional<std::string>      model_dir;
    std::optional<std::string>      palm;
    std::optional<std::string>      hand;
    std::optional<std::string>      classifier;
    std::optional<std::string>      face_detector;
    std::optional<std::string>      face_landmark;
    bool                            face_score_set = false;
    bool                            face_landmark_score_set = false;
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
        else if (option == "--model-dir")
        {
            model_dir = value;
        }
        else if (option == "--palm")
        {
            palm = value;
        }
        else if (option == "--hand")
        {
            hand = value;
        }
        else if (option == "--classifier")
        {
            classifier = value;
        }
        else if (option == "--face-detector")
        {
            face_detector = value;
        }
        else if (option == "--facemesh")
        {
            face_landmark = value;
        }
        else if (option == "--face-score")
        {
            result.face_detection_score_threshold = parse_score(value, option);
            face_score_set = true;
        }
        else if (option == "--facemesh-score")
        {
            result.face_landmark_score_threshold = parse_score(value, option);
            face_landmark_score_set = true;
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

    if (face_detector.has_value() != face_landmark.has_value())
    {
        fail("--face-detector and --facemesh must be supplied together");
    }
    if ((face_score_set || face_landmark_score_set) && !face_detector.has_value())
    {
        fail("face score options require --face-detector and --facemesh");
    }
    if (face_detector.has_value())
    {
        result.face_detector_model = std::filesystem::path(*face_detector);
        result.face_landmark_model = std::filesystem::path(*face_landmark);
    }

    if (*result.backend == Backend::Cpu)
    {
        if (!model_dir.has_value())
        {
            if (model_root.empty())
            {
                fail("--model-dir is required when KFCORE_MODEL_ROOT is not set");
            }
            model_dir = model_root.string();
        }
        if (palm.has_value() || hand.has_value() || classifier.has_value())
        {
            fail("--palm, --hand, and --classifier are TensorRT-only options");
        }
        const std::filesystem::path root(*model_dir);
        result.palm_model       = root / kPalmRelative;
        result.hand_model       = root / kHandRelative;
        result.classifier_model = root / kClassifierRelative;
    }
    else
    {
        if (model_dir.has_value())
        {
            fail("--model-dir is a cpu-only option");
        }
        if (!palm.has_value() || !hand.has_value() || !classifier.has_value())
        {
            if (model_root.empty() || tensorrt_profile.empty())
            {
                fail("--palm, --hand, and --classifier are required when the "
                     "TensorRT model root/profile are not set");
            }
            const std::filesystem::path root =
                model_root / kTensorRtRelative / tensorrt_profile;
            if (!palm.has_value()) palm = (root / "palm_detection.engine").string();
            if (!hand.has_value()) hand = (root / "hand_landmark.engine").string();
            if (!classifier.has_value())
                classifier = (root / "keypoint_classifier.engine").string();
        }
        result.palm_model       = *palm;
        result.hand_model       = *hand;
        result.classifier_model = *classifier;
    }

    require_readable_file(result.palm_model, "palm model");
    require_readable_file(result.hand_model, "hand landmark model");
    require_readable_file(result.classifier_model, "classifier model");
    if (result.face_detector_model.has_value())
    {
        require_readable_file(*result.face_detector_model, "face detector model");
        require_readable_file(*result.face_landmark_model, "face landmark model");
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
    std::optional<std::size_t> selected;
    int                        selected_rank = (std::numeric_limits<int>::max)();
    for (std::size_t index = 0U; index < modes.size(); ++index)
    {
        const auto& current = modes[index];
        const int   rank    = format_rank(current.format);
        const bool  identity_matches =
            request.mode_id.has_value() && current.mode_id == *request.mode_id;
        const bool geometry_matches =
            !request.mode_id.has_value() && current.width == request.width &&
            current.height == request.height && rounded_fps(current) == request.fps;
        if ((identity_matches || geometry_matches) && rank >= 0 && rank < selected_rank)
        {
            selected      = index;
            selected_rank = rank;
        }
    }
    if (!selected.has_value())
    {
        fail(request.mode_id.has_value()
                 ? "requested mode id is absent or uses an unsupported format"
                 : "no exact uncompressed camera mode matches width, height, and fps");
    }
    return *selected;
}

} // namespace kfcore::hand_interaction::demo
