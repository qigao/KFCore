#pragma once

#include <turbo_capture.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace kfcore::hand_interaction::demo
{

inline constexpr int         kDefaultCaptureWidth     = 640;
inline constexpr int         kDefaultCaptureHeight    = 480;
inline constexpr int         kDefaultCaptureFps       = 30;
inline constexpr std::size_t kDefaultMaxFrameBytes    = 64U * 1024U * 1024U;

enum class Backend
{
    Cpu,
    TensorRt,
};

struct BackendAvailability
{
    bool cpu      = false;
    bool tensorrt = false;
};

struct CaptureRequest
{
    int                          camera_index    = 0;
    int                          width           = kDefaultCaptureWidth;
    int                          height          = kDefaultCaptureHeight;
    int                          fps             = kDefaultCaptureFps;
    std::optional<std::uint64_t> mode_id;
    std::size_t                  max_frame_bytes = kDefaultMaxFrameBytes;
};

struct Arguments
{
    bool                         list_cameras                 = false;
    bool                         wave_require_horizontal_axis = false;
    std::optional<Backend>       backend;
    std::optional<std::uint64_t> max_frames;
    CaptureRequest               capture;
    float                        face_detection_score_threshold = 0.50F;
    float                        face_landmark_score_threshold = 0.50F;
};

[[nodiscard]] Arguments parse_arguments(
    const std::vector<std::string>& values, BackendAvailability availability);

// Returns the first processable camera-provided mode matching the exact mode id
// or geometry and frame rate. The device enumeration owns pixel-format order.
[[nodiscard]] std::size_t select_mode(const std::vector<turbo_video_native_mode_t>& modes,
                                      const CaptureRequest& request);

} // namespace kfcore::hand_interaction::demo
