#pragma once

#include <turbo_capture.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace kfcore::hand_interaction::demo
{

inline constexpr int         kDefaultCaptureWidth     = 1280;
inline constexpr int         kDefaultCaptureHeight    = 720;
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
    std::filesystem::path        palm_model;
    std::filesystem::path        hand_model;
    std::filesystem::path        classifier_model;
    std::optional<std::filesystem::path> face_detector_model;
    std::optional<std::filesystem::path> face_landmark_model;
    float                                face_detection_score_threshold = 0.50F;
    float                                face_landmark_score_threshold = 0.50F;
};

[[nodiscard]] Arguments parse_arguments(const std::vector<std::string>& values,
                                        BackendAvailability availability);

// Returns the index into modes. Selection is exact; it never changes geometry,
// frame rate, or compressed/uncompressed semantics behind the caller's back.
[[nodiscard]] std::size_t select_mode(const std::vector<turbo_video_native_mode_t>& modes,
                                      const CaptureRequest& request);

} // namespace kfcore::hand_interaction::demo
