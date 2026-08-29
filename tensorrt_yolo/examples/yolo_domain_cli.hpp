#pragma once

#include "yolo_domain_profile.hpp"

#include <turbo_capture.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace kfcore::yolo::demo
{

enum class Backend
{
    Cpu,
    TensorRt,
};

struct BackendAvailability
{
    bool cpu = false;
    bool tensorrt = false;
};

struct CaptureRequest
{
    int                          camera_index = 0;
    std::optional<std::uint64_t> mode_id;
    int                          width = 640;
    int                          height = 480;
    int                          fps = 30;
    std::size_t                  max_frame_bytes = 64U * 1024U * 1024U;
};

struct Arguments
{
    bool                         help = false;
    bool                         list_cameras = false;
    std::optional<DomainKind>    application;
    std::optional<Backend>       backend;
    std::filesystem::path        model;
    std::optional<std::filesystem::path> images;
    std::optional<std::filesystem::path> output;
    CaptureRequest               capture;
    std::optional<std::uint64_t> max_frames;
    float                        score_threshold = 0.25F;
    bool                         mirror = false;
    bool                         headless = false;
    int                          intra_op_threads = 0;
    int                          inter_op_threads = 0;
};

[[nodiscard]] Arguments parse_arguments(const std::vector<std::string>& values,
                                        BackendAvailability availability);
[[nodiscard]] std::size_t select_mode(
    const std::vector<turbo_video_native_mode_t>& modes,
    const CaptureRequest& request);
[[nodiscard]] const char* backend_name(Backend backend) noexcept;
[[nodiscard]] std::string usage_text();

} // namespace kfcore::yolo::demo
