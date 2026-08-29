#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace kfcore::yolo {

enum class PixelFormat { Bgr8, Rgb8, Nv12, I420 };
enum class MemoryKind { Host, CudaDevice };

struct ImageView {
    const void* data = nullptr;
    std::int32_t width = 0;
    std::int32_t height = 0;
    std::size_t row_stride = 0;
    PixelFormat pixel_format = PixelFormat::Bgr8;
    MemoryKind memory_kind = MemoryKind::Host;
};

struct BoxF {
    float left;
    float top;
    float right;
    float bottom;
};

struct Detection {
    BoxF box;
    float score;
    std::int32_t class_id;
};

struct DetectionFrame {
    std::int32_t image_width;
    std::int32_t image_height;
    std::vector<Detection> detections;
};

struct TrackedDetection {
    Detection detection;
    std::optional<std::uint64_t> track_id;
};

struct TrackFrame {
    std::int32_t image_width;
    std::int32_t image_height;
    std::vector<TrackedDetection> detections;
};

}  // namespace kfcore::yolo
