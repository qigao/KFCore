#pragma once

#include "kfcore/yolo/error.hpp"
#include "kfcore/yolo/types.hpp"

#include <cstddef>
#include <memory>

namespace kfcore::yolo {

struct ByteTrackOptions {
    int lost_track_buffer = 30;
    float frame_rate = 30.0f;
    float track_activation_threshold = 0.7f;
    int minimum_consecutive_frames = 2;
    float minimum_iou_threshold = 0.1f;
    float high_conf_det_threshold = 0.6f;
    std::size_t max_detections_per_frame = 300;
    std::size_t max_class_trackers = 128;
};

class ByteTrackSession final {
public:
    explicit ByteTrackSession(ByteTrackOptions options = {});
    ~ByteTrackSession();
    ByteTrackSession(ByteTrackSession&&) noexcept;
    ByteTrackSession& operator=(ByteTrackSession&&) noexcept;
    ByteTrackSession(const ByteTrackSession&) = delete;
    ByteTrackSession& operator=(const ByteTrackSession&) = delete;

    TrackFrame update(const DetectionFrame& frame);

    [[nodiscard]] std::uint64_t tracking_epoch() const noexcept;

    // Discards every class tracker and advances the identity epoch. Throws if
    // the 64-bit epoch space is exhausted rather than allowing ID aliasing.
    [[nodiscard]] std::uint64_t reset();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace kfcore::yolo
