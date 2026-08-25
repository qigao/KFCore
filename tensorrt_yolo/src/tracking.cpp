#include "kfcore/yolo/tracking.hpp"

#include "checked_size.hpp"
#include "trackers/tracker.h"

#include <cmath>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::yolo {
namespace {

struct TrackerDeleter {
    void operator()(bytetrack_t* tracker) const noexcept {
        bytetrack_destroy(tracker);
    }
};

using TrackerOwner = std::unique_ptr<bytetrack_t, TrackerDeleter>;

struct DetectionGroup {
    std::vector<detection_t> detections;
    std::vector<std::size_t> original_indices;
    std::vector<tracked_detection_ex_t> tracked;
};

[[noreturn]] void throw_invalid_argument(const char* message) {
    throw YoloError(YoloErrorCode::InvalidArgument, message);
}

[[noreturn]] void throw_resource_limit(const char* message) {
    throw YoloError(YoloErrorCode::ResourceLimitExceeded, message);
}

void validate_unit_interval(float value, const char* name) {
    if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
        throw_invalid_argument(name);
    }
}

void validate_options(const ByteTrackOptions& options) {
    if (options.lost_track_buffer < 0) {
        throw_invalid_argument("lost_track_buffer must not be negative");
    }
    if (!std::isfinite(options.frame_rate) || options.frame_rate <= 0.0f) {
        throw_invalid_argument("frame_rate must be finite and positive");
    }
    if (options.minimum_consecutive_frames < 1) {
        throw_invalid_argument("minimum_consecutive_frames must be positive");
    }
    validate_unit_interval(options.track_activation_threshold,
                           "track_activation_threshold must be within [0, 1]");
    validate_unit_interval(options.minimum_iou_threshold,
                           "minimum_iou_threshold must be within [0, 1]");
    validate_unit_interval(options.high_conf_det_threshold,
                           "high_conf_det_threshold must be within [0, 1]");
    if (options.max_detections_per_frame == 0) {
        throw_invalid_argument("max_detections_per_frame must be positive");
    }
    if (options.max_class_trackers == 0) {
        throw_invalid_argument("max_class_trackers must be positive");
    }
}

bool is_finite(const Detection& detection) noexcept {
    return std::isfinite(detection.box.left) && std::isfinite(detection.box.top) &&
           std::isfinite(detection.box.right) && std::isfinite(detection.box.bottom) &&
           std::isfinite(detection.score);
}

detection_t to_tracker_detection(const Detection& detection) noexcept {
    detection_t result{};
    result.box = {detection.box.left, detection.box.top,
                  detection.box.right, detection.box.bottom};
    result.confidence = detection.score;
    result.has_confidence = 1;
    result.class_id = detection.class_id;
    result.has_class_id = 1;
    return result;
}

std::uint64_t global_track_id(std::int32_t class_id, int local_id) noexcept {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(class_id)) << 32U) |
           static_cast<std::uint32_t>(local_id);
}

YoloError tracker_error(tracker_status_t status) {
    switch (status) {
    case TRACKER_STATUS_INVALID_ARGUMENT:
        return YoloError(YoloErrorCode::InvalidArgument, "ByteTrack rejected the update arguments");
    case TRACKER_STATUS_CAPACITY:
    case TRACKER_STATUS_OVERFLOW:
        return YoloError(YoloErrorCode::ResourceLimitExceeded,
                         "ByteTrack exceeded a configured resource limit");
    case TRACKER_STATUS_ALLOCATION_FAILED:
        return YoloError(YoloErrorCode::TrackerAllocationFailure,
                         "ByteTrack could not allocate update storage");
    case TRACKER_STATUS_OK:
        break;
    }
    return YoloError(YoloErrorCode::EngineContractMismatch,
                     "ByteTrack returned an unknown status");
}

TrackerOwner make_tracker(const ByteTrackOptions& options) {
    const bytetrack_config_t config{
        options.lost_track_buffer,
        options.frame_rate,
        options.track_activation_threshold,
        options.minimum_consecutive_frames,
        options.minimum_iou_threshold,
        options.high_conf_det_threshold,
    };
    TrackerOwner tracker(bytetrack_create(&config));
    if (!tracker) {
        throw YoloError(YoloErrorCode::TrackerAllocationFailure,
                        "ByteTrack tracker allocation failed");
    }
    return tracker;
}

void require_scratch_size(std::size_t count, std::size_t element_size) {
    std::size_t bytes = 0;
    if (!detail::checked_multiply_size(count, element_size, &bytes)) {
        throw_resource_limit("tracking scratch storage size overflowed");
    }
}

}  // namespace

YoloError::YoloError(YoloErrorCode code, std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

YoloErrorCode YoloError::code() const noexcept {
    return code_;
}

class ByteTrackSession::Impl {
public:
    explicit Impl(ByteTrackOptions options_in) : options(options_in) {}

    ByteTrackOptions options;
    std::map<std::int32_t, TrackerOwner> trackers;
};

ByteTrackSession::ByteTrackSession(ByteTrackOptions options) {
    validate_options(options);
    try {
        impl_ = std::make_unique<Impl>(options);
    } catch (const std::bad_alloc&) {
        throw YoloError(YoloErrorCode::TrackerAllocationFailure,
                        "ByteTrack session allocation failed");
    }
}

ByteTrackSession::~ByteTrackSession() = default;
ByteTrackSession::ByteTrackSession(ByteTrackSession&&) noexcept = default;
ByteTrackSession& ByteTrackSession::operator=(ByteTrackSession&&) noexcept = default;

TrackFrame ByteTrackSession::update(const DetectionFrame& frame) {
    const std::size_t detection_count = frame.detections.size();
    if (detection_count > impl_->options.max_detections_per_frame) {
        throw_resource_limit("frame exceeds max_detections_per_frame");
    }
    require_scratch_size(detection_count, sizeof(TrackedDetection));
    require_scratch_size(detection_count, sizeof(detection_t));
    require_scratch_size(detection_count, sizeof(tracked_detection_ex_t));

    try {
        std::map<std::int32_t, DetectionGroup> groups;
        for (std::size_t index = 0; index < detection_count; ++index) {
            const Detection& detection = frame.detections[index];
            if (!is_finite(detection)) {
                throw_invalid_argument("detection coordinates and score must be finite");
            }
            DetectionGroup& group = groups[detection.class_id];
            group.detections.push_back(to_tracker_detection(detection));
            group.original_indices.push_back(index);
        }

        std::size_t new_class_count = 0;
        for (const auto& entry : groups) {
            if (impl_->trackers.find(entry.first) == impl_->trackers.end()) {
                ++new_class_count;
            }
        }
        std::size_t class_count = 0;
        if (!detail::checked_add_size(impl_->trackers.size(), new_class_count, &class_count) ||
            class_count > impl_->options.max_class_trackers) {
            throw_resource_limit("frame exceeds max_class_trackers");
        }

        for (auto& entry : groups) {
            DetectionGroup& group = entry.second;
            group.tracked.resize(group.detections.size());
        }

        TrackFrame result{frame.image_width, frame.image_height, {}};
        result.detections.resize(detection_count);
        for (std::size_t index = 0; index < detection_count; ++index) {
            result.detections[index].detection = frame.detections[index];
        }

        for (const auto& entry : groups) {
            if (impl_->trackers.find(entry.first) == impl_->trackers.end()) {
                impl_->trackers.emplace(entry.first, make_tracker(impl_->options));
            }
        }

        for (auto& entry : impl_->trackers) {
            const std::int32_t class_id = entry.first;
            bytetrack_t* tracker = entry.second.get();
            auto group_it = groups.find(class_id);
            if (group_it == groups.end()) {
                std::size_t output_count = 0;
                const tracker_status_t status =
                    bytetrack_update_ex(tracker, nullptr, 0, nullptr, 0, &output_count);
                if (status != TRACKER_STATUS_OK) {
                    throw tracker_error(status);
                }
                continue;
            }

            DetectionGroup& group = group_it->second;
            std::size_t output_count = 0;
            const tracker_status_t status = bytetrack_update_ex(
                tracker,
                group.detections.data(),
                group.detections.size(),
                group.tracked.data(),
                group.tracked.size(),
                &output_count);
            if (status != TRACKER_STATUS_OK) {
                throw tracker_error(status);
            }
            if (output_count != group.detections.size()) {
                throw YoloError(YoloErrorCode::EngineContractMismatch,
                                "ByteTrack did not return each submitted detection");
            }
            for (std::size_t output_index = 0; output_index < output_count; ++output_index) {
                const tracked_detection_ex_t& tracked = group.tracked[output_index];
                if (tracked.detection_index >= group.original_indices.size()) {
                    throw YoloError(YoloErrorCode::EngineContractMismatch,
                                    "ByteTrack returned an invalid detection index");
                }
                if (tracked.tracked.tracker_id >= 0) {
                    const std::size_t original_index =
                        group.original_indices[tracked.detection_index];
                    result.detections[original_index].track_id =
                        global_track_id(class_id, tracked.tracked.tracker_id);
                }
            }
        }

        return result;
    } catch (const std::bad_alloc&) {
        throw YoloError(YoloErrorCode::TrackerAllocationFailure,
                        "tracking update allocation failed");
    } catch (const std::length_error&) {
        throw_resource_limit("tracking update exceeds container capacity");
    }
}

void ByteTrackSession::reset() noexcept {
    for (auto& entry : impl_->trackers) {
        bytetrack_reset(entry.second.get());
    }
}

}  // namespace kfcore::yolo
