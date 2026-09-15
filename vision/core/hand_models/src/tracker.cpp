#include "kfcore/hand_models/tracking.hpp"

#include "hand_appearance.hpp"
#include "trackers/tracker.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::hand_models
{
namespace
{

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw HandModelError(HandModelErrorCode::InvalidArgument,
                         "hand tracking: " + detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw HandModelError(HandModelErrorCode::ResourceLimitExceeded,
                         "hand tracking: " + detail);
}

double elapsed_ms(Clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

void validate_unit(float value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
    {
        throw_invalid(std::string(name) + " must be finite within [0,1]");
    }
}

void validate_model_unit(float value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
    {
        throw HandModelError(HandModelErrorCode::ModelContractMismatch,
                             std::string("hand tracking model result ") + name +
                                 " must be finite within [0,1]");
    }
}

bytetrack_config_t tracker_config(const ByteTrackOptions& options)
{
    if (options.lost_track_buffer < 0 || !std::isfinite(options.frame_rate) ||
        options.frame_rate <= 0.0F || options.minimum_consecutive_frames <= 0)
    {
        throw_invalid("ByteTrack counts and frame rate are invalid");
    }
    validate_unit(options.track_activation_threshold, "track activation threshold");
    validate_unit(options.minimum_iou_threshold, "minimum IoU threshold");
    validate_unit(options.high_confidence_threshold, "high-confidence threshold");
    return { options.lost_track_buffer,
             options.frame_rate,
             options.track_activation_threshold,
             options.minimum_consecutive_frames,
             options.minimum_iou_threshold,
             options.high_confidence_threshold };
}

void validate_hand(const HandResult& hand)
{
    const RectF& box = hand.palm.box;
    if (!std::isfinite(box.x) || !std::isfinite(box.y) ||
        !std::isfinite(box.width) || !std::isfinite(box.height) ||
        box.width <= 0.0F || box.height <= 0.0F)
    {
        throw HandModelError(HandModelErrorCode::ModelContractMismatch,
                             "hand tracking received an invalid Palm box");
    }
    validate_model_unit(hand.palm.confidence, "Palm confidence");
    validate_model_unit(hand.landmark_confidence, "landmark confidence");
    for (const HandLandmark& landmark : hand.landmarks)
    {
        if (!std::isfinite(landmark.x) || !std::isfinite(landmark.y) ||
            !std::isfinite(landmark.z))
        {
            throw HandModelError(HandModelErrorCode::ModelContractMismatch,
                                 "hand tracking received non-finite landmarks");
        }
    }
}

void validate_appearance_options(const HandAppearanceOptions& options)
{
    if (!std::isfinite(options.minimum_palm_span_pixels) ||
        options.minimum_palm_span_pixels <= 0.0F ||
        !std::isfinite(options.minimum_part_in_frame_sample_ratio) ||
        options.minimum_part_in_frame_sample_ratio <= 0.0F ||
        options.minimum_part_in_frame_sample_ratio > 1.0F)
    {
        throw_invalid("hand appearance options are invalid");
    }
}

class UseGuard final
{
public:
    explicit UseGuard(std::atomic_flag& flag) : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw HandModelError(HandModelErrorCode::ConcurrentExecution,
                                 "hand tracker instance is already in use");
        }
    }

    ~UseGuard()
    {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag& flag_;
};

} // namespace

struct HandTracker::Impl final
{
    Impl(HandTrackingOptions options_value, bytetrack_t* tracker_value)
        : options(std::move(options_value))
        , tracker(tracker_value, bytetrack_destroy)
    {
    }

    HandTrackingOptions options;
    std::unique_ptr<bytetrack_t, void (*)(bytetrack_t*)> tracker;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

HandTracker::HandTracker(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
HandTracker::~HandTracker() = default;

std::unique_ptr<HandTracker> HandTracker::create(const HandTrackingOptions& options)
{
    if (options.max_hands == 0U ||
        options.max_hands > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
    {
        throw_resource("max_hands must be within [1, INT_MAX]");
    }
    const bytetrack_config_t config = tracker_config(options.tracker);
    validate_appearance_options(options.appearance);
    bytetrack_t* tracker = bytetrack_create(&config);
    if (tracker == nullptr)
    {
        throw HandModelError(HandModelErrorCode::TrackerFailure,
                             "hand tracker creation failed");
    }
    try
    {
        return std::unique_ptr<HandTracker>(new HandTracker(
            std::make_unique<Impl>(options, tracker)));
    }
    catch (...)
    {
        bytetrack_destroy(tracker);
        throw;
    }
}

HandFrame HandTracker::update(HandFrame frame)
{
    if (!impl_)
    {
        throw_invalid("state is unavailable");
    }
    if (impl_->options.appearance.enabled)
    {
        throw_invalid("appearance extraction requires a source image");
    }
    static const image::ImageView no_source{};
    return update(no_source, std::move(frame));
}

HandFrame HandTracker::update(const image::ImageView& source, HandFrame frame)
{
    if (!impl_)
    {
        throw_invalid("state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    if (frame.hands.size() > impl_->options.max_hands)
    {
        throw_resource("hand count exceeds max_hands");
    }
    if (impl_->options.appearance.enabled)
    {
        detail::validate_hand_appearance_source(source);
    }

    const Clock::time_point post_started = Clock::now();
    std::vector<detection_t> detections;
    detections.reserve(frame.hands.size());
    Clock::time_point appearance_started{};
    if (impl_->options.appearance.enabled)
    {
        appearance_started = Clock::now();
    }

    for (HandResult& hand : frame.hands)
    {
        validate_hand(hand);
        if (impl_->options.appearance.enabled)
        {
            hand.appearance = detail::make_hand_appearance_descriptor(
                source, hand.landmarks, impl_->options.appearance);
        }
        hand.track_id = -1;
        const RectF& box = hand.palm.box;
        detection_t detection{};
        detection.box = { box.x, box.y, box.x + box.width, box.y + box.height };
        detection.confidence = hand.palm.confidence;
        detection.has_confidence = 1;
        detection.class_id = 0;
        detection.has_class_id = 1;
        detections.push_back(detection);
    }

    if (impl_->options.appearance.enabled)
    {
        frame.timings.appearance_ms = elapsed_ms(appearance_started);
    }

    const Clock::time_point tracking_started = Clock::now();
    bytetrack_t* candidate_tracker_raw = nullptr;
    const tracker_status_t clone_status =
        bytetrack_clone(impl_->tracker.get(), &candidate_tracker_raw);
    if (clone_status != TRACKER_STATUS_OK)
    {
        throw HandModelError(HandModelErrorCode::TrackerFailure,
                             "ByteTrack clone failed with status " +
                                 std::to_string(static_cast<int>(clone_status)));
    }
    std::unique_ptr<bytetrack_t, void (*)(bytetrack_t*)> candidate_tracker(
        candidate_tracker_raw, bytetrack_destroy);
    std::vector<tracked_detection_ex_t> tracked(frame.hands.size());
    std::size_t written = 0;
    const tracker_status_t status = bytetrack_update_ex(
        candidate_tracker.get(), detections.empty() ? nullptr : detections.data(),
        detections.size(), tracked.empty() ? nullptr : tracked.data(), tracked.size(),
        &written);
    if (status != TRACKER_STATUS_OK)
    {
        throw HandModelError(HandModelErrorCode::TrackerFailure,
                             "ByteTrack update failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
    if (written != frame.hands.size())
    {
        throw HandModelError(HandModelErrorCode::TrackerFailure,
                             "ByteTrack output count mismatch");
    }

    std::vector<bool> associated(frame.hands.size(), false);
    for (const tracked_detection_ex_t& item : tracked)
    {
        if (item.detection_index >= frame.hands.size() || associated[item.detection_index])
        {
            throw HandModelError(HandModelErrorCode::TrackerFailure,
                                 "ByteTrack returned an invalid detection index");
        }
        associated[item.detection_index] = true;
        frame.hands[item.detection_index].track_id = item.tracked.tracker_id;
    }
    impl_->tracker.swap(candidate_tracker);
    frame.timings.tracking_ms = elapsed_ms(tracking_started);
    frame.timings.total_ms += elapsed_ms(post_started);
    return frame;
}

void HandTracker::reset()
{
    if (!impl_)
    {
        throw_invalid("state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    bytetrack_reset(impl_->tracker.get());
}

} // namespace kfcore::hand_models
