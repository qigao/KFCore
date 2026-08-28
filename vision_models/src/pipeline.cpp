#include "kfcore/vision_models/core.hpp"

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

namespace kfcore::vision_models
{
namespace
{

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw VisionModelError(VisionModelErrorCode::InvalidArgument,
                           "hand pipeline stage: " + detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw VisionModelError(VisionModelErrorCode::ResourceLimitExceeded,
                           "hand pipeline stage: " + detail);
}

double elapsed_ms(Clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

[[noreturn]] void throw_face_invalid(const std::string& detail)
{
    throw VisionModelError(VisionModelErrorCode::InvalidArgument,
                           "FaceMesh pipeline stage: " + detail);
}

[[noreturn]] void throw_face_contract(const std::string& detail)
{
    throw VisionModelError(VisionModelErrorCode::ModelContractMismatch,
                           "FaceMesh pipeline stage: " + detail);
}

void validate_duration(double value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0)
    {
        throw_face_contract(std::string("backend ") + name +
                            " must be finite and non-negative");
    }
}

void validate_face_detection_result(const FaceDetectionResult& result)
{
    validate_duration(result.preprocess_ms, "detection preprocess_ms");
    validate_duration(result.inference_ms, "detection inference_ms");
    validate_duration(result.total_ms, "detection total_ms");
    if (!result.face)
    {
        return;
    }
    const RectF& box = result.face->box;
    if (!std::isfinite(box.x) || !std::isfinite(box.y) ||
        !std::isfinite(box.width) || !std::isfinite(box.height) ||
        box.width <= 0.0F || box.height <= 0.0F)
    {
        throw_face_contract("backend returned an invalid face box");
    }
    if (!std::isfinite(result.face->confidence) ||
        result.face->confidence < 0.0F || result.face->confidence > 1.0F)
    {
        throw_face_contract("backend face confidence must be finite within [0,1]");
    }
}

void validate_face_landmarks(const FaceLandmarkResult& result)
{
    if (!std::isfinite(result.confidence) || result.confidence < 0.0F ||
        result.confidence > 1.0F)
    {
        throw_face_contract("backend landmark confidence must be finite within [0,1]");
    }
    validate_duration(result.preprocess_ms, "landmark preprocess_ms");
    validate_duration(result.inference_ms, "landmark inference_ms");
    validate_duration(result.total_ms, "landmark total_ms");
    for (const Point3f& point : result.landmarks)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.z))
        {
            throw_face_contract("backend returned non-finite face landmarks");
        }
    }
}

void validate_unit(float value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
    {
        throw_invalid(std::string(name) + " must be finite within [0,1]");
    }
}

void validate_backend_unit(float value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
    {
        throw VisionModelError(VisionModelErrorCode::ModelContractMismatch,
                               std::string("hand pipeline stage: backend ") + name +
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
        throw VisionModelError(VisionModelErrorCode::ModelContractMismatch,
                               "hand pipeline stage: backend returned an invalid Palm box");
    }
    validate_backend_unit(hand.palm.confidence, "Palm confidence");
    validate_backend_unit(hand.landmark_confidence, "landmark confidence");
    for (const HandLandmark& landmark : hand.landmarks)
    {
        if (!std::isfinite(landmark.x) || !std::isfinite(landmark.y) ||
            !std::isfinite(landmark.z))
        {
            throw VisionModelError(VisionModelErrorCode::ModelContractMismatch,
                                   "hand pipeline stage: backend returned non-finite landmarks");
        }
    }
}

class UseGuard final
{
public:
    UseGuard(std::atomic_flag& flag, const char* stage)
        : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw VisionModelError(VisionModelErrorCode::ConcurrentExecution,
                                   std::string(stage) +
                                       " stage: instance is already in use");
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

struct HandPipeline::Impl final
{
    Impl(std::unique_ptr<HandInferenceBackend> backend_value,
         const HandPipelineOptions& options_value,
         bytetrack_t* tracker_value)
        : backend(std::move(backend_value))
        , options(options_value)
        , tracker(tracker_value, bytetrack_destroy)
    {
    }

    std::unique_ptr<HandInferenceBackend> backend;
    HandPipelineOptions options;
    std::unique_ptr<bytetrack_t, void (*)(bytetrack_t*)> tracker;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

HandPipeline::HandPipeline(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

HandPipeline::~HandPipeline() = default;

std::unique_ptr<HandPipeline> HandPipeline::create(
    std::unique_ptr<HandInferenceBackend> backend, const HandPipelineOptions& options)
{
    if (!backend)
    {
        throw_invalid("backend must not be null");
    }
    if (options.max_hands == 0U ||
        options.max_hands > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
    {
        throw_resource("max_hands must be within [1, INT_MAX]");
    }
    const bytetrack_config_t config = tracker_config(options.tracker);
    bytetrack_t* tracker = bytetrack_create(&config);
    if (tracker == nullptr)
    {
        throw VisionModelError(VisionModelErrorCode::TrackerFailure,
                               "hand pipeline stage: ByteTrack creation failed");
    }
    std::unique_ptr<bytetrack_t, void (*)(bytetrack_t*)> tracker_owner(tracker,
                                                                      bytetrack_destroy);
    auto impl = std::make_unique<Impl>(std::move(backend), options,
                                      tracker_owner.release());
    return std::unique_ptr<HandPipeline>(new HandPipeline(std::move(impl)));
}

HandFrame HandPipeline::process(const image::ImageView& image)
{
    if (!impl_)
    {
        throw_invalid("pipeline state is unavailable");
    }
    UseGuard guard(impl_->in_use, "hand pipeline");
    if (image.data == nullptr || image.width <= 0 || image.height <= 0)
    {
        throw_invalid("image data and dimensions must be valid");
    }

    const Clock::time_point total_started = Clock::now();
    HandFrame frame = impl_->backend->infer(image);
    if (frame.hands.size() > impl_->options.max_hands)
    {
        throw_resource("backend hand count exceeds max_hands");
    }

    std::vector<detection_t> detections;
    detections.reserve(frame.hands.size());
    for (HandResult& hand : frame.hands)
    {
        validate_hand(hand);
        hand.track_id = -1;
        const RectF& box = hand.palm.box;
        detection_t detection {};
        detection.box = { box.x, box.y, box.x + box.width, box.y + box.height };
        detection.confidence = hand.palm.confidence;
        detection.has_confidence = 1;
        detection.class_id = 0;
        detection.has_class_id = 1;
        detections.push_back(detection);
    }

    const Clock::time_point tracking_started = Clock::now();
    bytetrack_t* candidate_tracker_raw = nullptr;
    const tracker_status_t clone_status =
        bytetrack_clone(impl_->tracker.get(), &candidate_tracker_raw);
    if (clone_status != TRACKER_STATUS_OK)
    {
        throw VisionModelError(VisionModelErrorCode::TrackerFailure,
                               "hand pipeline stage: ByteTrack clone failed with status " +
                                   std::to_string(static_cast<int>(clone_status)));
    }
    std::unique_ptr<bytetrack_t, void (*)(bytetrack_t*)> candidate_tracker(
        candidate_tracker_raw, bytetrack_destroy);
    std::vector<tracked_detection_ex_t> tracked(frame.hands.size());
    std::size_t written = 0;
    const tracker_status_t status = bytetrack_update_ex(
        candidate_tracker.get(), detections.empty() ? nullptr : detections.data(), detections.size(),
        tracked.empty() ? nullptr : tracked.data(), tracked.size(), &written);
    if (status != TRACKER_STATUS_OK)
    {
        throw VisionModelError(VisionModelErrorCode::TrackerFailure,
                               "hand pipeline stage: ByteTrack update failed with status " +
                                   std::to_string(static_cast<int>(status)));
    }
    if (written != frame.hands.size())
    {
        throw VisionModelError(VisionModelErrorCode::TrackerFailure,
                               "hand pipeline stage: ByteTrack output count mismatch");
    }
    std::vector<bool> associated(frame.hands.size(), false);
    for (const tracked_detection_ex_t& item : tracked)
    {
        if (item.detection_index >= frame.hands.size() || associated[item.detection_index])
        {
            throw VisionModelError(VisionModelErrorCode::TrackerFailure,
                                   "hand pipeline stage: invalid ByteTrack detection_index");
        }
        associated[item.detection_index] = true;
        frame.hands[item.detection_index].track_id = item.tracked.tracker_id;
    }
    impl_->tracker.swap(candidate_tracker);
    frame.timings.tracking_ms = elapsed_ms(tracking_started);
    frame.timings.total_ms    = elapsed_ms(total_started);
    return frame;
}

void HandPipeline::reset()
{
    if (!impl_)
    {
        throw_invalid("pipeline state is unavailable");
    }
    UseGuard guard(impl_->in_use, "hand pipeline");
    bytetrack_reset(impl_->tracker.get());
}

struct FaceMeshPipeline::Impl final
{
    Impl(std::unique_ptr<FaceDetectorBackend> detector_value,
         std::unique_ptr<FaceLandmarkBackend> landmarker_value,
         const FaceMeshPipelineOptions& options_value)
        : detector(std::move(detector_value))
        , landmarker(std::move(landmarker_value))
        , options(options_value)
    {
    }

    std::unique_ptr<FaceDetectorBackend> detector;
    std::unique_ptr<FaceLandmarkBackend> landmarker;
    FaceMeshPipelineOptions              options;
    std::atomic_flag                     in_use = ATOMIC_FLAG_INIT;
};

FaceMeshPipeline::FaceMeshPipeline(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

FaceMeshPipeline::~FaceMeshPipeline() = default;

std::unique_ptr<FaceMeshPipeline> FaceMeshPipeline::create(
    std::unique_ptr<FaceDetectorBackend> detector,
    std::unique_ptr<FaceLandmarkBackend> landmarker,
    const FaceMeshPipelineOptions& options)
{
    if (!detector)
    {
        throw_face_invalid("detector must not be null");
    }
    if (!landmarker)
    {
        throw_face_invalid("landmarker must not be null");
    }
    if (!std::isfinite(options.landmark_score_threshold) ||
        options.landmark_score_threshold < 0.0F ||
        options.landmark_score_threshold > 1.0F)
    {
        throw_face_invalid("landmark_score_threshold must be finite within [0,1]");
    }
    auto impl = std::make_unique<Impl>(std::move(detector), std::move(landmarker),
                                      options);
    return std::unique_ptr<FaceMeshPipeline>(new FaceMeshPipeline(std::move(impl)));
}

FaceMeshFrame FaceMeshPipeline::process(const image::ImageView& image)
{
    if (!impl_)
    {
        throw_face_invalid("pipeline state is unavailable");
    }
    UseGuard guard(impl_->in_use, "FaceMesh pipeline");
    if (image.data == nullptr || image.width <= 0 || image.height <= 0)
    {
        throw_face_invalid("image data and dimensions must be valid");
    }

    const Clock::time_point total_started = Clock::now();
    FaceMeshFrame frame;
    const FaceDetectionResult detection = impl_->detector->infer(image);
    validate_face_detection_result(detection);
    frame.timings.detection_preprocess_ms = detection.preprocess_ms;
    frame.timings.detection_inference_ms  = detection.inference_ms;
    if (!detection.face)
    {
        frame.timings.total_ms = elapsed_ms(total_started);
        return frame;
    }

    frame.detection = detection.face;
    FaceLandmarkResult landmarks = impl_->landmarker->infer(image, detection.face->box);
    validate_face_landmarks(landmarks);
    frame.timings.landmark_preprocess_ms = landmarks.preprocess_ms;
    frame.timings.landmark_inference_ms  = landmarks.inference_ms;
    if (landmarks.confidence >= impl_->options.landmark_score_threshold)
    {
        frame.landmarks = std::move(landmarks);
    }
    frame.timings.total_ms = elapsed_ms(total_started);
    return frame;
}

} // namespace kfcore::vision_models
