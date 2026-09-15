#include "kfcore/face_models/runtime.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace kfcore::face_models
{
namespace
{

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::InvalidArgument,
                         "face mesh: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::ModelContractMismatch,
                         "face mesh contract: " + detail);
}

double elapsed_ms(Clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

void validate_duration(double value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0)
    {
        throw_contract(std::string(name) + " must be finite and non-negative");
    }
}

void validate_detection(const FaceDetectionResult& result)
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
        throw_contract("detector returned an invalid face box");
    }
    if (!std::isfinite(result.face->confidence) ||
        result.face->confidence < 0.0F || result.face->confidence > 1.0F)
    {
        throw_contract("detector confidence must be finite within [0,1]");
    }
}

void validate_landmarks(const FaceLandmarkResult& result)
{
    if (!std::isfinite(result.confidence) || result.confidence < 0.0F ||
        result.confidence > 1.0F)
    {
        throw_contract("landmark confidence must be finite within [0,1]");
    }
    validate_duration(result.preprocess_ms, "landmark preprocess_ms");
    validate_duration(result.inference_ms, "landmark inference_ms");
    validate_duration(result.total_ms, "landmark total_ms");
    for (const Point3f& point : result.landmarks)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.z))
        {
            throw_contract("landmarker returned non-finite landmarks");
        }
    }
}

bool describes_same_frame(const image::FrameView& frame) noexcept
{
    return frame.source.data != nullptr && frame.compute.data != nullptr &&
           frame.source.width > 0 && frame.source.height > 0 &&
           frame.compute.width > 0 && frame.compute.height > 0 &&
           frame.source.width == frame.compute.width &&
           frame.source.height == frame.compute.height &&
           frame.source.pixel_format == frame.compute.pixel_format;
}

class UseGuard final
{
public:
    explicit UseGuard(std::atomic_flag& flag) : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw FaceModelError(FaceModelErrorCode::ConcurrentExecution,
                                 "face mesh instance is already in use");
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

struct FaceMesh::Impl final
{
    Impl(std::unique_ptr<FaceDetector> detector_value,
         std::unique_ptr<FaceLandmarker> landmarker_value,
         FaceMeshOptions options_value)
        : detector(std::move(detector_value))
        , landmarker(std::move(landmarker_value))
        , options(options_value)
    {
    }

    std::unique_ptr<FaceDetector> detector;
    std::unique_ptr<FaceLandmarker> landmarker;
    FaceMeshOptions options;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

FaceMesh::FaceMesh(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FaceMesh::~FaceMesh() = default;

std::unique_ptr<FaceMesh> FaceMesh::load(
    runtime::Runtime& runtime,
    const runtime::ModelPackage& detector_package,
    const runtime::ExecutionPolicy& detector_policy,
    const runtime::ModelPackage& landmarker_package,
    const runtime::ExecutionPolicy& landmarker_policy,
    const FaceRuntimeOptions& runtime_options,
    const FaceMeshOptions& options)
{
    if (!std::isfinite(options.landmark_score_threshold) ||
        options.landmark_score_threshold < 0.0F ||
        options.landmark_score_threshold > 1.0F)
    {
        throw_invalid("landmark_score_threshold must be finite within [0,1]");
    }
    try
    {
        auto detector = FaceDetector::load(runtime, detector_package, detector_policy,
                                           runtime_options);
        auto landmarker = FaceLandmarker::load(runtime, landmarker_package, landmarker_policy,
                                               runtime_options);
        return std::unique_ptr<FaceMesh>(new FaceMesh(std::make_unique<Impl>(
            std::move(detector), std::move(landmarker), options)));
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        throw FaceModelError(FaceModelErrorCode::ResourceLimitExceeded,
                             "face mesh allocation failed");
    }
}

FaceMeshFrame FaceMesh::infer(const image::ImageView& image)
{
    return infer(image::FrameView::borrow(image));
}

FaceMeshFrame FaceMesh::infer(const image::FrameView& frame_view)
{
    if (!impl_)
    {
        throw_invalid("state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    if (!describes_same_frame(frame_view))
    {
        throw_invalid("source and compute images must describe the same valid frame");
    }

    const Clock::time_point total_started = Clock::now();
    FaceMeshFrame frame;
    const FaceDetectionResult detection = impl_->detector->infer(frame_view.compute);
    validate_detection(detection);
    frame.timings.detection_preprocess_ms = detection.preprocess_ms;
    frame.timings.detection_inference_ms  = detection.inference_ms;
    if (!detection.face)
    {
        frame.timings.total_ms = elapsed_ms(total_started);
        return frame;
    }

    frame.detection = detection.face;
    FaceLandmarkResult landmarks = impl_->landmarker->infer(
        frame_view.compute, detection.face->box);
    validate_landmarks(landmarks);
    frame.timings.landmark_preprocess_ms = landmarks.preprocess_ms;
    frame.timings.landmark_inference_ms  = landmarks.inference_ms;
    if (landmarks.confidence >= impl_->options.landmark_score_threshold)
    {
        frame.landmarks = std::move(landmarks);
    }
    frame.timings.total_ms = elapsed_ms(total_started);
    return frame;
}

const runtime::ExecutionRoute& FaceMesh::detector_execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty{};
    return impl_ ? impl_->detector->execution_route() : empty;
}

const runtime::ExecutionRoute& FaceMesh::landmarker_execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty{};
    return impl_ ? impl_->landmarker->execution_route() : empty;
}

} // namespace kfcore::face_models
