#include "kfcore/face_models/cpu.hpp"

#include "facemesh_decode.hpp"
#include "facemesh_geometry.hpp"
#include "support.hpp"

#include "kfcore/image_processor/cpu.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::face_models
{
namespace
{

using Clock = std::chrono::steady_clock;
constexpr std::int32_t kFaceDetectorInputExtent = 640;

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::InvalidArgument,
                         "CPU face model stage: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::ModelContractMismatch,
                         "CPU face model stage: " + detail);
}

double elapsed_ms(Clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

void validate_options(const CpuFaceMeshOptions& options)
{
    if (options.max_model_bytes == 0U || options.max_source_bytes == 0U ||
        options.max_tensor_bytes == 0U || options.max_output_bytes == 0U)
    {
        throw_invalid("resource limits must be positive");
    }
    if (options.intra_op_threads < 0 || options.inter_op_threads < 0 ||
        options.face_class_id < 0 ||
        !std::isfinite(options.face_detection_score_threshold) ||
        options.face_detection_score_threshold < 0.0F ||
        options.face_detection_score_threshold > 1.0F)
    {
        throw_invalid("thread counts, class id, or score threshold are invalid");
    }
}

CpuFaceModelOptions runtime_options(const CpuFaceMeshOptions& options)
{
    CpuFaceModelOptions result;
    result.intra_op_threads = options.intra_op_threads;
    result.inter_op_threads = options.inter_op_threads;
    result.max_model_bytes  = options.max_model_bytes;
    result.max_output_bytes = options.max_output_bytes;
    return result;
}

detail::CpuModelContract detector_contract()
{
    return { "YOLOv12 face detector",
             { detail::fp32_contract("images", { 1, 3, 640, 640 }) },
             { detail::fp32_contract("output0", { 1, 300, 6 }) } };
}

detail::CpuModelContract landmarker_contract()
{
    return { "MediaPipe face landmark",
             { detail::fp32_contract("image", { 1, 3, kFaceMeshInputExtent,
                                                  kFaceMeshInputExtent }) },
             { detail::fp32_contract("scores", { 1 }),
               detail::fp32_contract("landmarks",
                                     { 1, static_cast<std::int64_t>(kFaceMeshLandmarkCount),
                                       3 }) } };
}

image::PreprocessOptions rgb_unit_options(float border_value = 0.0F)
{
    image::PreprocessOptions options;
    options.output_format = image::PixelFormat::Rgb8;
    options.border_value  = border_value;
    return options;
}

void validate_host_image(const image::ImageView& image)
{
    if (image.memory_kind != image::MemoryKind::Host)
    {
        throw_invalid("CPU backend accepts Host images only");
    }
}

} // namespace

struct CpuFaceDetector::Impl final
{
    Impl(const std::filesystem::path& path, const CpuFaceMeshOptions& value)
        : options(value)
        , session(detail::make_session("Detector", path, detector_contract(),
                                       runtime_options(options)))
    {
    }

    CpuFaceMeshOptions          options;
    runtime_onnx::OwnedSession session;
    std::atomic_flag           in_use = ATOMIC_FLAG_INIT;
};

CpuFaceDetector::CpuFaceDetector(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

CpuFaceDetector::~CpuFaceDetector() = default;

std::unique_ptr<CpuFaceDetector> CpuFaceDetector::load(
    const std::filesystem::path& model_path, const CpuFaceMeshOptions& options)
{
    validate_options(options);
    try
    {
        return std::unique_ptr<CpuFaceDetector>(
            new CpuFaceDetector(std::make_unique<Impl>(model_path, options)));
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        detail::throw_allocation("face detector", "load");
    }
}

FaceDetectionResult CpuFaceDetector::infer(const image::ImageView& source)
{
    detail::CpuCallGuard guard(impl_->in_use, "face detector");
    validate_host_image(source);
    const Clock::time_point total_started = Clock::now();
    FaceDetectionResult result;
    image::LetterboxTransform letterbox;
    const Clock::time_point preprocess_started = Clock::now();
    std::vector<float> input = image::CpuImageProcessor::letterbox_nchw(
        source, kFaceDetectorInputExtent, kFaceDetectorInputExtent,
        rgb_unit_options(114.0F), impl_->options.max_source_bytes,
        impl_->options.max_tensor_bytes, &letterbox);
    result.preprocess_ms = elapsed_ms(preprocess_started);

    const Clock::time_point inference_started = Clock::now();
    std::vector<detail::CpuSessionOutput> outputs = detail::run(
        impl_->session,
        { { input.data(), input.size(), { 1, 3, 640, 640 } } });
    result.inference_ms = elapsed_ms(inference_started);
    if (outputs.size() != 1U || outputs[0].float_values.size() != 300U * 6U)
    {
        throw_contract("YOLOv12 face detector output set is invalid");
    }
    result.face = detail::decode_yolo12_face(
        outputs[0].float_values.data(), outputs[0].float_values.size(),
        impl_->options.face_class_id,
        impl_->options.face_detection_score_threshold, letterbox,
        source.width, source.height);
    result.total_ms = elapsed_ms(total_started);
    return result;
}

struct CpuFaceLandmarker::Impl final
{
    Impl(const std::filesystem::path& path, const CpuFaceMeshOptions& value)
        : options(value)
        , session(detail::make_session("Landmarker", path, landmarker_contract(),
                                       runtime_options(options)))
    {
    }

    CpuFaceMeshOptions          options;
    runtime_onnx::OwnedSession session;
    std::atomic_flag           in_use = ATOMIC_FLAG_INIT;
};

CpuFaceLandmarker::CpuFaceLandmarker(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

CpuFaceLandmarker::~CpuFaceLandmarker() = default;

std::unique_ptr<CpuFaceLandmarker> CpuFaceLandmarker::load(
    const std::filesystem::path& model_path, const CpuFaceMeshOptions& options)
{
    validate_options(options);
    try
    {
        return std::unique_ptr<CpuFaceLandmarker>(
            new CpuFaceLandmarker(std::make_unique<Impl>(model_path, options)));
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        detail::throw_allocation("face landmarker", "load");
    }
}

FaceLandmarkResult CpuFaceLandmarker::infer(const image::ImageView& source,
                                             const RectF& face_box)
{
    detail::CpuCallGuard guard(impl_->in_use, "face landmarker");
    validate_host_image(source);
    const Clock::time_point total_started = Clock::now();
    FaceLandmarkResult result;
    const Clock::time_point preprocess_started = Clock::now();
    const image::BgrImage packed = image::CpuImageProcessor::copy_bgr(
        source, impl_->options.max_source_bytes);
    const detail::FaceRoi roi = detail::make_face_roi(face_box, kFaceMeshInputExtent);
    const image::BgrImage crop = image::CpuImageProcessor::warp_affine_bgr(
        packed, kFaceMeshInputExtent, kFaceMeshInputExtent,
        roi.destination_to_source, 0.0F, impl_->options.max_source_bytes);
    std::vector<float> input = image::CpuImageProcessor::to_nchw(
        crop, rgb_unit_options(), impl_->options.max_tensor_bytes);
    result.preprocess_ms = elapsed_ms(preprocess_started);

    const Clock::time_point inference_started = Clock::now();
    std::vector<detail::CpuSessionOutput> outputs = detail::run(
        impl_->session,
        { { input.data(), input.size(),
            { 1, 3, kFaceMeshInputExtent, kFaceMeshInputExtent } } });
    result.inference_ms = elapsed_ms(inference_started);
    if (outputs.size() != 2U || outputs[0].float_values.size() != 1U ||
        outputs[1].float_values.size() != kFaceMeshLandmarkCount * 3U)
    {
        throw_contract("MediaPipe face landmark output set is invalid");
    }
    result.confidence = outputs[0].float_values[0];
    if (!std::isfinite(result.confidence))
    {
        throw_contract("MediaPipe face confidence is non-finite");
    }
    result.landmarks = detail::decode_face_landmarks(
        outputs[1].float_values.data(), outputs[1].float_values.size(), roi,
        kFaceMeshInputExtent, impl_->options.face_coordinates_normalized);
    result.total_ms = elapsed_ms(total_started);
    return result;
}

} // namespace kfcore::face_models
