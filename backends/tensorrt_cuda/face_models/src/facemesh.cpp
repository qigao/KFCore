#include "kfcore/face_models/tensorrt.hpp"
#include "kfcore/face_models/cuda_model_names.hpp"

#include "facemesh_decode.hpp"
#include "facemesh_geometry.hpp"

#include "kfcore/image_processor/error.hpp"
#include "kfcore/image_processor/image_processor.hpp"
#include "kfcore/tensorrt/error.hpp"
#include "kfcore/yolo/tensorrt.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace kfcore::face_models
{
namespace
{

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_face(FaceModelErrorCode code, const std::string& detail)
{
    throw FaceModelError(code, "TensorRT face model stage: " + detail);
}

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw_face(FaceModelErrorCode::InvalidArgument, detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw_face(FaceModelErrorCode::ResourceLimitExceeded, detail);
}

double elapsed_ms(Clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

void validate_options(const TensorRtFaceMeshOptions& options)
{
    if (options.device_id < 0)
    {
        throw_invalid("CUDA device id must not be negative");
    }
    if (options.max_engine_bytes == 0U || options.max_source_bytes == 0U ||
        options.max_tensor_bytes == 0U || options.max_output_bytes == 0U ||
        options.max_face_detections == 0U)
    {
        throw_resource("all CUDA resource limits must be positive");
    }
    if (!std::isfinite(options.face_detection_score_threshold) ||
        options.face_detection_score_threshold < 0.0F ||
        options.face_detection_score_threshold > 1.0F)
    {
        throw_invalid("face detection score threshold must be finite within [0,1]");
    }
}

void validate_model_asset(const std::filesystem::path& path, const char* model,
                          std::size_t max_engine_bytes)
{
    if (path.empty())
    {
        throw_face(FaceModelErrorCode::InvalidModelAsset,
                   std::string(model) + " engine path must not be empty");
    }
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
    {
        throw_face(FaceModelErrorCode::InvalidModelAsset,
                   std::string(model) + " engine must be an existing regular file: " +
                       path.string());
    }
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (error || size == 0U)
    {
        throw_face(FaceModelErrorCode::InvalidModelAsset,
                   std::string(model) + " engine cannot be read or is empty: " +
                       path.string());
    }
    if (size > max_engine_bytes)
    {
        throw_resource(std::string(model) + " engine exceeds max_engine_bytes");
    }
}

FaceModelErrorCode map_tensorrt_error(tensorrt::TensorRtErrorCode code)
{
    using tensorrt::TensorRtErrorCode;
    switch (code)
    {
    case TensorRtErrorCode::InvalidArgument:
        return FaceModelErrorCode::InvalidArgument;
    case TensorRtErrorCode::InvalidTensorView:
        return FaceModelErrorCode::InvalidTensorView;
    case TensorRtErrorCode::FileIo:
    case TensorRtErrorCode::EngineDeserialize:
        return FaceModelErrorCode::InvalidModelAsset;
    case TensorRtErrorCode::EngineContractMismatch:
        return FaceModelErrorCode::ModelContractMismatch;
    case TensorRtErrorCode::ResourceLimitExceeded:
        return FaceModelErrorCode::ResourceLimitExceeded;
    case TensorRtErrorCode::ConcurrentExecution:
        return FaceModelErrorCode::ConcurrentExecution;
    case TensorRtErrorCode::TensorRtFailure:
    case TensorRtErrorCode::CudaFailure:
        return FaceModelErrorCode::RuntimeFailure;
    }
    return FaceModelErrorCode::RuntimeFailure;
}

[[noreturn]] void rethrow_tensorrt(const tensorrt::TensorRtError& error)
{
    throw_face(map_tensorrt_error(error.code()), error.what());
}

[[noreturn]] void rethrow_image(const image::ImageProcessorError& error)
{
    FaceModelErrorCode code = FaceModelErrorCode::RuntimeFailure;
    if (error.code() == image::ImageProcessorErrorCode::InvalidArgument)
    {
        code = FaceModelErrorCode::InvalidArgument;
    }
    else if (error.code() == image::ImageProcessorErrorCode::ResourceLimitExceeded)
    {
        code = FaceModelErrorCode::ResourceLimitExceeded;
    }
    throw_face(code, error.what());
}

FaceModelErrorCode map_yolo_error(yolo::YoloErrorCode code)
{
    using yolo::YoloErrorCode;
    switch (code)
    {
    case YoloErrorCode::InvalidArgument:
        return FaceModelErrorCode::InvalidArgument;
    case YoloErrorCode::FileIo:
    case YoloErrorCode::EngineDeserialize:
        return FaceModelErrorCode::InvalidModelAsset;
    case YoloErrorCode::EngineContractMismatch:
        return FaceModelErrorCode::ModelContractMismatch;
    case YoloErrorCode::ResourceLimitExceeded:
        return FaceModelErrorCode::ResourceLimitExceeded;
    case YoloErrorCode::TensorRtFailure:
    case YoloErrorCode::CudaFailure:
    case YoloErrorCode::TrackerAllocationFailure:
        return FaceModelErrorCode::RuntimeFailure;
    }
    return FaceModelErrorCode::RuntimeFailure;
}

[[noreturn]] void rethrow_yolo(const yolo::YoloError& error)
{
    throw_face(map_yolo_error(error.code()), error.what());
}

class UseGuard final
{
public:
    explicit UseGuard(std::atomic_flag& flag)
        : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw_face(FaceModelErrorCode::ConcurrentExecution,
                       "instance is already in use");
        }
    }

    ~UseGuard()
    {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag& flag_;
};

yolo::PixelFormat to_yolo_format(image::PixelFormat format)
{
    switch (format)
    {
    case image::PixelFormat::Bgr8:
        return yolo::PixelFormat::Bgr8;
    case image::PixelFormat::Rgb8:
        return yolo::PixelFormat::Rgb8;
    case image::PixelFormat::Nv12:
        return yolo::PixelFormat::Nv12;
    case image::PixelFormat::I420:
        return yolo::PixelFormat::I420;
    case image::PixelFormat::Nv21:
        return yolo::PixelFormat::Nv21;
    case image::PixelFormat::Yuy2:
        return yolo::PixelFormat::Yuy2;
    case image::PixelFormat::Uyvy:
        return yolo::PixelFormat::Uyvy;
    default:
        throw_invalid(
            "face detector requires BGR8, RGB8, NV12, I420, NV21, YUY2, or UYVY input");
    }
}

image::CudaImageProcessorOptions processor_options(const TensorRtFaceMeshOptions& options)
{
    return { options.device_id, options.max_source_bytes, options.max_tensor_bytes };
}

image::PreprocessOptions rgb_unit_options()
{
    image::PreprocessOptions options;
    options.output_format = image::PixelFormat::Rgb8;
    options.border_value  = 0.0F;
    return options;
}

image::ImageView stage_host_or_borrow_cuda(
    image::CudaImageProcessor& processor, const image::ImageView& source)
{
    if (source.memory_kind == image::MemoryKind::CudaDevice)
    {
        return source;
    }
    return processor.stage(source);
}

struct ExpectedTensor
{
    const char*            name;
    tensorrt::TensorIoMode mode;
    tensorrt::DataType     type;
    tensorrt::TensorShape  shape;
};

void validate_landmarker_contract(const std::shared_ptr<const tensorrt::Engine>& engine)
{
    const std::vector<ExpectedTensor> expected {
        { "image", tensorrt::TensorIoMode::Input, tensorrt::DataType::Float32,
          { 1, 3, kFaceMeshInputExtent, kFaceMeshInputExtent } },
        { "scores", tensorrt::TensorIoMode::Output, tensorrt::DataType::Float32, { 1 } },
        { "landmarks", tensorrt::TensorIoMode::Output, tensorrt::DataType::Float32,
          { 1, static_cast<std::int64_t>(kFaceMeshLandmarkCount), 3 } }
    };
    if (engine->tensors().size() != expected.size())
    {
        throw tensorrt::TensorRtError(
            tensorrt::TensorRtErrorCode::EngineContractMismatch,
            "MediaPipe face landmark engine exposes an unexpected tensor count");
    }
    for (const ExpectedTensor& contract : expected)
    {
        bool matched = false;
        for (const tensorrt::TensorDescriptor& tensor : engine->tensors())
        {
            if (tensor.name == contract.name)
            {
                matched = tensor.mode == contract.mode && tensor.data_type == contract.type &&
                          tensor.declared_shape == contract.shape;
                break;
            }
        }
        if (!matched)
        {
            throw tensorrt::TensorRtError(
                tensorrt::TensorRtErrorCode::EngineContractMismatch,
                std::string("MediaPipe face landmark engine tensor contract mismatch: ") +
                    contract.name);
        }
    }
}

tensorrt::TensorView device_input(const image::TensorView& input)
{
    if (input.data == nullptr || input.memory_kind != image::MemoryKind::CudaDevice ||
        input.element_type != image::TensorElementType::Float32 ||
        input.layout != image::TensorLayout::Nchw || input.batch != 1 ||
        input.channels != 3 || input.height != kFaceMeshInputExtent ||
        input.width != kFaceMeshInputExtent)
    {
        throw tensorrt::TensorRtError(
            tensorrt::TensorRtErrorCode::InvalidTensorView,
            "image expects a CUDA FP32 NCHW [1,3,192,192] tensor");
    }
    return { "image", tensorrt::DataType::Float32,
             { 1, 3, kFaceMeshInputExtent, kFaceMeshInputExtent }, input.data,
             input.byte_size, tensorrt::MemoryKind::CudaDevice };
}

} // namespace

struct TensorRtFaceDetector::Impl final
{
    Impl(const std::filesystem::path& path, const TensorRtFaceMeshOptions& value)
        : options(value)
    {
        yolo::EngineOptions engine_options;
        engine_options.device_id        = options.device_id;
        engine_options.max_batch        = 1U;
        engine_options.max_detections   = options.max_face_detections;
        engine_options.max_input_bytes  = options.max_tensor_bytes;
        engine_options.max_output_bytes = options.max_output_bytes;
        engine   = yolo::Engine::load(path, engine_options);
        detector = engine->create_detector();
    }

    TensorRtFaceMeshOptions               options;
    std::shared_ptr<const yolo::Engine>    engine;
    std::unique_ptr<yolo::TensorRtDetector> detector;
    std::atomic_flag                       in_use = ATOMIC_FLAG_INIT;
};

TensorRtFaceDetector::TensorRtFaceDetector(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

TensorRtFaceDetector::~TensorRtFaceDetector() = default;

std::unique_ptr<TensorRtFaceDetector> TensorRtFaceDetector::load(
    const TensorRtFaceMeshOptions& options)
{
    namespace names = kfcore::face_models::cuda_model_names;
    const std::filesystem::path engine_root =
        std::filesystem::path(names::default_model_root) /
        names::engine_profile_directory / names::default_engine_profile;
    return load(engine_root / names::face_detector, options);
}

std::unique_ptr<TensorRtFaceDetector> TensorRtFaceDetector::load(
    const std::filesystem::path& engine_path, const TensorRtFaceMeshOptions& options)
{
    validate_options(options);
    validate_model_asset(engine_path, "face detector", options.max_engine_bytes);
    try
    {
        return std::unique_ptr<TensorRtFaceDetector>(new TensorRtFaceDetector(
            std::make_unique<Impl>(engine_path, options)));
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const yolo::YoloError& error)
    {
        rethrow_yolo(error);
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("face detector allocation failed");
    }
}

FaceDetectionResult TensorRtFaceDetector::infer(const image::ImageView& source)
{
    if (!impl_)
    {
        throw_invalid("face detector state is unavailable");
    }
    const yolo::PixelFormat format = to_yolo_format(source.pixel_format);
    UseGuard guard(impl_->in_use);
    try
    {
        const Clock::time_point started = Clock::now();
        const yolo::ImageView view {
            source.data, source.width, source.height, source.row_stride, format,
            source.memory_kind == image::MemoryKind::Host
                ? yolo::MemoryKind::Host
                : yolo::MemoryKind::CudaDevice
        };
        const yolo::DetectionFrame detections = impl_->detector->detect(view);
        FaceDetectionResult result;
        for (const yolo::Detection& detection : detections.detections)
        {
            if (detection.class_id != 0 ||
                detection.score < impl_->options.face_detection_score_threshold ||
                (result.face.has_value() && detection.score <= result.face->confidence))
            {
                continue;
            }
            result.face = FaceDetection {
                { detection.box.left, detection.box.top,
                  detection.box.right - detection.box.left,
                  detection.box.bottom - detection.box.top },
                detection.score
            };
        }
        result.inference_ms = elapsed_ms(started);
        result.total_ms     = result.inference_ms;
        return result;
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const yolo::YoloError& error)
    {
        rethrow_yolo(error);
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("face detection allocation failed");
    }
}

struct TensorRtFaceLandmarker::Impl final
{
    Impl(const std::filesystem::path& path, const TensorRtFaceMeshOptions& value)
        : options(value)
        , processor(image::CudaImageProcessor::create(processor_options(options)))
    {
        tensorrt::EngineOptions engine_options;
        engine_options.device_id                   = options.device_id;
        engine_options.max_serialized_engine_bytes = options.max_engine_bytes;
        engine_options.max_input_bytes             = options.max_tensor_bytes;
        engine_options.max_output_bytes            = options.max_output_bytes;
        engine = tensorrt::Engine::load(path, engine_options);
        validate_landmarker_contract(engine);
        executor = engine->create_executor();
    }

    TensorRtFaceMeshOptions                  options;
    std::unique_ptr<image::CudaImageProcessor> processor;
    std::shared_ptr<const tensorrt::Engine>    engine;
    std::unique_ptr<tensorrt::Executor>        executor;
    std::atomic_flag                          in_use = ATOMIC_FLAG_INIT;
};

TensorRtFaceLandmarker::TensorRtFaceLandmarker(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

TensorRtFaceLandmarker::~TensorRtFaceLandmarker() = default;

std::unique_ptr<TensorRtFaceLandmarker> TensorRtFaceLandmarker::load(
    const TensorRtFaceMeshOptions& options)
{
    namespace names = kfcore::face_models::cuda_model_names;
    const std::filesystem::path engine_root =
        std::filesystem::path(names::default_model_root) /
        names::engine_profile_directory / names::default_engine_profile;
    return load(engine_root / names::face_mesh_landmarker, options);
}

std::unique_ptr<TensorRtFaceLandmarker> TensorRtFaceLandmarker::load(
    const std::filesystem::path& engine_path, const TensorRtFaceMeshOptions& options)
{
    validate_options(options);
    validate_model_asset(engine_path, "face landmark", options.max_engine_bytes);
    try
    {
        return std::unique_ptr<TensorRtFaceLandmarker>(new TensorRtFaceLandmarker(
            std::make_unique<Impl>(engine_path, options)));
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const tensorrt::TensorRtError& error)
    {
        rethrow_tensorrt(error);
    }
    catch (const image::ImageProcessorError& error)
    {
        rethrow_image(error);
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("face landmarker allocation failed");
    }
}

FaceLandmarkResult TensorRtFaceLandmarker::infer(const image::ImageView& source,
                                                 const RectF& face_box)
{
    if (!impl_)
    {
        throw_invalid("face landmarker state is unavailable");
    }
    if (!std::isfinite(face_box.x) || !std::isfinite(face_box.y) ||
        !std::isfinite(face_box.width) || !std::isfinite(face_box.height) ||
        face_box.width <= 0.0F || face_box.height <= 0.0F)
    {
        throw_invalid("face box must be finite and positive");
    }
    UseGuard guard(impl_->in_use);
    try
    {
        const Clock::time_point total_started = Clock::now();
        FaceLandmarkResult result;
        const Clock::time_point preprocess_started = Clock::now();
        const image::ImageView staged = stage_host_or_borrow_cuda(*impl_->processor, source);
        const detail::FaceRoi roi = detail::make_face_roi(face_box, kFaceMeshInputExtent);
        const image::TensorView input = impl_->processor->process_affine(
            staged, kFaceMeshInputExtent, kFaceMeshInputExtent,
            roi.destination_to_source, rgb_unit_options(),
            image::TensorElementType::Float32);
        result.preprocess_ms = elapsed_ms(preprocess_started);

        std::vector<float> landmarks(kFaceMeshLandmarkCount * 3U);
        float score = 0.0F;
        const Clock::time_point inference_started = Clock::now();
        impl_->executor->run(
            { device_input(input) },
            { { "scores", tensorrt::DataType::Float32, { 1 }, &score, sizeof(score),
                tensorrt::MemoryKind::Host },
              { "landmarks", tensorrt::DataType::Float32,
                { 1, static_cast<std::int64_t>(kFaceMeshLandmarkCount), 3 },
                landmarks.data(), landmarks.size() * sizeof(float),
                tensorrt::MemoryKind::Host } });
        result.inference_ms = elapsed_ms(inference_started);
        if (!std::isfinite(score))
        {
            throw_face(FaceModelErrorCode::ModelContractMismatch,
                       "MediaPipe face confidence is non-finite");
        }
        result.confidence = score;
        result.landmarks = detail::decode_face_landmarks(
            landmarks.data(), landmarks.size(), roi, kFaceMeshInputExtent,
            impl_->options.face_coordinates_normalized);
        result.total_ms = elapsed_ms(total_started);
        return result;
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const tensorrt::TensorRtError& error)
    {
        rethrow_tensorrt(error);
    }
    catch (const image::ImageProcessorError& error)
    {
        rethrow_image(error);
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("face landmark inference allocation failed");
    }
}

} // namespace kfcore::face_models
