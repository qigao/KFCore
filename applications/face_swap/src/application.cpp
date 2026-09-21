#include "kfcore/face_applications/application.hpp"

#include "kfcore/face_models/runtime.hpp"
#include "kfcore/runtime/error.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/face_models/error.hpp"
#include "kfcore/face_models/inswapper_embedding.hpp"
#include "kfcore/image_processor/cpu.hpp"

#include "composer.hpp"
#include "geometry.hpp"
#include "preprocess.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace kfcore::face_applications
{
namespace
{

class StageTimer final
{
public:
    explicit StageTimer(FaceSwapDuration* destination)
        : destination_(destination)
        , started_(std::chrono::steady_clock::now())
    {
    }

    ~StageTimer()
    {
        if (destination_ != nullptr)
        {
            *destination_ = std::chrono::duration_cast<FaceSwapDuration>(
                std::chrono::steady_clock::now() - started_);
        }
    }

private:
    FaceSwapDuration* destination_;
    std::chrono::steady_clock::time_point started_;
};

class CallGuard final
{
public:
    explicit CallGuard(std::atomic_flag& flag)
        : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw FaceApplicationError(
                FaceApplicationErrorCode::ConcurrentExecution,
                "face application does not support concurrent calls on one instance");
        }
    }

    ~CallGuard() { flag_.clear(std::memory_order_release); }

private:
    std::atomic_flag& flag_;
};

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw FaceApplicationError(FaceApplicationErrorCode::InvalidArgument,
                                  "face application validation stage: " + detail);
}

void validate_paths(const FaceApplicationModelPackages& paths)
{
    const std::pair<const std::filesystem::path*, const char*> required[] = {
        { &paths.detector, "detector" },
        { &paths.face68, "face68" },
        { &paths.arcface, "arcface" },
        { &paths.inswapper, "inswapper" },
        { &paths.inswapper_matrix, "inswapper_matrix" },
    };
    for (const auto& entry : required)
    {
        if (entry.first->empty())
        {
            throw_invalid(std::string(entry.second) + " path must not be empty");
        }
    }
    if ((paths.gfpgan && paths.gfpgan->empty()) ||
        (paths.age_gender && paths.age_gender->empty()))
    {
        throw_invalid("optional model paths must not be empty when configured");
    }
}

void validate_options(const FaceSwapOptions& options)
{
    if (!std::isfinite(options.detector_score_threshold) ||
        options.detector_score_threshold < 0.0F ||
        options.detector_score_threshold > 1.0F)
    {
        throw_invalid("detector_score_threshold must be finite within [0,1]");
    }
    if (options.face_class_id < 0)
    {
        throw_invalid("face_class_id must not be negative");
    }
    if (!std::isfinite(options.enhancer_blend) || options.enhancer_blend < 0.0F ||
        options.enhancer_blend > 1.0F)
    {
        throw_invalid("enhancer_blend must be finite within [0,1]");
    }
    if (options.max_image_bytes == 0U || options.max_tensor_bytes == 0U ||
        options.max_output_bytes == 0U)
    {
        throw_invalid("resource byte limits must be positive");
    }
}

kfcore::face_models::PreparedFaceModelOptions
face_model_options(const FaceSwapOptions& options)
{
    kfcore::face_models::PreparedFaceModelOptions result;
    result.max_tensor_bytes = options.max_tensor_bytes;
    result.max_output_bytes = options.max_output_bytes;
    return result;
}

kfcore::face_models::FaceRuntimeOptions
detector_options(const FaceSwapOptions& options)
{
    kfcore::face_models::FaceRuntimeOptions result;
    result.max_source_bytes = options.max_image_bytes;
    result.max_tensor_bytes = options.max_tensor_bytes;
    result.max_output_bytes = options.max_output_bytes;
    result.face_detection_score_threshold = options.detector_score_threshold;
    result.face_class_id = options.face_class_id;
    return result;
}

void validate_image(const kfcore::image::BgrImage& image, std::size_t max_image_bytes)
{
    if (image.width <= 0 || image.height <= 0)
    {
        throw_invalid("BGR image dimensions must be positive");
    }
    const std::uintmax_t pixels = static_cast<std::uintmax_t>(image.width) *
                                  static_cast<std::uintmax_t>(image.height);
    const std::uintmax_t bytes = pixels * 3U;
    if (bytes > static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)()) ||
        bytes > max_image_bytes || image.pixels.size() != static_cast<std::size_t>(bytes))
    {
        throw_invalid("BGR image storage is malformed or exceeds max_image_bytes");
    }
}

kfcore::image::BgrImage crop_detection(const kfcore::image::BgrImage& image,
                                       const FaceBox& box,
                                       std::size_t max_image_bytes)
{
    const int left = (std::clamp)(static_cast<int>(std::floor(box.left)), 0, image.width - 1);
    const int top = (std::clamp)(static_cast<int>(std::floor(box.top)), 0, image.height - 1);
    const int right = (std::clamp)(static_cast<int>(std::ceil(box.right)), left + 1, image.width);
    const int bottom = (std::clamp)(static_cast<int>(std::ceil(box.bottom)), top + 1, image.height);
    kfcore::image::BgrImage result;
    result.width = right - left;
    result.height = bottom - top;
    const std::size_t row_bytes = static_cast<std::size_t>(result.width) * 3U;
    const std::size_t bytes = row_bytes * static_cast<std::size_t>(result.height);
    if (bytes > max_image_bytes)
    {
        throw_invalid("face crop exceeds max_image_bytes");
    }
    result.pixels.resize(bytes);
    for (int row = 0; row < result.height; ++row)
    {
        const std::size_t source_offset =
            (static_cast<std::size_t>(top + row) * image.width + left) * 3U;
        std::memcpy(result.pixels.data() + static_cast<std::size_t>(row) * row_bytes,
                    image.pixels.data() + source_offset, row_bytes);
    }
    return result;
}

[[noreturn]] void translate_image_error(const kfcore::image::ImageProcessorError& error)
{
    const FaceApplicationErrorCode code =
        error.code() == kfcore::image::ImageProcessorErrorCode::ResourceLimitExceeded
            ? FaceApplicationErrorCode::ResourceLimitExceeded
            : FaceApplicationErrorCode::InvalidArgument;
    throw FaceApplicationError(code, error.what());
}

[[noreturn]] void translate_face_model_error(
    const kfcore::face_models::FaceModelError& error)
{
    FaceApplicationErrorCode code = FaceApplicationErrorCode::RuntimeFailure;
    switch (error.code())
    {
    case kfcore::face_models::FaceModelErrorCode::InvalidArgument:
    case kfcore::face_models::FaceModelErrorCode::InvalidTensorView:
        code = FaceApplicationErrorCode::InvalidArgument;
        break;
    case kfcore::face_models::FaceModelErrorCode::ModelContractMismatch:
        code = FaceApplicationErrorCode::ModelContractMismatch;
        break;
    case kfcore::face_models::FaceModelErrorCode::ResourceLimitExceeded:
        code = FaceApplicationErrorCode::ResourceLimitExceeded;
        break;
    case kfcore::face_models::FaceModelErrorCode::InvalidModelAsset:
        code = FaceApplicationErrorCode::InvalidModelAsset;
        break;
    case kfcore::face_models::FaceModelErrorCode::ConcurrentExecution:
        code = FaceApplicationErrorCode::ConcurrentExecution;
        break;
    case kfcore::face_models::FaceModelErrorCode::RuntimeFailure:
        break;
    }
    throw FaceApplicationError(code, error.what());
}

[[noreturn]] void translate_runtime_error(const kfcore::runtime::RuntimeError& error)
{
    FaceApplicationErrorCode code = FaceApplicationErrorCode::RuntimeFailure;
    switch (error.code())
    {
    case kfcore::runtime::RuntimeErrorCode::InvalidArgument:
        code = FaceApplicationErrorCode::InvalidArgument;
        break;
    case kfcore::runtime::RuntimeErrorCode::FileIo:
    case kfcore::runtime::RuntimeErrorCode::InvalidModelPackage:
    case kfcore::runtime::RuntimeErrorCode::ArtifactIntegrity:
        code = FaceApplicationErrorCode::InvalidModelAsset;
        break;
    case kfcore::runtime::RuntimeErrorCode::ModuleLoad:
    case kfcore::runtime::RuntimeErrorCode::SymbolLookup:
    case kfcore::runtime::RuntimeErrorCode::AbiMismatch:
    case kfcore::runtime::RuntimeErrorCode::BackendFailure:
    case kfcore::runtime::RuntimeErrorCode::DuplicateBackend:
    case kfcore::runtime::RuntimeErrorCode::NotFound:
    case kfcore::runtime::RuntimeErrorCode::NoCompatibleExecution:
        break;
    }
    throw FaceApplicationError(code, error.what());
}

} // namespace

FaceApplicationError::FaceApplicationError(FaceApplicationErrorCode code,
                                                 std::string message)
    : std::runtime_error(std::move(message))
    , code_(code)
{
}

FaceApplicationErrorCode FaceApplicationError::code() const noexcept
{
    return code_;
}


FaceApplicationPolicies::FaceApplicationPolicies(
    runtime::ExecutionPolicy detector_policy,
    runtime::ExecutionPolicy face68_policy,
    runtime::ExecutionPolicy arcface_policy,
    runtime::ExecutionPolicy inswapper_policy,
    runtime::ExecutionPolicy gfpgan_policy,
    runtime::ExecutionPolicy age_gender_policy)
    : detector(std::move(detector_policy))
    , face68(std::move(face68_policy))
    , arcface(std::move(arcface_policy))
    , inswapper(std::move(inswapper_policy))
    , gfpgan(std::move(gfpgan_policy))
    , age_gender(std::move(age_gender_policy))
{
}

FaceApplicationPolicies FaceApplicationPolicies::onnx_cuda(
    std::uint32_t device_ordinal)
{
    const std::string cuda_device = "cuda:" + std::to_string(device_ordinal);
    const auto cuda_policy = [&]()
    {
        return runtime::ExecutionPolicy::exact("onnxruntime", cuda_device);
    };
    return FaceApplicationPolicies(
        cuda_policy(), cuda_policy(), cuda_policy(), cuda_policy(), cuda_policy(),
        cuda_policy());
}


struct FaceSwapApplication::Impl final
{
    Impl(runtime::Runtime& runtime,
         const FaceApplicationModelPackages& packages,
         const FaceApplicationPolicies& policies,
         const FaceSwapOptions& options_in)
        : options(options_in)
        , detector(kfcore::face_models::FaceDetector::load(
              runtime, runtime::ModelPackage::load(packages.detector),
              policies.detector, detector_options(options)))
        , face68(kfcore::face_models::Face68::load(
              runtime, runtime::ModelPackage::load(packages.face68),
              policies.face68, face_model_options(options)))
        , arcface(kfcore::face_models::ArcFace::load(
              runtime, runtime::ModelPackage::load(packages.arcface),
              policies.arcface, face_model_options(options)))
        , inswapper(kfcore::face_models::InSwapper::load(
              runtime, runtime::ModelPackage::load(packages.inswapper),
              policies.inswapper, face_model_options(options)))
        , projector(std::make_unique<kfcore::face_models::InSwapperEmbeddingProjector>(
              kfcore::face_models::InSwapperEmbeddingProjector::load(
                  packages.inswapper_matrix)))
    {
        if (packages.gfpgan)
        {
            gfpgan = kfcore::face_models::GfpGan::load(
                runtime, runtime::ModelPackage::load(*packages.gfpgan),
                policies.gfpgan, face_model_options(options));
        }
        if (packages.age_gender)
        {
            age_gender = kfcore::face_models::AgeGender::load(
                runtime, runtime::ModelPackage::load(*packages.age_gender),
                policies.age_gender, face_model_options(options));
        }
    }

    [[nodiscard]] FaceAnalysis analyze_detection(
        const kfcore::image::BgrImage& image, const FaceDetection& detection,
        FaceAnalysisTimingReport* timings)
    {
        FaceAnalysis analysis;
        analysis.detection = detection;

        detail::AlignedFace face68_crop;
        {
            StageTimer timer(timings != nullptr ? &timings->face68_preprocess : nullptr);
            face68_crop = detail::crop_face68(image, analysis.detection.box,
                                              options.max_image_bytes);
        }
        {
            StageTimer timer(timings != nullptr
                                 ? &timings->face68_inference_and_postprocess
                                 : nullptr);
            const std::vector<float> tensor =
                detail::preprocess_face68(face68_crop.image, options.max_tensor_bytes);
            analysis.landmarks68 = detail::map_face68_to_source(
                face68->infer({ tensor.data(), tensor.size() }),
                face68_crop.aligned_to_source);
            analysis.landmarks = detail::extract_five_landmarks(analysis.landmarks68);
        }

        detail::AlignedFace arcface_crop;
        std::vector<float> arcface_tensor;
        {
            StageTimer timer(timings != nullptr ? &timings->arcface_preprocess : nullptr);
            arcface_crop = detail::align_face(
                image, analysis.landmarks, detail::arcface_template(),
                static_cast<int>(kfcore::face_models::kArcFaceInputExtent),
                options.max_image_bytes);
            arcface_tensor =
                detail::preprocess_arcface(arcface_crop.image, options.max_tensor_bytes);
        }
        {
            StageTimer timer(timings != nullptr ? &timings->arcface_inference : nullptr);
            analysis.embedding = arcface->infer(
                { arcface_tensor.data(), arcface_tensor.size() });
        }

        if (age_gender != nullptr)
        {
            if (timings != nullptr)
            {
                timings->age_gender.emplace();
            }
            StageTimer timer(timings != nullptr ? &*timings->age_gender : nullptr);
            const kfcore::image::BgrImage crop = crop_detection(
                image, analysis.detection.box, options.max_image_bytes);
            const std::vector<float> tensor = detail::preprocess_age_gender(
                crop, options.max_image_bytes, options.max_tensor_bytes);
            analysis.age_gender_logits = age_gender->infer(
                { tensor.data(), tensor.size() });
        }
        return analysis;
    }

    [[nodiscard]] std::vector<FaceDetection> detect_all(
        const kfcore::image::BgrImage& image)
    {
        validate_image(image, options.max_image_bytes);
        const kfcore::face_models::FaceDetectionsResult detections =
            detector->infer_all(image.view());
        std::vector<FaceDetection> result;
        result.reserve(detections.faces.size());
        for (const kfcore::face_models::FaceDetection& detection : detections.faces)
        {
            result.push_back({
                { detection.box.x, detection.box.y,
                  detection.box.x + detection.box.width,
                  detection.box.y + detection.box.height },
                detection.confidence,
                options.face_class_id,
            });
        }
        return result;
    }

    [[nodiscard]] std::vector<FaceAnalysis> analyze_all_internal(
        const kfcore::image::BgrImage& image)
    {
        const std::vector<FaceDetection> detections = detect_all(image);
        std::vector<FaceAnalysis> result;
        result.reserve(detections.size());
        for (const FaceDetection& detection : detections)
        {
            result.push_back(analyze_detection(image, detection, nullptr));
        }
        return result;
    }

    [[nodiscard]] FaceAnalysis analyze_internal(
        const kfcore::image::BgrImage& image, FaceAnalysisTimingReport* timings)
    {
        if (timings != nullptr)
        {
            *timings = {};
        }
        StageTimer total(timings != nullptr ? &timings->total : nullptr);
        {
            StageTimer timer(timings != nullptr ? &timings->initial_staging : nullptr);
            validate_image(image, options.max_image_bytes);
        }
        std::vector<FaceDetection> detections;
        {
            StageTimer timer(timings != nullptr ? &timings->detection : nullptr);
            detections = detect_all(image);
        }
        const auto best = std::max_element(
            detections.begin(), detections.end(),
            [](const FaceDetection& left, const FaceDetection& right)
            {
                return left.score < right.score;
            });
        if (best == detections.end())
        {
            throw FaceApplicationError(FaceApplicationErrorCode::NoFaceDetected,
                                          "YOLOv12Face found no matching face");
        }
        return analyze_detection(image, *best, timings);
    }

    FaceSwapOptions options;
    std::unique_ptr<kfcore::face_models::FaceDetector> detector;
    std::unique_ptr<kfcore::face_models::Face68>         face68;
    std::unique_ptr<kfcore::face_models::ArcFace>        arcface;
    std::unique_ptr<kfcore::face_models::InSwapper>      inswapper;
    std::unique_ptr<kfcore::face_models::GfpGan>         gfpgan;
    std::unique_ptr<kfcore::face_models::AgeGender>      age_gender;
    std::unique_ptr<kfcore::face_models::InSwapperEmbeddingProjector> projector;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

FaceSwapApplication::FaceSwapApplication(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

FaceSwapApplication::~FaceSwapApplication() = default;

std::unique_ptr<FaceSwapApplication>
FaceSwapApplication::load(runtime::Runtime& runtime,
                          const FaceApplicationModelPackages& packages,
                          const FaceApplicationPolicies& policies,
                          const FaceSwapOptions& options)
{
    validate_paths(packages);
    validate_options(options);
    try
    {
        return std::unique_ptr<FaceSwapApplication>(
            new FaceSwapApplication(
                std::make_unique<Impl>(runtime, packages, policies, options)));
    }
    catch (const FaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        translate_face_model_error(error);
    }
    catch (const kfcore::runtime::RuntimeError& error)
    {
        translate_runtime_error(error);
    }
    catch (const std::bad_alloc&)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ResourceLimitExceeded,
                                   "face application allocation failed");
    }
}

FaceAnalysis FaceSwapApplication::analyze(const kfcore::image::BgrImage& image)
{
    CallGuard guard(impl_->in_use);
    try
    {
        return impl_->analyze_internal(image, nullptr);
    }
    catch (const FaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::image::ImageProcessorError& error)
    {
        translate_image_error(error);
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        translate_face_model_error(error);
    }

    catch (const std::bad_alloc&)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ResourceLimitExceeded,
                                      "face analysis allocation failed");
    }
}

FaceAnalysis FaceSwapApplication::analyze(const kfcore::image::ImageView& image)
{
    CallGuard guard(impl_->in_use);
    try
    {
        const kfcore::image::BgrImage owned = kfcore::image::CpuImageProcessor::copy_bgr(
            image, impl_->options.max_image_bytes);
        return impl_->analyze_internal(owned, nullptr);
    }
    catch (const FaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::image::ImageProcessorError& error)
    {
        translate_image_error(error);
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        translate_face_model_error(error);
    }

    catch (const std::bad_alloc&)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ResourceLimitExceeded,
                                      "face analysis allocation failed");
    }
}

void validate_prepared_inputs(
    const kfcore::face_models::ArcFaceResult& source_embedding,
    const FiveLandmarks& target_landmarks)
{
    if (!std::all_of(source_embedding.begin(), source_embedding.end(),
                     [](float value) { return std::isfinite(value); }))
    {
        throw_invalid("source embedding must contain only finite values");
    }
    for (const Point2f& point : target_landmarks)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y))
        {
            throw_invalid("target landmarks must contain only finite coordinates");
        }
    }
}

std::vector<FaceAnalysis> FaceSwapApplication::analyze_all(
    const kfcore::image::BgrImage& image)
{
    CallGuard guard(impl_->in_use);
    try
    {
        return impl_->analyze_all_internal(image);
    }
    catch (const FaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::image::ImageProcessorError& error)
    {
        translate_image_error(error);
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        translate_face_model_error(error);
    }
    catch (const std::bad_alloc&)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ResourceLimitExceeded,
                                      "CPU all-face analysis allocation failed");
    }
}

std::vector<FaceAnalysis> FaceSwapApplication::analyze_all(
    const kfcore::image::ImageView& image)
{
    CallGuard guard(impl_->in_use);
    try
    {
        const kfcore::image::BgrImage owned = kfcore::image::CpuImageProcessor::copy_bgr(
            image, impl_->options.max_image_bytes);
        return impl_->analyze_all_internal(owned);
    }
    catch (const FaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::image::ImageProcessorError& error)
    {
        translate_image_error(error);
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        translate_face_model_error(error);
    }
    catch (const std::bad_alloc&)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ResourceLimitExceeded,
                                      "CPU all-face analysis allocation failed");
    }
}

kfcore::image::BgrImage FaceSwapApplication::swap(
    const kfcore::image::BgrImage& source,
    const kfcore::image::BgrImage& target)
{
    CallGuard guard(impl_->in_use);
    try
    {
        return swap_internal(source, target, nullptr);
    }
    catch (const FaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::image::ImageProcessorError& error)
    {
        translate_image_error(error);
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        translate_face_model_error(error);
    }

    catch (const std::bad_alloc&)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ResourceLimitExceeded,
                                      "face swap allocation failed");
    }
}

kfcore::image::BgrImage FaceSwapApplication::swap(
    const kfcore::image::ImageView& source,
    const kfcore::image::ImageView& target)
{
    CallGuard guard(impl_->in_use);
    try
    {
        const kfcore::image::BgrImage source_bgr =
            kfcore::image::CpuImageProcessor::copy_bgr(
                source, impl_->options.max_image_bytes);
        const kfcore::image::BgrImage target_bgr =
            kfcore::image::CpuImageProcessor::copy_bgr(
                target, impl_->options.max_image_bytes);
        return swap_internal(source_bgr, target_bgr, nullptr);
    }
    catch (const FaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::image::ImageProcessorError& error)
    {
        translate_image_error(error);
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        translate_face_model_error(error);
    }

    catch (const std::bad_alloc&)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ResourceLimitExceeded,
                                      "face swap allocation failed");
    }
}

ProfiledFaceSwapResult FaceSwapApplication::swap_profiled(
    const kfcore::image::BgrImage& source,
    const kfcore::image::BgrImage& target)
{
    CallGuard guard(impl_->in_use);
    try
    {
        ProfiledFaceSwapResult result;
        result.image = swap_internal(source, target, &result.timings);
        return result;
    }
    catch (const FaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::image::ImageProcessorError& error)
    {
        translate_image_error(error);
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        translate_face_model_error(error);
    }

    catch (const std::bad_alloc&)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ResourceLimitExceeded,
                                      "profiled face swap allocation failed");
    }
}

ProfiledFaceSwapResult FaceSwapApplication::swap_profiled(
    const kfcore::image::ImageView& source,
    const kfcore::image::ImageView& target)
{
    CallGuard guard(impl_->in_use);
    try
    {
        const auto source_started = std::chrono::steady_clock::now();
        const kfcore::image::BgrImage source_bgr =
            kfcore::image::CpuImageProcessor::copy_bgr(
                source, impl_->options.max_image_bytes);
        const FaceSwapDuration source_conversion =
            std::chrono::duration_cast<FaceSwapDuration>(
                std::chrono::steady_clock::now() - source_started);

        const auto target_started = std::chrono::steady_clock::now();
        const kfcore::image::BgrImage target_bgr =
            kfcore::image::CpuImageProcessor::copy_bgr(
                target, impl_->options.max_image_bytes);
        const FaceSwapDuration target_conversion =
            std::chrono::duration_cast<FaceSwapDuration>(
                std::chrono::steady_clock::now() - target_started);

        ProfiledFaceSwapResult result;
        result.image = swap_internal(source_bgr, target_bgr, &result.timings);
        result.timings.source_analysis.initial_staging += source_conversion;
        result.timings.source_analysis.total += source_conversion;
        result.timings.target_analysis.initial_staging += target_conversion;
        result.timings.target_analysis.total += target_conversion;
        result.timings.total += source_conversion + target_conversion;
        return result;
    }
    catch (const FaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::image::ImageProcessorError& error)
    {
        translate_image_error(error);
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        translate_face_model_error(error);
    }

    catch (const std::bad_alloc&)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ResourceLimitExceeded,
                                      "profiled face swap allocation failed");
    }
}

kfcore::image::BgrImage FaceSwapApplication::swap_prepared(
    const kfcore::image::BgrImage& target,
    const kfcore::face_models::ArcFaceResult& source_embedding,
    const FiveLandmarks& target_landmarks, bool enhance)
{
    return swap_prepared(target, source_embedding, target_landmarks, enhance,
                         impl_->options.enhancer_blend);
}

kfcore::image::BgrImage FaceSwapApplication::swap_prepared(
    const kfcore::image::BgrImage& target,
    const kfcore::face_models::ArcFaceResult& source_embedding,
    const FiveLandmarks& target_landmarks, bool enhance, float enhancer_blend)
{
    CallGuard guard(impl_->in_use);
    try
    {
        return swap_prepared_internal(target, source_embedding, target_landmarks,
                                      enhance, enhancer_blend, nullptr);
    }
    catch (const FaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::image::ImageProcessorError& error)
    {
        translate_image_error(error);
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        translate_face_model_error(error);
    }
    catch (const std::bad_alloc&)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ResourceLimitExceeded,
                                      "prepared face swap allocation failed");
    }
}

kfcore::image::BgrImage FaceSwapApplication::swap_prepared(
    const kfcore::image::ImageView& target,
    const kfcore::face_models::ArcFaceResult& source_embedding,
    const FiveLandmarks& target_landmarks, bool enhance)
{
    return swap_prepared(target, source_embedding, target_landmarks, enhance,
                         impl_->options.enhancer_blend);
}

kfcore::image::BgrImage FaceSwapApplication::swap_prepared(
    const kfcore::image::ImageView& target,
    const kfcore::face_models::ArcFaceResult& source_embedding,
    const FiveLandmarks& target_landmarks, bool enhance, float enhancer_blend)
{
    CallGuard guard(impl_->in_use);
    try
    {
        const kfcore::image::BgrImage owned = kfcore::image::CpuImageProcessor::copy_bgr(
            target, impl_->options.max_image_bytes);
        return swap_prepared_internal(owned, source_embedding, target_landmarks,
                                      enhance, enhancer_blend, nullptr);
    }
    catch (const FaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::image::ImageProcessorError& error)
    {
        translate_image_error(error);
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        translate_face_model_error(error);
    }
    catch (const std::bad_alloc&)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ResourceLimitExceeded,
                                      "prepared face swap allocation failed");
    }
}

kfcore::image::BgrImage FaceSwapApplication::swap_internal(
    const kfcore::image::BgrImage& source,
    const kfcore::image::BgrImage& target,
    FaceSwapTimingReport* timings)
{
    if (timings != nullptr)
    {
        *timings = {};
    }
    StageTimer total(timings != nullptr ? &timings->total : nullptr);
    const FaceAnalysis source_analysis = impl_->analyze_internal(
        source, timings != nullptr ? &timings->source_analysis : nullptr);
    const FaceAnalysis target_analysis = impl_->analyze_internal(
        target, timings != nullptr ? &timings->target_analysis : nullptr);
    return swap_prepared_internal(target, source_analysis.embedding,
                                  target_analysis.landmarks,
                                  impl_->gfpgan != nullptr,
                                  impl_->options.enhancer_blend, timings);
}

kfcore::image::BgrImage FaceSwapApplication::swap_prepared_internal(
    const kfcore::image::BgrImage& target,
    const kfcore::face_models::ArcFaceResult& source_embedding,
    const FiveLandmarks& target_landmarks, bool enhance, float enhancer_blend,
    FaceSwapTimingReport* timings)
{
    validate_image(target, impl_->options.max_image_bytes);
    validate_prepared_inputs(source_embedding, target_landmarks);
    if (!std::isfinite(enhancer_blend) || enhancer_blend < 0.0F ||
        enhancer_blend > 1.0F)
    {
        throw_invalid("enhancer_blend must be finite within [0,1]");
    }
    if (enhance && impl_->gfpgan == nullptr)
    {
        throw_invalid("face enhancement was requested without a GFPGAN model");
    }
    if (impl_->inswapper == nullptr || impl_->projector == nullptr)
    {
        throw_invalid("face swap was requested without InSwapper assets");
    }

    kfcore::face_models::ArcFaceResult projected_embedding;
    {
        StageTimer timer(timings != nullptr ? &timings->embedding_projection : nullptr);
        projected_embedding = impl_->projector->project(source_embedding);
    }

    detail::AlignedFace inswapper_crop;
    std::vector<float> inswapper_tensor;
    {
        StageTimer timer(timings != nullptr ? &timings->inswapper_preprocess : nullptr);
        inswapper_crop = detail::align_face(
            target, target_landmarks, detail::inswapper_template(),
            static_cast<int>(kfcore::face_models::kInSwapperInputExtent),
            impl_->options.max_image_bytes);
        inswapper_tensor = detail::preprocess_inswapper(
            inswapper_crop.image, impl_->options.max_tensor_bytes);
    }

    kfcore::image::BgrImage swapped_face;
    {
        StageTimer timer(timings != nullptr
                             ? &timings->inswapper_inference_and_decode
                             : nullptr);
        const kfcore::face_models::InSwapperResult output = impl_->inswapper->infer(
            { inswapper_tensor.data(), inswapper_tensor.size() },
            { projected_embedding.data(), projected_embedding.size() });
        swapped_face = detail::decode_rgb_chw(
            output.values,
            static_cast<int>(kfcore::face_models::kInSwapperInputExtent), false);
    }

    kfcore::image::BgrImage result;
    {
        StageTimer timer(timings != nullptr ? &timings->inswapper_composition : nullptr);
        const detail::FloatMask mask = detail::create_static_box_mask(
            swapped_face.width, swapped_face.height);
        result = detail::paste_back(target, swapped_face, mask,
                                    inswapper_crop.aligned_to_source,
                                    impl_->options.max_image_bytes);
    }

    if (enhance)
    {
        if (timings != nullptr)
        {
            timings->gfpgan_preprocess.emplace();
            timings->gfpgan_inference_and_decode.emplace();
            timings->gfpgan_composition.emplace();
        }
        detail::AlignedFace gfpgan_crop;
        std::vector<float> gfpgan_tensor;
        {
            StageTimer timer(timings != nullptr ? &*timings->gfpgan_preprocess : nullptr);
            gfpgan_crop = detail::align_face(
                result, target_landmarks, detail::gfpgan_template(),
                static_cast<int>(kfcore::face_models::kGfpGanInputExtent),
                impl_->options.max_image_bytes);
            gfpgan_tensor = detail::preprocess_gfpgan(
                gfpgan_crop.image, impl_->options.max_tensor_bytes);
        }
        kfcore::image::BgrImage enhanced_face;
        {
            StageTimer timer(timings != nullptr
                                 ? &*timings->gfpgan_inference_and_decode
                                 : nullptr);
            const kfcore::face_models::GfpGanResult output = impl_->gfpgan->infer(
                { gfpgan_tensor.data(), gfpgan_tensor.size() });
            enhanced_face = detail::decode_rgb_chw(
                output.values,
                static_cast<int>(kfcore::face_models::kGfpGanInputExtent), true);
        }
        {
            StageTimer timer(timings != nullptr ? &*timings->gfpgan_composition : nullptr);
            detail::FaceMaskOptions mask_options;
            mask_options.blur_fraction = 0.3F;
            mask_options.padding_percent = { 0, 0, 0, 0 };
            const detail::FloatMask mask = detail::create_static_box_mask(
                enhanced_face.width, enhanced_face.height, mask_options);
            const kfcore::image::BgrImage enhanced = detail::paste_back(
                result, enhanced_face, mask, gfpgan_crop.aligned_to_source,
                impl_->options.max_image_bytes);
            result = detail::blend_images(result, enhanced,
                                          enhancer_blend,
                                          impl_->options.max_image_bytes);
        }
    }
    return result;
}


FaceApplicationExecutionRoutes FaceSwapApplication::execution_routes() const
{
    if (impl_ == nullptr)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::RuntimeFailure,
                                   "face application state is unavailable");
    }
    FaceApplicationExecutionRoutes routes{
        impl_->detector->execution_route(),
        impl_->face68->execution_route(),
        impl_->arcface->execution_route(),
        impl_->inswapper->execution_route(),
        std::nullopt,
        std::nullopt,
    };
    if (impl_->gfpgan != nullptr)
    {
        routes.gfpgan = impl_->gfpgan->execution_route();
    }
    if (impl_->age_gender != nullptr)
    {
        routes.age_gender = impl_->age_gender->execution_route();
    }
    return routes;
}

} // namespace kfcore::face_applications
