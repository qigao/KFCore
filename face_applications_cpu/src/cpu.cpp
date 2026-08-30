#include "kfcore/face_applications/cpu.hpp"

#include "kfcore/face_models/cpu.hpp"
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
    explicit StageTimer(CpuFaceSwapDuration* destination)
        : destination_(destination)
        , started_(std::chrono::steady_clock::now())
    {
    }

    ~StageTimer()
    {
        if (destination_ != nullptr)
        {
            *destination_ = std::chrono::duration_cast<CpuFaceSwapDuration>(
                std::chrono::steady_clock::now() - started_);
        }
    }

private:
    CpuFaceSwapDuration* destination_;
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
            throw CpuFaceApplicationError(
                CpuFaceApplicationErrorCode::ConcurrentExecution,
                "CPU face application does not support concurrent calls on one instance");
        }
    }

    ~CallGuard() { flag_.clear(std::memory_order_release); }

private:
    std::atomic_flag& flag_;
};

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::InvalidArgument,
                                  "CPU face application validation stage: " + detail);
}

void validate_paths(const CpuFaceApplicationModelPaths& paths)
{
    const std::pair<const std::filesystem::path*, const char*> required[] = {
        { &paths.detector_model, "detector_model" },
        { &paths.face68_model, "face68_model" },
        { &paths.arcface_model, "arcface_model" },
        { &paths.inswapper_model, "inswapper_model" },
        { &paths.inswapper_matrix, "inswapper_matrix" },
    };
    for (const auto& entry : required)
    {
        if (entry.first->empty())
        {
            throw_invalid(std::string(entry.second) + " path must not be empty");
        }
    }
    if ((paths.gfpgan_model && paths.gfpgan_model->empty()) ||
        (paths.age_gender_model && paths.age_gender_model->empty()))
    {
        throw_invalid("optional model paths must not be empty when configured");
    }
}

void validate_options(const CpuFaceSwapOptions& options)
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
    if (options.intra_op_threads < 0 || options.inter_op_threads < 0)
    {
        throw_invalid("ONNX Runtime thread counts must not be negative");
    }
    if (options.max_image_bytes == 0U || options.max_tensor_bytes == 0U ||
        options.max_model_bytes == 0U)
    {
        throw_invalid("resource byte limits must be positive");
    }
}

kfcore::face_models::CpuFaceModelOptions
face_model_options(const CpuFaceSwapOptions& options)
{
    kfcore::face_models::CpuFaceModelOptions result;
    result.intra_op_threads = options.intra_op_threads;
    result.inter_op_threads = options.inter_op_threads;
    result.max_model_bytes  = options.max_model_bytes;
    return result;
}

kfcore::face_models::CpuFaceMeshOptions
detector_options(const CpuFaceSwapOptions& options)
{
    kfcore::face_models::CpuFaceMeshOptions result;
    result.intra_op_threads = options.intra_op_threads;
    result.inter_op_threads = options.inter_op_threads;
    result.max_model_bytes  = options.max_model_bytes;
    result.max_source_bytes = options.max_image_bytes;
    result.max_tensor_bytes = options.max_tensor_bytes;
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
                                       const CpuFaceBox& box,
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
    const CpuFaceApplicationErrorCode code =
        error.code() == kfcore::image::ImageProcessorErrorCode::ResourceLimitExceeded
            ? CpuFaceApplicationErrorCode::ResourceLimitExceeded
            : CpuFaceApplicationErrorCode::InvalidArgument;
    throw CpuFaceApplicationError(code, error.what());
}

[[noreturn]] void translate_face_model_error(
    const kfcore::face_models::FaceModelError& error)
{
    CpuFaceApplicationErrorCode code = CpuFaceApplicationErrorCode::RuntimeFailure;
    switch (error.code())
    {
    case kfcore::face_models::FaceModelErrorCode::InvalidArgument:
    case kfcore::face_models::FaceModelErrorCode::InvalidTensorView:
        code = CpuFaceApplicationErrorCode::InvalidArgument;
        break;
    case kfcore::face_models::FaceModelErrorCode::ModelContractMismatch:
        code = CpuFaceApplicationErrorCode::ModelContractMismatch;
        break;
    case kfcore::face_models::FaceModelErrorCode::ResourceLimitExceeded:
        code = CpuFaceApplicationErrorCode::ResourceLimitExceeded;
        break;
    case kfcore::face_models::FaceModelErrorCode::InvalidModelAsset:
        code = CpuFaceApplicationErrorCode::InvalidModelAsset;
        break;
    case kfcore::face_models::FaceModelErrorCode::ConcurrentExecution:
        code = CpuFaceApplicationErrorCode::ConcurrentExecution;
        break;
    case kfcore::face_models::FaceModelErrorCode::RuntimeFailure:
        break;
    }
    throw CpuFaceApplicationError(code, error.what());
}

} // namespace

CpuFaceApplicationError::CpuFaceApplicationError(CpuFaceApplicationErrorCode code,
                                                 std::string message)
    : std::runtime_error(std::move(message))
    , code_(code)
{
}

CpuFaceApplicationErrorCode CpuFaceApplicationError::code() const noexcept
{
    return code_;
}

struct OnnxFaceSwapApplication::Impl final
{
    Impl(const CpuFaceApplicationModelPaths& paths, const CpuFaceSwapOptions& options_in)
        : options(options_in)
        , detector(kfcore::face_models::CpuFaceDetector::load(
              paths.detector_model, detector_options(options)))
        , face68(kfcore::face_models::CpuFace68::load(
              paths.face68_model, face_model_options(options)))
        , arcface(kfcore::face_models::CpuArcFace::load(
              paths.arcface_model, face_model_options(options)))
        , inswapper(kfcore::face_models::CpuInSwapper::load(
              paths.inswapper_model, face_model_options(options)))
    {
        if (paths.gfpgan_model)
        {
            gfpgan = kfcore::face_models::CpuGfpGan::load(
                *paths.gfpgan_model, face_model_options(options));
        }
        if (paths.age_gender_model)
        {
            age_gender = kfcore::face_models::CpuAgeGender::load(
                *paths.age_gender_model, face_model_options(options));
        }
        projector = std::make_unique<kfcore::face_models::InSwapperEmbeddingProjector>(
            kfcore::face_models::InSwapperEmbeddingProjector::load(paths.inswapper_matrix));
    }

    [[nodiscard]] CpuFaceAnalysis analyze_internal(
        const kfcore::image::BgrImage& image, CpuFaceAnalysisTimingReport* timings)
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

        CpuFaceAnalysis analysis;
        {
            StageTimer timer(timings != nullptr ? &timings->detection : nullptr);
            const kfcore::face_models::FaceDetectionResult detection =
                detector->infer(image.view());
            if (!detection.face)
            {
                throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::NoFaceDetected,
                                              "YOLOv12Face found no matching face");
            }
            analysis.detection.box = {
                detection.face->box.x,
                detection.face->box.y,
                detection.face->box.x + detection.face->box.width,
                detection.face->box.y + detection.face->box.height,
            };
            analysis.detection.score = detection.face->confidence;
            analysis.detection.class_id = options.face_class_id;
        }

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

    CpuFaceSwapOptions options;
    std::unique_ptr<kfcore::face_models::CpuFaceDetector> detector;
    std::unique_ptr<kfcore::face_models::CpuFace68>         face68;
    std::unique_ptr<kfcore::face_models::CpuArcFace>        arcface;
    std::unique_ptr<kfcore::face_models::CpuInSwapper>      inswapper;
    std::unique_ptr<kfcore::face_models::CpuGfpGan>         gfpgan;
    std::unique_ptr<kfcore::face_models::CpuAgeGender>      age_gender;
    std::unique_ptr<kfcore::face_models::InSwapperEmbeddingProjector> projector;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

OnnxFaceSwapApplication::OnnxFaceSwapApplication(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

OnnxFaceSwapApplication::~OnnxFaceSwapApplication() = default;

std::unique_ptr<OnnxFaceSwapApplication>
OnnxFaceSwapApplication::load(const CpuFaceApplicationModelPaths& paths,
                              const CpuFaceSwapOptions& options)
{
    validate_paths(paths);
    validate_options(options);
    try
    {
        return std::unique_ptr<OnnxFaceSwapApplication>(
            new OnnxFaceSwapApplication(std::make_unique<Impl>(paths, options)));
    }
    catch (const CpuFaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        translate_face_model_error(error);
    }

    catch (const std::bad_alloc&)
    {
        throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::ResourceLimitExceeded,
                                      "CPU face application allocation failed");
    }
}

CpuFaceAnalysis OnnxFaceSwapApplication::analyze(const kfcore::image::BgrImage& image)
{
    CallGuard guard(impl_->in_use);
    try
    {
        return impl_->analyze_internal(image, nullptr);
    }
    catch (const CpuFaceApplicationError&)
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
        throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::ResourceLimitExceeded,
                                      "CPU face analysis allocation failed");
    }
}

CpuFaceAnalysis OnnxFaceSwapApplication::analyze(const kfcore::image::ImageView& image)
{
    CallGuard guard(impl_->in_use);
    try
    {
        const kfcore::image::BgrImage owned = kfcore::image::CpuImageProcessor::copy_bgr(
            image, impl_->options.max_image_bytes);
        return impl_->analyze_internal(owned, nullptr);
    }
    catch (const CpuFaceApplicationError&)
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
        throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::ResourceLimitExceeded,
                                      "CPU face analysis allocation failed");
    }
}

kfcore::image::BgrImage OnnxFaceSwapApplication::swap(
    const kfcore::image::BgrImage& source,
    const kfcore::image::BgrImage& target)
{
    CallGuard guard(impl_->in_use);
    try
    {
        return swap_internal(source, target, nullptr);
    }
    catch (const CpuFaceApplicationError&)
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
        throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::ResourceLimitExceeded,
                                      "CPU face swap allocation failed");
    }
}

kfcore::image::BgrImage OnnxFaceSwapApplication::swap(
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
    catch (const CpuFaceApplicationError&)
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
        throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::ResourceLimitExceeded,
                                      "CPU face swap allocation failed");
    }
}

ProfiledCpuFaceSwapResult OnnxFaceSwapApplication::swap_profiled(
    const kfcore::image::BgrImage& source,
    const kfcore::image::BgrImage& target)
{
    CallGuard guard(impl_->in_use);
    try
    {
        ProfiledCpuFaceSwapResult result;
        result.image = swap_internal(source, target, &result.timings);
        return result;
    }
    catch (const CpuFaceApplicationError&)
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
        throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::ResourceLimitExceeded,
                                      "profiled CPU face swap allocation failed");
    }
}

ProfiledCpuFaceSwapResult OnnxFaceSwapApplication::swap_profiled(
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
        const CpuFaceSwapDuration source_conversion =
            std::chrono::duration_cast<CpuFaceSwapDuration>(
                std::chrono::steady_clock::now() - source_started);

        const auto target_started = std::chrono::steady_clock::now();
        const kfcore::image::BgrImage target_bgr =
            kfcore::image::CpuImageProcessor::copy_bgr(
                target, impl_->options.max_image_bytes);
        const CpuFaceSwapDuration target_conversion =
            std::chrono::duration_cast<CpuFaceSwapDuration>(
                std::chrono::steady_clock::now() - target_started);

        ProfiledCpuFaceSwapResult result;
        result.image = swap_internal(source_bgr, target_bgr, &result.timings);
        result.timings.source_analysis.initial_staging += source_conversion;
        result.timings.source_analysis.total += source_conversion;
        result.timings.target_analysis.initial_staging += target_conversion;
        result.timings.target_analysis.total += target_conversion;
        result.timings.total += source_conversion + target_conversion;
        return result;
    }
    catch (const CpuFaceApplicationError&)
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
        throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::ResourceLimitExceeded,
                                      "profiled CPU face swap allocation failed");
    }
}

kfcore::image::BgrImage OnnxFaceSwapApplication::swap_internal(
    const kfcore::image::BgrImage& source,
    const kfcore::image::BgrImage& target,
    CpuFaceSwapTimingReport* timings)
{
    if (timings != nullptr)
    {
        *timings = {};
    }
    StageTimer total(timings != nullptr ? &timings->total : nullptr);
    const CpuFaceAnalysis source_analysis = impl_->analyze_internal(
        source, timings != nullptr ? &timings->source_analysis : nullptr);
    const CpuFaceAnalysis target_analysis = impl_->analyze_internal(
        target, timings != nullptr ? &timings->target_analysis : nullptr);

    kfcore::face_models::ArcFaceResult projected_embedding;
    {
        StageTimer timer(timings != nullptr ? &timings->embedding_projection : nullptr);
        projected_embedding = impl_->projector->project(source_analysis.embedding);
    }

    detail::AlignedFace inswapper_crop;
    std::vector<float> inswapper_tensor;
    {
        StageTimer timer(timings != nullptr ? &timings->inswapper_preprocess : nullptr);
        inswapper_crop = detail::align_face(
            target, target_analysis.landmarks, detail::inswapper_template(),
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

    if (impl_->gfpgan != nullptr)
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
                result, target_analysis.landmarks, detail::gfpgan_template(),
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
                                          impl_->options.enhancer_blend,
                                          impl_->options.max_image_bytes);
        }
    }
    return result;
}

} // namespace kfcore::face_applications
