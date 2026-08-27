#include "kfcore/face_applications/tensorrt.hpp"

#include "kfcore/face_applications/composer.hpp"
#include "kfcore/face_applications/error.hpp"
#include "kfcore/face_applications/preprocess.hpp"
#include "kfcore/yolo/error.hpp"
#include "kfcore/yolo/opencv.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::face_applications
{
namespace
{

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw FaceApplicationError(FaceApplicationErrorCode::InvalidArgument,
                               "face application configuration stage: " + detail);
}

void validate_options(const FaceSwapOptions& options)
{
    if (!std::isfinite(options.detector_score_threshold) ||
        options.detector_score_threshold < 0.0F || options.detector_score_threshold > 1.0F)
    {
        throw_invalid("detector score threshold must be in [0,1]");
    }
    if (options.face_class_id < 0)
    {
        throw_invalid("face class id must be non-negative");
    }
    if (!std::isfinite(options.enhancer_blend) || options.enhancer_blend < 0.0F ||
        options.enhancer_blend > 1.0F)
    {
        throw_invalid("enhancer blend must be in [0,1]");
    }
}

void validate_paths(const FaceApplicationModelPaths& paths)
{
    if (paths.detector_engine.empty())
    {
        throw_invalid("12face detector engine path must not be empty");
    }
    if (paths.face68_engine.empty())
    {
        throw_invalid("Face68 engine path must not be empty");
    }
    if (paths.arcface_engine.empty())
    {
        throw_invalid("ArcFace engine path must not be empty");
    }
    if (paths.inswapper_engine.empty())
    {
        throw_invalid("InSwapper engine path must not be empty");
    }
    if (paths.inswapper_matrix.empty())
    {
        throw_invalid("InSwapper matrix path must not be empty");
    }
    if (paths.gfpgan_engine && paths.gfpgan_engine->empty())
    {
        throw_invalid("optional GFPGAN engine path must not be empty when present");
    }
    if (paths.age_gender_engine && paths.age_gender_engine->empty())
    {
        throw_invalid("optional AgeGender engine path must not be empty when present");
    }
}

FaceApplicationErrorCode map_model_error(kfcore::face_models::FaceModelErrorCode code)
{
    using kfcore::face_models::FaceModelErrorCode;
    switch (code)
    {
    case FaceModelErrorCode::InvalidArgument:
    case FaceModelErrorCode::InvalidTensorView:
        return FaceApplicationErrorCode::InvalidArgument;
    case FaceModelErrorCode::InvalidModelAsset:
        return FaceApplicationErrorCode::InvalidModelMatrix;
    case FaceModelErrorCode::ModelContractMismatch:
        return FaceApplicationErrorCode::ModelContractMismatch;
    case FaceModelErrorCode::ResourceLimitExceeded:
        return FaceApplicationErrorCode::ResourceLimitExceeded;
    case FaceModelErrorCode::RuntimeFailure:
        return FaceApplicationErrorCode::RuntimeFailure;
    }
    return FaceApplicationErrorCode::RuntimeFailure;
}

[[noreturn]] void rethrow_model(const kfcore::face_models::FaceModelError& error,
                                const char* stage)
{
    throw FaceApplicationError(map_model_error(error.code()),
                               std::string("face application ") + stage + " stage: " +
                                   error.what());
}

[[noreturn]] void rethrow_yolo(const kfcore::yolo::YoloError& error, const char* stage)
{
    throw FaceApplicationError(FaceApplicationErrorCode::RuntimeFailure,
                               std::string("face application ") + stage + " stage: " +
                                   error.what());
}

kfcore::tensorrt::TensorView image_tensor(const std::string& name, std::int64_t extent,
                                          const std::vector<float>& values)
{
    return { name,
             kfcore::tensorrt::DataType::Float32,
             { 1, kfcore::face_models::kFaceModelInputChannels, extent, extent },
             values.data(),
             values.size() * sizeof(float),
             kfcore::tensorrt::MemoryKind::Host };
}

kfcore::tensorrt::TensorView embedding_tensor(const std::string& name,
                                              const kfcore::face_models::ArcFaceResult& values)
{
    return { name,
             kfcore::tensorrt::DataType::Float32,
             { 1, static_cast<std::int64_t>(values.size()) },
             values.data(),
             values.size() * sizeof(float),
             kfcore::tensorrt::MemoryKind::Host };
}

FaceBox face_box(const kfcore::yolo::Detection& detection)
{
    return { detection.box.left, detection.box.top, detection.box.right,
             detection.box.bottom };
}

cv::Mat face_roi(const cv::Mat& image, const kfcore::yolo::Detection& detection)
{
    const int left = (std::clamp)(static_cast<int>(std::floor(detection.box.left)), 0,
                                  image.cols - 1);
    const int top = (std::clamp)(static_cast<int>(std::floor(detection.box.top)), 0,
                                 image.rows - 1);
    const int right = (std::clamp)(static_cast<int>(std::ceil(detection.box.right)), left + 1,
                                   image.cols);
    const int bottom = (std::clamp)(static_cast<int>(std::ceil(detection.box.bottom)), top + 1,
                                    image.rows);
    return image(cv::Rect(left, top, right - left, bottom - top));
}

class ApplicationCallGuard final
{
public:
    explicit ApplicationCallGuard(std::atomic_flag& in_use)
        : in_use_(in_use)
    {
        if (in_use_.test_and_set(std::memory_order_acquire))
        {
            throw FaceApplicationError(
                FaceApplicationErrorCode::InvalidArgument,
                "face application call stage: overlapping calls on one instance are unsupported");
        }
    }

    ~ApplicationCallGuard()
    {
        in_use_.clear(std::memory_order_release);
    }

    ApplicationCallGuard(const ApplicationCallGuard&) = delete;
    ApplicationCallGuard& operator=(const ApplicationCallGuard&) = delete;

private:
    std::atomic_flag& in_use_;
};

} // namespace

std::optional<kfcore::yolo::Detection> select_highest_score_face(
    const std::vector<kfcore::yolo::Detection>& detections, int face_class_id,
    float score_threshold)
{
    std::optional<kfcore::yolo::Detection> result;
    for (const kfcore::yolo::Detection& detection : detections)
    {
        if (detection.class_id != face_class_id || !std::isfinite(detection.score) ||
            detection.score < score_threshold)
        {
            continue;
        }
        if (!result || detection.score > result->score)
        {
            result = detection;
        }
    }
    return result;
}

struct TensorRtFaceSwapApplication::Impl final
{
    Impl(FaceSwapOptions options_in,
         std::shared_ptr<const kfcore::yolo::Engine> detector_engine_in,
         std::unique_ptr<kfcore::yolo::TensorRtDetector> detector_in,
         std::unique_ptr<kfcore::face_models::TensorRtFace68> face68_in,
         std::unique_ptr<kfcore::face_models::TensorRtArcFace> arcface_in,
         std::unique_ptr<kfcore::face_models::TensorRtInSwapper> inswapper_in,
         kfcore::face_models::InSwapperEmbeddingProjector projector_in,
         std::unique_ptr<kfcore::face_models::TensorRtGfpGan> gfpgan_in,
         std::unique_ptr<kfcore::face_models::TensorRtAgeGender> age_gender_in)
        : options(std::move(options_in))
        , detector_engine(std::move(detector_engine_in))
        , detector(std::move(detector_in))
        , face68(std::move(face68_in))
        , arcface(std::move(arcface_in))
        , inswapper(std::move(inswapper_in))
        , projector(std::move(projector_in))
        , gfpgan(std::move(gfpgan_in))
        , age_gender(std::move(age_gender_in))
    {
    }

    FaceAnalysis analyze_internal(const cv::Mat& bgr_image)
    {
        const kfcore::yolo::DetectionFrame frame =
            detector->detect(kfcore::yolo::image_view(bgr_image));
        const std::optional<kfcore::yolo::Detection> detection =
            select_highest_score_face(frame.detections, options.face_class_id,
                                      options.detector_score_threshold);
        if (!detection)
        {
            throw FaceApplicationError(FaceApplicationErrorCode::NoFaceDetected,
                                       "face application detect stage: 12face found no qualifying face");
        }

        const AlignedFace face68_aligned = crop_face68(bgr_image, face_box(*detection));
        const std::vector<float> face68_values = preprocess_face68(face68_aligned.image);
        const std::vector<kfcore::face_models::Face68Result> face68_results =
            face68->infer(image_tensor(options.face68.input_name,
                                       kfcore::face_models::kFace68InputExtent,
                                       face68_values));
        if (face68_results.size() != 1U)
        {
            throw FaceApplicationError(FaceApplicationErrorCode::RuntimeFailure,
                                       "face application landmark stage: Face68 returned a non-unit batch");
        }
        const kfcore::face_models::Face68Result landmarks68 =
            map_face68_to_source(face68_results.front(), face68_aligned.aligned_to_source);
        const FiveLandmarks landmarks = extract_five_landmarks(landmarks68);

        const AlignedFace arcface_aligned =
            align_face(bgr_image, landmarks, arcface_template(),
                       static_cast<int>(kfcore::face_models::kArcFaceInputExtent));
        const std::vector<float> arcface_values = preprocess_arcface(arcface_aligned.image);
        const std::vector<kfcore::face_models::ArcFaceResult> embeddings =
            arcface->infer(image_tensor(options.arcface.input_name,
                                        kfcore::face_models::kArcFaceInputExtent,
                                        arcface_values));
        if (embeddings.size() != 1U)
        {
            throw FaceApplicationError(FaceApplicationErrorCode::RuntimeFailure,
                                       "face application embedding stage: ArcFace returned a non-unit batch");
        }

        FaceAnalysis result { *detection, landmarks68, landmarks, embeddings.front(),
                              std::nullopt };
        if (age_gender)
        {
            const std::vector<float> age_values = preprocess_age_gender(face_roi(bgr_image, *detection));
            const auto logits = age_gender->infer(
                image_tensor(options.age_gender.input_name,
                             kfcore::face_models::kAgeGenderInputExtent, age_values));
            if (logits.size() != 1U)
            {
                throw FaceApplicationError(
                    FaceApplicationErrorCode::RuntimeFailure,
                    "face application age gender stage: model returned a non-unit batch");
            }
            result.age_gender_logits = logits.front();
        }
        return result;
    }

    FaceSwapOptions options;
    std::shared_ptr<const kfcore::yolo::Engine> detector_engine;
    std::unique_ptr<kfcore::yolo::TensorRtDetector> detector;
    std::unique_ptr<kfcore::face_models::TensorRtFace68> face68;
    std::unique_ptr<kfcore::face_models::TensorRtArcFace> arcface;
    std::unique_ptr<kfcore::face_models::TensorRtInSwapper> inswapper;
    kfcore::face_models::InSwapperEmbeddingProjector projector;
    std::unique_ptr<kfcore::face_models::TensorRtGfpGan> gfpgan;
    std::unique_ptr<kfcore::face_models::TensorRtAgeGender> age_gender;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

TensorRtFaceSwapApplication::TensorRtFaceSwapApplication(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

TensorRtFaceSwapApplication::~TensorRtFaceSwapApplication() = default;

std::unique_ptr<TensorRtFaceSwapApplication>
TensorRtFaceSwapApplication::load(const FaceApplicationModelPaths& paths,
                                  const FaceSwapOptions& options)
{
    validate_options(options);
    validate_paths(paths);
    try
    {
        auto detector_engine = kfcore::yolo::Engine::load(paths.detector_engine,
                                                          options.detector_engine);
        auto detector = detector_engine->create_detector(options.detector);
        auto face68 = kfcore::face_models::TensorRtFace68::load(paths.face68_engine,
                                                                options.face68);
        auto arcface = kfcore::face_models::TensorRtArcFace::load(paths.arcface_engine,
                                                                  options.arcface);
        auto inswapper = kfcore::face_models::TensorRtInSwapper::load(paths.inswapper_engine,
                                                                      options.inswapper);
        auto projector = kfcore::face_models::InSwapperEmbeddingProjector::load(
            paths.inswapper_matrix);
        std::unique_ptr<kfcore::face_models::TensorRtGfpGan> gfpgan;
        if (paths.gfpgan_engine)
        {
            gfpgan = kfcore::face_models::TensorRtGfpGan::load(*paths.gfpgan_engine,
                                                               options.gfpgan);
        }
        std::unique_ptr<kfcore::face_models::TensorRtAgeGender> age_gender;
        if (paths.age_gender_engine)
        {
            age_gender = kfcore::face_models::TensorRtAgeGender::load(
                *paths.age_gender_engine, options.age_gender);
        }

        auto impl = std::make_unique<Impl>(options, detector_engine, std::move(detector),
                                           std::move(face68), std::move(arcface),
                                           std::move(inswapper), std::move(projector),
                                           std::move(gfpgan), std::move(age_gender));
        return std::unique_ptr<TensorRtFaceSwapApplication>(
            new TensorRtFaceSwapApplication(std::move(impl)));
    }
    catch (const FaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        rethrow_model(error, "load");
    }
    catch (const kfcore::yolo::YoloError& error)
    {
        rethrow_yolo(error, "load");
    }
    catch (const std::bad_alloc&)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ResourceLimitExceeded,
                                   "face application load stage: allocation failed");
    }
}

FaceAnalysis TensorRtFaceSwapApplication::analyze(const cv::Mat& bgr_image)
{
    try
    {
        ApplicationCallGuard guard(impl_->in_use);
        return impl_->analyze_internal(bgr_image);
    }
    catch (const FaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        rethrow_model(error, "analysis");
    }
    catch (const kfcore::yolo::YoloError& error)
    {
        rethrow_yolo(error, "analysis");
    }
    catch (const cv::Exception& error)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ImageProcessingFailure,
                                   std::string("face application analysis stage: ") + error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ResourceLimitExceeded,
                                   "face application analysis stage: allocation failed");
    }
}

cv::Mat TensorRtFaceSwapApplication::swap(const cv::Mat& source_bgr,
                                          const cv::Mat& target_bgr)
{
    try
    {
        ApplicationCallGuard guard(impl_->in_use);
        const FaceAnalysis source = impl_->analyze_internal(source_bgr);
        const FaceAnalysis target = impl_->analyze_internal(target_bgr);
        const kfcore::face_models::ArcFaceResult projected =
            impl_->projector.project(source.embedding);

        const AlignedFace swap_aligned =
            align_face(target_bgr, target.landmarks, inswapper_template(),
                       static_cast<int>(kfcore::face_models::kInSwapperInputExtent));
        const std::vector<float> swap_values = preprocess_inswapper(swap_aligned.image);
        const auto swap_result = impl_->inswapper->infer(
            image_tensor(impl_->options.inswapper.target_input_name,
                         kfcore::face_models::kInSwapperInputExtent, swap_values),
            embedding_tensor(impl_->options.inswapper.source_input_name, projected));
        const cv::Mat swapped_face = decode_inswapper(swap_result);
        const cv::Mat swap_mask = create_static_box_mask(swapped_face.size());
        cv::Mat result = paste_back(target_bgr, swapped_face, swap_mask,
                                    swap_aligned.aligned_to_source);

        if (impl_->gfpgan)
        {
            const AlignedFace enhance_aligned =
                align_face(result, target.landmarks, gfpgan_template(),
                           static_cast<int>(kfcore::face_models::kGfpGanInputExtent));
            const std::vector<float> enhance_values = preprocess_gfpgan(enhance_aligned.image);
            const auto enhance_result = impl_->gfpgan->infer(
                image_tensor(impl_->options.gfpgan.input_name,
                             kfcore::face_models::kGfpGanInputExtent, enhance_values));
            const cv::Mat enhanced_face = decode_gfpgan(enhance_result);
            const cv::Mat enhance_mask = create_static_box_mask(enhanced_face.size());
            const cv::Mat pasted = paste_back(result, enhanced_face, enhance_mask,
                                              enhance_aligned.aligned_to_source);
            result = blend_images(result, pasted, impl_->options.enhancer_blend);
        }
        return result;
    }
    catch (const FaceApplicationError&)
    {
        throw;
    }
    catch (const kfcore::face_models::FaceModelError& error)
    {
        rethrow_model(error, "swap");
    }
    catch (const kfcore::yolo::YoloError& error)
    {
        rethrow_yolo(error, "swap");
    }
    catch (const cv::Exception& error)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ImageProcessingFailure,
                                   std::string("face application swap stage: ") + error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::ResourceLimitExceeded,
                                   "face application swap stage: allocation failed");
    }
}

} // namespace kfcore::face_applications
