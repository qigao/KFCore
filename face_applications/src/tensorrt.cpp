#include "kfcore/face_applications/tensorrt.hpp"

#include "kfcore/face_applications/composer.hpp"
#include "kfcore/face_applications/error.hpp"
#include "kfcore/face_applications/preprocess.hpp"
#include "kfcore/image_processor/image_processor.hpp"
#include "kfcore/yolo/error.hpp"
#include "kfcore/yolo/opencv.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::face_applications
{
namespace
{

class StageTimer final
{
public:
    explicit StageTimer(FaceSwapDuration* destination) noexcept
        : destination_(destination), started_(destination ? Clock::now() : Clock::time_point {})
    {
    }

    ~StageTimer()
    {
        if (destination_ != nullptr)
        {
            *destination_ =
                std::chrono::duration_cast<FaceSwapDuration>(Clock::now() - started_);
        }
    }

    StageTimer(const StageTimer&) = delete;
    StageTimer& operator=(const StageTimer&) = delete;

private:
    using Clock = std::chrono::steady_clock;
    FaceSwapDuration* destination_;
    Clock::time_point started_;
};

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

void validate_final_pipeline_device(const FaceApplicationModelPaths& paths,
                                    const FaceSwapOptions& options)
{
    if (paths.gfpgan_engine &&
        options.inswapper.engine.device_id != options.gfpgan.engine.device_id)
    {
        throw_invalid("InSwapper and GFPGAN must use the same CUDA device");
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

FaceApplicationErrorCode map_image_error(kfcore::image::ImageProcessorErrorCode code)
{
    using kfcore::image::ImageProcessorErrorCode;
    switch (code)
    {
    case ImageProcessorErrorCode::InvalidArgument:
        return FaceApplicationErrorCode::InvalidArgument;
    case ImageProcessorErrorCode::ResourceLimitExceeded:
        return FaceApplicationErrorCode::ResourceLimitExceeded;
    case ImageProcessorErrorCode::CudaFailure:
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

[[noreturn]] void rethrow_image(const kfcore::image::ImageProcessorError& error,
                                const char* stage)
{
    throw FaceApplicationError(map_image_error(error.code()),
                               std::string("face application ") + stage + " stage: " +
                                   error.what());
}

kfcore::tensorrt::TensorView device_image_tensor(
    const std::string& name, std::int64_t extent,
    const kfcore::image::TensorView& values)
{
    return { name,
             kfcore::tensorrt::DataType::Float32,
             { 1, kfcore::face_models::kFaceModelInputChannels, extent, extent },
             values.data,
             values.byte_size,
             kfcore::tensorrt::MemoryKind::CudaDevice };
}

kfcore::tensorrt::MutableTensorView mutable_device_image_tensor(
    const std::string& name, std::int64_t extent, const kfcore::image::TensorView& values)
{
    return { name,
             kfcore::tensorrt::DataType::Float32,
             { 1, kfcore::face_models::kFaceModelInputChannels, extent, extent },
             values.data,
             values.byte_size,
             kfcore::tensorrt::MemoryKind::CudaDevice };
}

kfcore::image::TensorView host_alpha_tensor(const cv::Mat& alpha)
{
    if (alpha.empty() || alpha.type() != CV_32FC1 || !alpha.isContinuous())
    {
        throw FaceApplicationError(
            FaceApplicationErrorCode::ImageProcessingFailure,
            "face application mask stage: alpha mask must be continuous CV_32FC1");
    }
    return { alpha.data,
             alpha.total() * alpha.elemSize(),
             1,
             1,
             alpha.rows,
             alpha.cols,
             kfcore::image::TensorElementType::Float32,
             kfcore::image::TensorLayout::Nchw,
             kfcore::image::MemoryKind::Host };
}

kfcore::tensorrt::TensorView host_image_tensor(const std::string& name, std::int64_t extent,
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

cv::Rect face_roi(const cv::Mat& image, const kfcore::yolo::Detection& detection)
{
    const int left = (std::clamp)(static_cast<int>(std::floor(detection.box.left)), 0,
                                  image.cols - 1);
    const int top = (std::clamp)(static_cast<int>(std::floor(detection.box.top)), 0,
                                 image.rows - 1);
    const int right = (std::clamp)(static_cast<int>(std::ceil(detection.box.right)), left + 1,
                                   image.cols);
    const int bottom = (std::clamp)(static_cast<int>(std::ceil(detection.box.bottom)), top + 1,
                                    image.rows);
    return { left, top, right - left, bottom - top };
}

kfcore::image::ImageView borrowed_bgr(const cv::Mat& image)
{
    const kfcore::yolo::ImageView validated = kfcore::yolo::image_view(image);
    const std::size_t row_bytes = static_cast<std::size_t>(validated.width) * 3U;
    const std::size_t span =
        static_cast<std::size_t>(validated.height - 1) * validated.row_stride + row_bytes;
    return { validated.data, span, validated.width, validated.height, validated.row_stride,
             kfcore::image::PixelFormat::Bgr8, kfcore::image::MemoryKind::Host };
}

kfcore::image::AffineTransform affine_transform(const cv::Matx23f& destination_to_source)
{
    return { { destination_to_source(0, 0), destination_to_source(0, 1),
               destination_to_source(0, 2), destination_to_source(1, 0),
               destination_to_source(1, 1), destination_to_source(1, 2) } };
}

kfcore::yolo::ImageView yolo_image_view(const kfcore::image::ImageView& image)
{
    return { image.data, image.width, image.height, image.row_stride,
             image.pixel_format == kfcore::image::PixelFormat::Rgb8
                 ? kfcore::yolo::PixelFormat::Rgb8
                 : kfcore::yolo::PixelFormat::Bgr8,
             kfcore::yolo::MemoryKind::CudaDevice };
}

kfcore::image::PreprocessOptions bgr_unit_options()
{
    kfcore::image::PreprocessOptions result;
    result.output_format = kfcore::image::PixelFormat::Bgr8;
    result.border_value = 0.0F;
    return result;
}

kfcore::image::PreprocessOptions rgb_unit_options()
{
    kfcore::image::PreprocessOptions result;
    result.output_format = kfcore::image::PixelFormat::Rgb8;
    result.border_value = 0.0F;
    return result;
}

kfcore::image::PreprocessOptions rgb_signed_options()
{
    kfcore::image::PreprocessOptions result = rgb_unit_options();
    result.mean = { 0.5F, 0.5F, 0.5F };
    result.stddev = { 0.5F, 0.5F, 0.5F };
    return result;
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
    struct DeviceProcessor
    {
        int device_id = 0;
        std::unique_ptr<kfcore::image::CudaImageProcessor> processor;
        std::uint64_t staged_generation = 0;
        kfcore::image::ImageView staged_image;
    };

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
        inswapper_mask = create_static_box_mask(
            { static_cast<int>(kfcore::face_models::kInSwapperInputExtent),
              static_cast<int>(kfcore::face_models::kInSwapperInputExtent) });
        if (gfpgan)
        {
            gfpgan_mask = create_static_box_mask(
                { static_cast<int>(kfcore::face_models::kGfpGanInputExtent),
                  static_cast<int>(kfcore::face_models::kGfpGanInputExtent) });
        }
        add_processor(options.detector_engine.device_id);
        add_processor(options.face68.engine.device_id);
        add_processor(options.arcface.engine.device_id);
        add_processor(options.inswapper.engine.device_id);
        if (gfpgan)
        {
            add_processor(options.gfpgan.engine.device_id);
        }
    }

    void add_processor(int device_id)
    {
        const auto found = std::find_if(
            processors.begin(), processors.end(),
            [device_id](const DeviceProcessor& entry) { return entry.device_id == device_id; });
        if (found != processors.end())
        {
            return;
        }
        kfcore::image::CudaImageProcessorOptions processor_options;
        processor_options.device_id = device_id;
        processor_options.max_source_bytes = options.detector_engine.max_input_bytes;
        processor_options.max_tensor_bytes = (std::max)(
            { options.face68.engine.max_input_bytes, options.arcface.engine.max_input_bytes,
              options.inswapper.engine.max_input_bytes, options.gfpgan.engine.max_input_bytes,
              options.age_gender.engine.max_input_bytes, options.inswapper.engine.max_output_bytes,
              options.gfpgan.engine.max_output_bytes });
        processors.push_back(
            { device_id, kfcore::image::CudaImageProcessor::create(processor_options), 0, {} });
    }

    kfcore::image::CudaImageProcessor& processor_for(int device_id)
    {
        const auto found = std::find_if(
            processors.begin(), processors.end(),
            [device_id](const DeviceProcessor& entry) { return entry.device_id == device_id; });
        if (found == processors.end())
        {
            throw FaceApplicationError(
                FaceApplicationErrorCode::RuntimeFailure,
                "face application preprocessing stage: CUDA processor device is unavailable");
        }
        return *found->processor;
    }

    void begin_frame()
    {
        if (frame_generation == (std::numeric_limits<std::uint64_t>::max)())
        {
            for (DeviceProcessor& entry : processors)
            {
                entry.staged_generation = 0;
            }
            frame_generation = 1;
            return;
        }
        ++frame_generation;
    }

    const kfcore::image::ImageView& stage_for(const cv::Mat& image, int device_id)
    {
        const auto found = std::find_if(
            processors.begin(), processors.end(),
            [device_id](const DeviceProcessor& entry) { return entry.device_id == device_id; });
        if (found == processors.end())
        {
            throw FaceApplicationError(
                FaceApplicationErrorCode::RuntimeFailure,
                "face application staging stage: CUDA processor device is unavailable");
        }
        if (found->staged_generation != frame_generation)
        {
            found->staged_image = found->processor->stage(borrowed_bgr(image));
            found->staged_generation = frame_generation;
        }
        return found->staged_image;
    }

    kfcore::image::TensorView preprocess(
        const kfcore::image::ImageView& image, std::int32_t extent,
        const kfcore::image::AffineTransform& transform,
        const kfcore::image::PreprocessOptions& preprocess_options, int device_id)
    {
        return processor_for(device_id).process_affine(
            image, extent, extent, transform, preprocess_options,
            kfcore::image::TensorElementType::Float32);
    }

    FaceAnalysis analyze_internal(const cv::Mat& bgr_image,
                                  FaceAnalysisTimingReport* timings = nullptr)
    {
        StageTimer total_timer(timings != nullptr ? &timings->total : nullptr);
        const kfcore::image::ImageView* detector_image = nullptr;
        {
            StageTimer timer(timings != nullptr ? &timings->initial_staging : nullptr);
            begin_frame();
            detector_image = &stage_for(bgr_image, options.detector_engine.device_id);
        }
        std::optional<kfcore::yolo::Detection> detection;
        {
            StageTimer timer(timings != nullptr ? &timings->detection : nullptr);
            const kfcore::yolo::DetectionFrame frame =
                detector->detect(yolo_image_view(*detector_image));
            detection = select_highest_score_face(frame.detections, options.face_class_id,
                                                  options.detector_score_threshold);
        }
        if (!detection)
        {
            throw FaceApplicationError(FaceApplicationErrorCode::NoFaceDetected,
                                       "face application detect stage: 12face found no qualifying face");
        }

        FaceTransform face68_aligned;
        kfcore::image::TensorView face68_values;
        {
            StageTimer timer(timings != nullptr ? &timings->face68_preprocess : nullptr);
            face68_aligned = face68_transform(face_box(*detection));
            const kfcore::image::ImageView& face68_image =
                stage_for(bgr_image, options.face68.engine.device_id);
            face68_values = preprocess(
                face68_image,
                static_cast<std::int32_t>(kfcore::face_models::kFace68InputExtent),
                affine_transform(face68_aligned.aligned_to_source), bgr_unit_options(),
                options.face68.engine.device_id);
        }
        kfcore::face_models::Face68Result landmarks68;
        FiveLandmarks landmarks;
        {
            StageTimer timer(
                timings != nullptr ? &timings->face68_inference_and_postprocess : nullptr);
            const std::vector<kfcore::face_models::Face68Result> face68_results =
                face68->infer(device_image_tensor(options.face68.input_name,
                                                  kfcore::face_models::kFace68InputExtent,
                                                  face68_values));
            if (face68_results.size() != 1U)
            {
                throw FaceApplicationError(
                    FaceApplicationErrorCode::RuntimeFailure,
                    "face application landmark stage: Face68 returned a non-unit batch");
            }
            landmarks68 = map_face68_to_source(face68_results.front(),
                                               face68_aligned.aligned_to_source);
            landmarks = extract_five_landmarks(landmarks68);
        }

        kfcore::image::TensorView arcface_values;
        {
            StageTimer timer(timings != nullptr ? &timings->arcface_preprocess : nullptr);
            const FaceTransform arcface_aligned =
                alignment_transform(landmarks, arcface_template());
            const kfcore::image::ImageView& arcface_image =
                stage_for(bgr_image, options.arcface.engine.device_id);
            arcface_values = preprocess(
                arcface_image,
                static_cast<std::int32_t>(kfcore::face_models::kArcFaceInputExtent),
                affine_transform(arcface_aligned.aligned_to_source), rgb_signed_options(),
                options.arcface.engine.device_id);
        }
        kfcore::face_models::ArcFaceResult embedding;
        {
            StageTimer timer(timings != nullptr ? &timings->arcface_inference : nullptr);
            const std::vector<kfcore::face_models::ArcFaceResult> embeddings =
                arcface->infer(device_image_tensor(options.arcface.input_name,
                                                   kfcore::face_models::kArcFaceInputExtent,
                                                   arcface_values));
            if (embeddings.size() != 1U)
            {
                throw FaceApplicationError(
                    FaceApplicationErrorCode::RuntimeFailure,
                    "face application embedding stage: ArcFace returned a non-unit batch");
            }
            embedding = embeddings.front();
        }

        FaceAnalysis result { *detection, landmarks68, landmarks, embedding, std::nullopt };
        if (age_gender)
        {
            if (timings != nullptr)
            {
                timings->age_gender.emplace();
            }
            StageTimer timer(timings != nullptr ? &*timings->age_gender : nullptr);
            const cv::Rect roi = face_roi(bgr_image, *detection);
            const std::vector<float> age_values = preprocess_age_gender(bgr_image(roi));
            const auto logits = age_gender->infer(
                host_image_tensor(options.age_gender.input_name,
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
    cv::Mat inswapper_mask;
    cv::Mat gfpgan_mask;
    std::vector<DeviceProcessor> processors;
    std::uint64_t frame_generation = 0;
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
    validate_final_pipeline_device(paths, options);
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
    catch (const kfcore::image::ImageProcessorError& error)
    {
        rethrow_image(error, "load");
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
    catch (const kfcore::image::ImageProcessorError& error)
    {
        rethrow_image(error, "analysis");
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
    return swap_internal(source_bgr, target_bgr, nullptr);
}

ProfiledFaceSwapResult TensorRtFaceSwapApplication::swap_profiled(
    const cv::Mat& source_bgr, const cv::Mat& target_bgr)
{
    FaceSwapTimingReport timings;
    cv::Mat image = swap_internal(source_bgr, target_bgr, &timings);
    return { std::move(image), std::move(timings) };
}

cv::Mat TensorRtFaceSwapApplication::swap_internal(const cv::Mat& source_bgr,
                                                   const cv::Mat& target_bgr,
                                                   FaceSwapTimingReport* timings)
{
    try
    {
        StageTimer total_timer(timings != nullptr ? &timings->total : nullptr);
        ApplicationCallGuard guard(impl_->in_use);
        const FaceAnalysis source = impl_->analyze_internal(
            source_bgr, timings != nullptr ? &timings->source_analysis : nullptr);
        const FaceAnalysis target = impl_->analyze_internal(
            target_bgr, timings != nullptr ? &timings->target_analysis : nullptr);
        kfcore::face_models::ArcFaceResult projected;
        {
            StageTimer timer(timings != nullptr ? &timings->embedding_projection : nullptr);
            projected = impl_->projector.project(source.embedding);
        }

        FaceTransform swap_aligned;
        kfcore::image::TensorView swap_values;
        const kfcore::image::ImageView* target_device_image = nullptr;
        kfcore::image::CudaImageProcessor& final_processor =
            impl_->processor_for(impl_->options.inswapper.engine.device_id);
        {
            StageTimer timer(timings != nullptr ? &timings->inswapper_preprocess : nullptr);
            swap_aligned = alignment_transform(target.landmarks, inswapper_template());
            target_device_image = &impl_->stage_for(
                target_bgr, impl_->options.inswapper.engine.device_id);
            swap_values = impl_->preprocess(
                *target_device_image,
                static_cast<std::int32_t>(kfcore::face_models::kInSwapperInputExtent),
                affine_transform(swap_aligned.aligned_to_source), rgb_unit_options(),
                impl_->options.inswapper.engine.device_id);
        }
        kfcore::image::TensorView swap_output;
        {
            StageTimer timer(
                timings != nullptr ? &timings->inswapper_inference_and_decode : nullptr);
            swap_output = final_processor.acquire_tensor(
                1, kfcore::face_models::kFaceModelInputChannels,
                static_cast<std::int32_t>(kfcore::face_models::kInSwapperInputExtent),
                static_cast<std::int32_t>(kfcore::face_models::kInSwapperInputExtent));
            impl_->inswapper->infer_into(
                device_image_tensor(impl_->options.inswapper.target_input_name,
                                    kfcore::face_models::kInSwapperInputExtent, swap_values),
                embedding_tensor(impl_->options.inswapper.source_input_name, projected),
                mutable_device_image_tensor(impl_->options.inswapper.output_name,
                                            kfcore::face_models::kInSwapperInputExtent,
                                            swap_output));
        }
        cv::Mat result;
        kfcore::image::ImageView final_device_image;
        {
            StageTimer timer(timings != nullptr ? &timings->inswapper_composition : nullptr);
            final_device_image = final_processor.composite_affine(
                *target_device_image, swap_output, host_alpha_tensor(impl_->inswapper_mask),
                affine_transform(swap_aligned.source_to_aligned));
            if (!impl_->gfpgan)
            {
                result.create(target_bgr.rows, target_bgr.cols, CV_8UC3);
                final_processor.download_bgr(
                    final_device_image, { result.data, result.total() * result.elemSize() });
            }
        }

        if (impl_->gfpgan)
        {
            if (timings != nullptr)
            {
                timings->gfpgan_preprocess.emplace();
                timings->gfpgan_inference_and_decode.emplace();
                timings->gfpgan_composition.emplace();
            }
            FaceTransform enhance_aligned;
            kfcore::image::TensorView enhance_values;
            {
                StageTimer timer(timings != nullptr ? &*timings->gfpgan_preprocess : nullptr);
                enhance_aligned = alignment_transform(target.landmarks, gfpgan_template());
                enhance_values = impl_->preprocess(
                    final_device_image,
                    static_cast<std::int32_t>(kfcore::face_models::kGfpGanInputExtent),
                    affine_transform(enhance_aligned.aligned_to_source), rgb_signed_options(),
                    impl_->options.gfpgan.engine.device_id);
            }
            kfcore::image::TensorView enhance_output;
            {
                StageTimer timer(
                    timings != nullptr ? &*timings->gfpgan_inference_and_decode : nullptr);
                enhance_output = final_processor.acquire_tensor(
                    1, kfcore::face_models::kFaceModelInputChannels,
                    static_cast<std::int32_t>(kfcore::face_models::kGfpGanInputExtent),
                    static_cast<std::int32_t>(kfcore::face_models::kGfpGanInputExtent));
                impl_->gfpgan->infer_into(
                    device_image_tensor(impl_->options.gfpgan.input_name,
                                        kfcore::face_models::kGfpGanInputExtent,
                                        enhance_values),
                    mutable_device_image_tensor(impl_->options.gfpgan.output_name,
                                                kfcore::face_models::kGfpGanInputExtent,
                                                enhance_output));
            }
            {
                StageTimer timer(timings != nullptr ? &*timings->gfpgan_composition : nullptr);
                kfcore::image::TensorCompositeOptions composite_options;
                composite_options.input_range = kfcore::image::TensorValueRange::SignedUnit;
                composite_options.strength = impl_->options.enhancer_blend;
                final_device_image = final_processor.composite_affine(
                    final_device_image, enhance_output, host_alpha_tensor(impl_->gfpgan_mask),
                    affine_transform(enhance_aligned.source_to_aligned), composite_options);
                result.create(target_bgr.rows, target_bgr.cols, CV_8UC3);
                final_processor.download_bgr(
                    final_device_image, { result.data, result.total() * result.elemSize() });
            }
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
    catch (const kfcore::image::ImageProcessorError& error)
    {
        rethrow_image(error, "swap");
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
