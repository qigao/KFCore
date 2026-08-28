#include "kfcore/vision_models/cpu.hpp"

#include "decode.hpp"
#include "geometry.hpp"
#include "onnx_session.hpp"

#include "kfcore/image_processor/cpu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
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

constexpr std::size_t kImageChannels = 3U;
constexpr std::size_t kHandOutputWidth = kHandLandmarkCount * 3U;
constexpr std::size_t kClassifierFeatureWidth = kHandLandmarkCount * 2U;
constexpr std::int32_t kFaceDetectorInputExtent = 640;

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw VisionModelError(VisionModelErrorCode::InvalidArgument,
                           "CPU vision model stage: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw VisionModelError(VisionModelErrorCode::ModelContractMismatch,
                           "CPU vision model stage: " + detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw VisionModelError(VisionModelErrorCode::ResourceLimitExceeded,
                           "CPU vision model stage: " + detail);
}

double elapsed_ms(Clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

std::size_t checked_multiply(std::size_t left, std::size_t right, const char* subject)
{
    if (left != 0U && right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_resource(std::string(subject) + " size overflow");
    }
    return left * right;
}

void validate_options(const CpuVisionOptions& options)
{
    if (options.max_model_bytes == 0U || options.max_source_bytes == 0U ||
        options.max_tensor_bytes == 0U || options.max_output_bytes == 0U ||
        options.max_palm_candidates == 0U || options.max_hands == 0U)
    {
        throw_resource("all CPU resource limits must be positive");
    }
    if (options.max_hands > options.max_palm_candidates)
    {
        throw_invalid("max_hands must not exceed max_palm_candidates");
    }
    if (options.intra_op_threads < 0 || options.inter_op_threads < 0)
    {
        throw_invalid("ONNX Runtime thread counts must not be negative");
    }
    if (!std::isfinite(options.palm_score_threshold) ||
        options.palm_score_threshold < 0.0F || options.palm_score_threshold > 1.0F ||
        !std::isfinite(options.hand_score_threshold) ||
        options.hand_score_threshold < 0.0F || options.hand_score_threshold > 1.0F)
    {
        throw_invalid("model score thresholds must be finite within [0,1]");
    }
    if (!std::isfinite(options.face_detection_score_threshold) ||
        options.face_detection_score_threshold < 0.0F ||
        options.face_detection_score_threshold > 1.0F)
    {
        throw_invalid("face_detection_score_threshold must be finite within [0,1]");
    }
    if (options.face_class_id < 0)
    {
        throw_invalid("face_class_id must not be negative");
    }

    std::size_t hand_batch_elements = checked_multiply(
        options.max_hands, kImageChannels, "hand batch");
    hand_batch_elements = checked_multiply(
        hand_batch_elements, static_cast<std::size_t>(kHandLandmarkInputExtent), "hand batch");
    hand_batch_elements = checked_multiply(
        hand_batch_elements, static_cast<std::size_t>(kHandLandmarkInputExtent), "hand batch");
    const std::size_t hand_batch_bytes = checked_multiply(
        hand_batch_elements, sizeof(float), "hand batch");
    if (hand_batch_bytes > options.max_tensor_bytes)
    {
        throw_resource("maximum hand batch exceeds max_tensor_bytes");
    }
}

detail::OnnxSessionOptions session_options(const CpuVisionOptions& options)
{
    return { options.intra_op_threads, options.inter_op_threads,
             options.max_model_bytes, options.max_output_bytes };
}

detail::OnnxModelContract palm_contract()
{
    return { "Palm",
             { { "input", detail::OnnxElementType::Float32,
                 { 1, 3, kPalmInputExtent, kPalmInputExtent } } },
             { { "pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y",
                 detail::OnnxElementType::Float32, { -1, 8 } } } };
}

detail::OnnxModelContract hand_contract()
{
    return { "hand landmark",
             { { "input", detail::OnnxElementType::Float32,
                 { -1, 3, kHandLandmarkInputExtent, kHandLandmarkInputExtent } } },
             { { "xyz_x21", detail::OnnxElementType::Float32, { -1, 63 } },
               { "hand_score", detail::OnnxElementType::Float32, { -1, 1 } },
               { "lefthand_0_or_righthand_1", detail::OnnxElementType::Float32,
                 { -1, 1 } } } };
}

detail::OnnxModelContract classifier_contract()
{
    return { "keypoint classifier",
             { { "input", detail::OnnxElementType::Float32, { -1, 42 } } },
             { { "class_ids", detail::OnnxElementType::Int64, { -1 } } } };
}

detail::OnnxModelContract face_contract()
{
    return { "MediaPipe face landmark",
             { { "image", detail::OnnxElementType::Float32,
                 { 1, 3, kFaceLandmarkInputExtent, kFaceLandmarkInputExtent } } },
             { { "scores", detail::OnnxElementType::Float32, { 1 } },
               { "landmarks", detail::OnnxElementType::Float32,
                 { 1, static_cast<std::int64_t>(kFaceLandmarkCount), 3 } } } };
}

detail::OnnxModelContract face_detector_contract()
{
    return { "YOLOv12 face detector",
             { { "images", detail::OnnxElementType::Float32,
                 { 1, 3, kFaceDetectorInputExtent, kFaceDetectorInputExtent } } },
             { { "output0", detail::OnnxElementType::Float32, { 1, 300, 6 } } } };
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

class UseGuard final
{
public:
    explicit UseGuard(std::atomic_flag& flag)
        : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw VisionModelError(VisionModelErrorCode::ConcurrentExecution,
                                   "CPU vision model stage: instance is already in use");
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

struct CpuHandBackend::Impl final
{
    Impl(const HandOnnxModelPaths& paths, const CpuVisionOptions& options_value)
        : options(options_value)
        , environment(ORT_LOGGING_LEVEL_ERROR, "KFCoreVisionModelsCpuHand")
        , palm(environment, paths.palm, palm_contract(), session_options(options))
        , hand(environment, paths.hand_landmark, hand_contract(), session_options(options))
        , classifier(environment, paths.keypoint_classifier, classifier_contract(),
                     session_options(options))
    {
    }

    CpuVisionOptions    options;
    Ort::Env            environment;
    detail::OnnxSession palm;
    detail::OnnxSession hand;
    detail::OnnxSession classifier;
    std::atomic_flag    in_use = ATOMIC_FLAG_INIT;
};

CpuHandBackend::CpuHandBackend(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

CpuHandBackend::~CpuHandBackend() = default;

std::unique_ptr<CpuHandBackend> CpuHandBackend::load(
    const HandOnnxModelPaths& paths, const CpuVisionOptions& options)
{
    validate_options(options);
    try
    {
        auto impl = std::make_unique<Impl>(paths, options);
        return std::unique_ptr<CpuHandBackend>(new CpuHandBackend(std::move(impl)));
    }
    catch (const VisionModelError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("hand backend allocation failed");
    }
}

HandFrame CpuHandBackend::infer(const image::ImageView& source)
{
    if (!impl_)
    {
        throw_invalid("hand backend state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    validate_host_image(source);
    try
    {
        const Clock::time_point total_started = Clock::now();
        HandFrame result;

        const Clock::time_point palm_preprocess_started = Clock::now();
        image::LetterboxTransform letterbox;
        std::vector<float> palm_input = image::CpuImageProcessor::letterbox_nchw(
            source, kPalmInputExtent, kPalmInputExtent, rgb_unit_options(0.0F),
            impl_->options.max_source_bytes, impl_->options.max_tensor_bytes, &letterbox);
        result.timings.preprocess_ms += elapsed_ms(palm_preprocess_started);

        const Clock::time_point palm_inference_started = Clock::now();
        std::vector<detail::OnnxHostTensor> palm_outputs = impl_->palm.run(
            { { palm_input.data(), palm_input.size(),
                { 1, 3, kPalmInputExtent, kPalmInputExtent } } });
        result.timings.palm_inference_ms = elapsed_ms(palm_inference_started);
        if (palm_outputs.size() != 1U ||
            palm_outputs[0].element_type != detail::OnnxElementType::Float32)
        {
            throw_contract("Palm output set is invalid");
        }
        std::vector<PalmDetection> palms = detail::decode_palms(
            palm_outputs[0].float_values.data(), palm_outputs[0].float_values.size(),
            impl_->options.palm_score_threshold, impl_->options.max_palm_candidates,
            impl_->options.max_hands, letterbox, kPalmInputExtent);
        if (palms.empty())
        {
            result.timings.total_ms = elapsed_ms(total_started);
            return result;
        }

        const Clock::time_point hand_preprocess_started = Clock::now();
        const image::BgrImage packed = image::CpuImageProcessor::copy_bgr(
            source, impl_->options.max_source_bytes);
        const std::size_t one_hand_elements =
            kImageChannels * static_cast<std::size_t>(kHandLandmarkInputExtent) *
            static_cast<std::size_t>(kHandLandmarkInputExtent);
        std::vector<float> hand_batch;
        hand_batch.reserve(checked_multiply(palms.size(), one_hand_elements, "hand batch"));
        for (const PalmDetection& palm : palms)
        {
            const image::AffineTransform transform = detail::hand_roi_transform(
                palm.roi, kHandLandmarkInputExtent);
            const image::BgrImage crop = image::CpuImageProcessor::warp_affine_bgr(
                packed, kHandLandmarkInputExtent, kHandLandmarkInputExtent, transform,
                0.0F, impl_->options.max_source_bytes);
            std::vector<float> tensor = image::CpuImageProcessor::to_nchw(
                crop, rgb_unit_options(0.0F), impl_->options.max_tensor_bytes);
            hand_batch.insert(hand_batch.end(), tensor.begin(), tensor.end());
        }
        result.timings.preprocess_ms += elapsed_ms(hand_preprocess_started);

        const Clock::time_point hand_inference_started = Clock::now();
        const std::int64_t hand_count = static_cast<std::int64_t>(palms.size());
        std::vector<detail::OnnxHostTensor> hand_outputs = impl_->hand.run(
            { { hand_batch.data(), hand_batch.size(),
                { hand_count, 3, kHandLandmarkInputExtent, kHandLandmarkInputExtent } } });
        result.timings.landmark_inference_ms = elapsed_ms(hand_inference_started);
        if (hand_outputs.size() != 3U)
        {
            throw_contract("hand landmark output set is invalid");
        }
        if (hand_outputs[0].float_values.size() != palms.size() * kHandOutputWidth ||
            hand_outputs[1].float_values.size() != palms.size() ||
            hand_outputs[2].float_values.size() != palms.size())
        {
            throw_contract("hand landmark runtime batch does not match Palm candidates");
        }

        std::vector<std::array<float, kClassifierFeatureWidth>> classifier_features;
        classifier_features.reserve(palms.size());
        result.hands.reserve(palms.size());
        const Clock::time_point classifier_preprocess_started = Clock::now();
        for (std::size_t index = 0; index < palms.size(); ++index)
        {
            const float score = hand_outputs[1].float_values[index];
            if (!std::isfinite(score))
            {
                throw_contract("hand landmark score is non-finite");
            }
            if (score < impl_->options.hand_score_threshold)
            {
                continue;
            }
            HandResult hand_result;
            hand_result.palm = palms[index];
            hand_result.landmark_confidence = score;
            hand_result.handedness = detail::decode_handedness(
                hand_outputs[2].float_values[index]);
            const float* xyz = hand_outputs[0].float_values.data() + index * kHandOutputWidth;
            hand_result.landmarks = detail::decode_hand_landmarks(
                xyz, kHandOutputWidth, palms[index].roi, kHandLandmarkInputExtent);
            classifier_features.push_back(detail::make_keypoint_features(
                hand_result.landmarks));
            result.hands.push_back(std::move(hand_result));
        }
        result.timings.preprocess_ms += elapsed_ms(classifier_preprocess_started);
        if (!result.hands.empty())
        {
            std::vector<float> classifier_input;
            classifier_input.reserve(checked_multiply(result.hands.size(),
                                                       kClassifierFeatureWidth,
                                                       "classifier batch"));
            for (const auto& feature : classifier_features)
            {
                classifier_input.insert(classifier_input.end(), feature.begin(), feature.end());
            }
            const Clock::time_point classifier_inference_started = Clock::now();
            std::vector<detail::OnnxHostTensor> classifier_outputs = impl_->classifier.run(
                { { classifier_input.data(), classifier_input.size(),
                    { static_cast<std::int64_t>(result.hands.size()),
                      static_cast<std::int64_t>(kClassifierFeatureWidth) } } });
            result.timings.classifier_inference_ms = elapsed_ms(classifier_inference_started);
            if (classifier_outputs.size() != 1U ||
                classifier_outputs[0].int64_values.size() != result.hands.size())
            {
                throw_contract("keypoint classifier output set is invalid");
            }
            for (std::size_t index = 0; index < result.hands.size(); ++index)
            {
                result.hands[index].gesture = detail::decode_gesture(
                    classifier_outputs[0].int64_values[index]);
            }
        }
        result.timings.total_ms = elapsed_ms(total_started);
        return result;
    }
    catch (const VisionModelError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("hand inference allocation failed");
    }
}

struct CpuFaceDetector::Impl final
{
    Impl(const std::filesystem::path& path, const CpuVisionOptions& options_value)
        : options(options_value)
        , environment(ORT_LOGGING_LEVEL_ERROR, "KFCoreVisionModelsCpuFaceDetector")
        , session(environment, path, face_detector_contract(), session_options(options))
    {
    }

    CpuVisionOptions    options;
    Ort::Env            environment;
    detail::OnnxSession session;
    std::atomic_flag    in_use = ATOMIC_FLAG_INIT;
};

CpuFaceDetector::CpuFaceDetector(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

CpuFaceDetector::~CpuFaceDetector() = default;

std::unique_ptr<CpuFaceDetector> CpuFaceDetector::load(
    const std::filesystem::path& model_path, const CpuVisionOptions& options)
{
    validate_options(options);
    try
    {
        auto impl = std::make_unique<Impl>(model_path, options);
        return std::unique_ptr<CpuFaceDetector>(new CpuFaceDetector(std::move(impl)));
    }
    catch (const VisionModelError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("face detector allocation failed");
    }
}

FaceDetectionResult CpuFaceDetector::infer(const image::ImageView& source)
{
    if (!impl_)
    {
        throw_invalid("face detector state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    validate_host_image(source);
    try
    {
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
        std::vector<detail::OnnxHostTensor> outputs = impl_->session.run(
            { { input.data(), input.size(),
                { 1, 3, kFaceDetectorInputExtent, kFaceDetectorInputExtent } } });
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
    catch (const VisionModelError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("face detector inference allocation failed");
    }
}

struct CpuFaceLandmarker::Impl final
{
    Impl(const std::filesystem::path& path, const CpuVisionOptions& options_value)
        : options(options_value)
        , environment(ORT_LOGGING_LEVEL_ERROR, "KFCoreVisionModelsCpuFace")
        , session(environment, path, face_contract(), session_options(options))
    {
    }

    CpuVisionOptions    options;
    Ort::Env            environment;
    detail::OnnxSession session;
    std::atomic_flag    in_use = ATOMIC_FLAG_INIT;
};

CpuFaceLandmarker::CpuFaceLandmarker(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

CpuFaceLandmarker::~CpuFaceLandmarker() = default;

std::unique_ptr<CpuFaceLandmarker> CpuFaceLandmarker::load(
    const std::filesystem::path& model_path, const CpuVisionOptions& options)
{
    validate_options(options);
    try
    {
        auto impl = std::make_unique<Impl>(model_path, options);
        return std::unique_ptr<CpuFaceLandmarker>(new CpuFaceLandmarker(std::move(impl)));
    }
    catch (const VisionModelError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("face landmarker allocation failed");
    }
}

FaceLandmarkResult CpuFaceLandmarker::infer(const image::ImageView& source,
                                            const RectF& face_box)
{
    if (!impl_)
    {
        throw_invalid("face landmarker state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    validate_host_image(source);
    if (!std::isfinite(face_box.x) || !std::isfinite(face_box.y) ||
        !std::isfinite(face_box.width) || !std::isfinite(face_box.height) ||
        face_box.width <= 0.0F || face_box.height <= 0.0F)
    {
        throw_invalid("face box must be finite and positive");
    }
    try
    {
        const Clock::time_point total_started = Clock::now();
        FaceLandmarkResult result;
        const Clock::time_point preprocess_started = Clock::now();
        const image::BgrImage packed = image::CpuImageProcessor::copy_bgr(
            source, impl_->options.max_source_bytes);
        const detail::FaceRoi roi = detail::make_face_roi(
            face_box, kFaceLandmarkInputExtent);
        const image::BgrImage crop = image::CpuImageProcessor::warp_affine_bgr(
            packed, kFaceLandmarkInputExtent, kFaceLandmarkInputExtent,
            roi.destination_to_source, 0.0F, impl_->options.max_source_bytes);
        std::vector<float> input = image::CpuImageProcessor::to_nchw(
            crop, rgb_unit_options(0.0F), impl_->options.max_tensor_bytes);
        result.preprocess_ms = elapsed_ms(preprocess_started);

        const Clock::time_point inference_started = Clock::now();
        std::vector<detail::OnnxHostTensor> outputs = impl_->session.run(
            { { input.data(), input.size(),
                { 1, 3, kFaceLandmarkInputExtent, kFaceLandmarkInputExtent } } });
        result.inference_ms = elapsed_ms(inference_started);
        if (outputs.size() != 2U || outputs[0].float_values.size() != 1U ||
            outputs[1].float_values.size() != kFaceLandmarkCount * 3U)
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
            kFaceLandmarkInputExtent, impl_->options.face_coordinates_normalized);
        result.total_ms = elapsed_ms(total_started);
        return result;
    }
    catch (const VisionModelError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("face landmark inference allocation failed");
    }
}

} // namespace kfcore::vision_models
