#include "kfcore/hand_models/tensorrt.hpp"

#include "decode.hpp"
#include "geometry.hpp"
#include "tensorrt_image_source.hpp"
#include "tensorrt_models.hpp"

#include "kfcore/image_processor/error.hpp"
#include "kfcore/image_processor/image_processor.hpp"
#include "kfcore/tensorrt/error.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace kfcore::hand_models
{
namespace
{

using Clock = std::chrono::steady_clock;

constexpr std::size_t kRgbChannels = 3U;
constexpr std::size_t kClassifierFeatureWidth = kHandLandmarkCount * 2U;

[[noreturn]] void throw_vision(HandModelErrorCode code, const std::string& detail)
{
    throw HandModelError(code, "TensorRT hand model stage: " + detail);
}

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw_vision(HandModelErrorCode::InvalidArgument, detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw_vision(HandModelErrorCode::ResourceLimitExceeded, detail);
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

void validate_options(const TensorRtHandOptions& options)
{
    if (options.max_engine_bytes == 0U || options.max_source_bytes == 0U ||
        options.max_tensor_bytes == 0U || options.max_output_bytes == 0U ||
        options.max_palm_candidates == 0U || options.max_hands == 0U)
    {
        throw_resource("all CUDA resource limits must be positive");
    }
    if (options.device_id < 0)
    {
        throw_invalid("CUDA device id must not be negative");
    }
    if (options.max_hands > options.max_palm_candidates)
    {
        throw_invalid("max_hands must not exceed max_palm_candidates");
    }
    if (!std::isfinite(options.palm_score_threshold) ||
        options.palm_score_threshold < 0.0F || options.palm_score_threshold > 1.0F ||
        !std::isfinite(options.hand_score_threshold) ||
        options.hand_score_threshold < 0.0F || options.hand_score_threshold > 1.0F)
    {
        throw_invalid("model score thresholds must be finite within [0,1]");
    }
    std::size_t largest_tensor = checked_multiply(
        kRgbChannels, static_cast<std::size_t>(kHandLandmarkInputExtent), "hand tensor");
    largest_tensor = checked_multiply(
        largest_tensor, static_cast<std::size_t>(kHandLandmarkInputExtent), "hand tensor");
    largest_tensor = checked_multiply(largest_tensor, sizeof(float), "hand tensor");
    if (largest_tensor > options.max_tensor_bytes)
    {
        throw_resource("hand input exceeds max_tensor_bytes");
    }
    std::size_t palm_output = checked_multiply(
        options.max_palm_candidates, kPalmRowWidth, "Palm output");
    palm_output = checked_multiply(palm_output, sizeof(float), "Palm output");
    if (palm_output > options.max_output_bytes)
    {
        throw_resource("maximum Palm output exceeds max_output_bytes");
    }
}

void validate_model_asset(const std::filesystem::path& path, const char* model,
                          std::size_t max_engine_bytes)
{
    if (path.empty())
    {
        throw_vision(HandModelErrorCode::InvalidModelAsset,
                     std::string(model) + " engine path must not be empty");
    }
    std::error_code error;
    const bool regular = std::filesystem::is_regular_file(path, error);
    if (error || !regular)
    {
        throw_vision(HandModelErrorCode::InvalidModelAsset,
                     std::string(model) + " engine must be an existing regular file: " +
                         path.string());
    }
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (error || size == 0U)
    {
        throw_vision(HandModelErrorCode::InvalidModelAsset,
                     std::string(model) + " engine cannot be read or is empty: " +
                         path.string());
    }
    if (size > max_engine_bytes)
    {
        throw_resource(std::string(model) + " engine exceeds max_engine_bytes");
    }
}

detail::TensorRtModelOptions model_options(const TensorRtHandOptions& options)
{
    return { options.device_id, options.max_engine_bytes, options.max_tensor_bytes,
             options.max_output_bytes, options.max_palm_candidates, options.max_hands };
}

image::CudaImageProcessorOptions processor_options(const TensorRtHandOptions& options)
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

image::AffineTransform letterbox_affine(const image::LetterboxTransform& letterbox)
{
    const float inverse_scale = 1.0F / letterbox.scale;
    return { { inverse_scale, 0.0F,
               (0.5F - letterbox.pad_x) * inverse_scale - 0.5F,
               0.0F, inverse_scale,
               (0.5F - letterbox.pad_y) * inverse_scale - 0.5F } };
}

HandModelErrorCode map_tensorrt_error(tensorrt::TensorRtErrorCode code)
{
    using tensorrt::TensorRtErrorCode;
    switch (code)
    {
    case TensorRtErrorCode::InvalidArgument:
    case TensorRtErrorCode::InvalidTensorView:
        return HandModelErrorCode::InvalidArgument;
    case TensorRtErrorCode::FileIo:
    case TensorRtErrorCode::EngineDeserialize:
        return HandModelErrorCode::InvalidModelAsset;
    case TensorRtErrorCode::EngineContractMismatch:
        return HandModelErrorCode::ModelContractMismatch;
    case TensorRtErrorCode::ResourceLimitExceeded:
        return HandModelErrorCode::ResourceLimitExceeded;
    case TensorRtErrorCode::ConcurrentExecution:
        return HandModelErrorCode::ConcurrentExecution;
    case TensorRtErrorCode::TensorRtFailure:
    case TensorRtErrorCode::CudaFailure:
        return HandModelErrorCode::RuntimeFailure;
    }
    return HandModelErrorCode::RuntimeFailure;
}

[[noreturn]] void rethrow_tensorrt(const tensorrt::TensorRtError& error)
{
    throw_vision(map_tensorrt_error(error.code()), error.what());
}

[[noreturn]] void rethrow_image(const image::ImageProcessorError& error)
{
    HandModelErrorCode code = HandModelErrorCode::RuntimeFailure;
    if (error.code() == image::ImageProcessorErrorCode::InvalidArgument)
    {
        code = HandModelErrorCode::InvalidArgument;
    }
    else if (error.code() == image::ImageProcessorErrorCode::ResourceLimitExceeded)
    {
        code = HandModelErrorCode::ResourceLimitExceeded;
    }
    throw_vision(code, error.what());
}

class UseGuard final
{
public:
    explicit UseGuard(std::atomic_flag& flag)
        : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw_vision(HandModelErrorCode::ConcurrentExecution,
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

} // namespace

struct TensorRtHandInput::Impl final
{
    explicit Impl(const TensorRtHandOptions& options)
        : processor(image::CudaImageProcessor::create(processor_options(options)))
    {
    }

    std::unique_ptr<image::CudaImageProcessor> processor;
    std::atomic_flag                           in_use = ATOMIC_FLAG_INIT;
};

TensorRtHandInput::TensorRtHandInput(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

TensorRtHandInput::~TensorRtHandInput() = default;

std::unique_ptr<TensorRtHandInput> TensorRtHandInput::create(
    const TensorRtHandOptions& options)
{
    validate_options(options);
    try
    {
        return std::unique_ptr<TensorRtHandInput>(
            new TensorRtHandInput(std::make_unique<Impl>(options)));
    }
    catch (const HandModelError&)
    {
        throw;
    }
    catch (const image::ImageProcessorError& error)
    {
        rethrow_image(error);
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("shared frame input allocation failed");
    }
}

image::FrameView TensorRtHandInput::prepare(const image::ImageView& source)
{
    if (!impl_)
    {
        throw_invalid("shared frame input state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    try
    {
        const image::ImageView compute = detail::stage_host_or_borrow_cuda(
            source, [this](const image::ImageView& host) {
                return impl_->processor->stage(host);
            });
        return { source, compute };
    }
    catch (const HandModelError&)
    {
        throw;
    }
    catch (const image::ImageProcessorError& error)
    {
        rethrow_image(error);
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("shared frame input preparation failed");
    }
}

struct TensorRtHandBackend::Impl final
{
    Impl(const HandTensorRtEnginePaths& paths, const TensorRtHandOptions& options_value)
        : options(options_value)
        , processor(image::CudaImageProcessor::create(processor_options(options)))
        , palm(paths.palm, model_options(options))
        , hand(paths.hand_landmark, model_options(options))
        , classifier(paths.keypoint_classifier, model_options(options))
    {
    }

    TensorRtHandOptions                     options;
    std::unique_ptr<image::CudaImageProcessor> processor;
    detail::PalmTensorRtModel                  palm;
    detail::HandLandmarkTensorRtModel          hand;
    detail::KeypointClassifierTensorRtModel    classifier;
    std::atomic_flag                           in_use = ATOMIC_FLAG_INIT;
};

TensorRtHandBackend::TensorRtHandBackend(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

TensorRtHandBackend::~TensorRtHandBackend() = default;

std::unique_ptr<TensorRtHandBackend> TensorRtHandBackend::load(
    const HandTensorRtEnginePaths& paths, const TensorRtHandOptions& options)
{
    validate_options(options);
    validate_model_asset(paths.palm, "Palm", options.max_engine_bytes);
    validate_model_asset(paths.hand_landmark, "hand landmark", options.max_engine_bytes);
    validate_model_asset(paths.keypoint_classifier, "keypoint classifier",
                         options.max_engine_bytes);
    try
    {
        auto impl = std::make_unique<Impl>(paths, options);
        return std::unique_ptr<TensorRtHandBackend>(new TensorRtHandBackend(std::move(impl)));
    }
    catch (const HandModelError&)
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
        throw_resource("hand backend allocation failed");
    }
}

HandFrame TensorRtHandBackend::infer(const image::ImageView& source)
{
    if (!impl_)
    {
        throw_invalid("hand backend state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    try
    {
        const Clock::time_point total_started = Clock::now();
        HandFrame result;

        const Clock::time_point palm_preprocess_started = Clock::now();
        const image::ImageView staged = detail::stage_host_or_borrow_cuda(
            source, [this](const image::ImageView& host) {
                return impl_->processor->stage(host);
            });
        const image::LetterboxTransform letterbox = image::ImageProcessor::letterbox_transform(
            staged.width, staged.height, kPalmInputExtent, kPalmInputExtent);
        const image::TensorView palm_input = impl_->processor->process_affine(
            staged, kPalmInputExtent, kPalmInputExtent, letterbox_affine(letterbox),
            rgb_unit_options(), image::TensorElementType::Float32);
        result.timings.preprocess_ms += elapsed_ms(palm_preprocess_started);

        const Clock::time_point palm_inference_started = Clock::now();
        const std::vector<float> palm_values = impl_->palm.run(palm_input);
        result.timings.palm_inference_ms = elapsed_ms(palm_inference_started);
        const std::vector<PalmDetection> palms = detail::decode_palms(
            palm_values.data(), palm_values.size(), impl_->options.palm_score_threshold,
            impl_->options.max_palm_candidates, impl_->options.max_hands, letterbox,
            kPalmInputExtent);
        if (palms.empty())
        {
            result.timings.total_ms = elapsed_ms(total_started);
            return result;
        }

        std::vector<std::array<float, kClassifierFeatureWidth>> classifier_features;
        classifier_features.reserve(palms.size());
        result.hands.reserve(palms.size());
        for (const PalmDetection& palm : palms)
        {
            const Clock::time_point hand_preprocess_started = Clock::now();
            const image::TensorView hand_input = impl_->processor->process_affine(
                staged, kHandLandmarkInputExtent, kHandLandmarkInputExtent,
                detail::hand_roi_transform(palm.roi, kHandLandmarkInputExtent),
                rgb_unit_options(), image::TensorElementType::Float32);
            result.timings.preprocess_ms += elapsed_ms(hand_preprocess_started);

            const Clock::time_point hand_inference_started = Clock::now();
            const detail::HandTensorRtOutputs hand_output = impl_->hand.run(hand_input);
            result.timings.landmark_inference_ms += elapsed_ms(hand_inference_started);
            if (!std::isfinite(hand_output.score))
            {
                throw_vision(HandModelErrorCode::ModelContractMismatch,
                             "hand landmark score is non-finite");
            }
            if (hand_output.score < impl_->options.hand_score_threshold)
            {
                continue;
            }

            HandResult hand_result;
            hand_result.palm                = palm;
            hand_result.landmark_confidence = hand_output.score;
            hand_result.handedness = detail::decode_handedness(hand_output.handedness);
            hand_result.landmarks = detail::decode_hand_landmarks(
                hand_output.landmarks.data(), hand_output.landmarks.size(), palm.roi,
                kHandLandmarkInputExtent);
            classifier_features.push_back(
                detail::make_keypoint_features(hand_result.landmarks));
            result.hands.push_back(std::move(hand_result));
        }

        if (!result.hands.empty())
        {
            const Clock::time_point classifier_preprocess_started = Clock::now();
            std::vector<float> classifier_input;
            classifier_input.reserve(result.hands.size() * kClassifierFeatureWidth);
            for (const auto& features : classifier_features)
            {
                classifier_input.insert(classifier_input.end(), features.begin(),
                                        features.end());
            }
            result.timings.preprocess_ms += elapsed_ms(classifier_preprocess_started);
            const Clock::time_point classifier_inference_started = Clock::now();
            const std::vector<std::int64_t> gestures = impl_->classifier.run(
                classifier_input, result.hands.size());
            result.timings.classifier_inference_ms = elapsed_ms(classifier_inference_started);
            if (gestures.size() != result.hands.size())
            {
                throw_vision(HandModelErrorCode::ModelContractMismatch,
                             "keypoint classifier output batch is invalid");
            }
            for (std::size_t index = 0; index < result.hands.size(); ++index)
            {
                result.hands[index].gesture = detail::decode_gesture(gestures[index]);
            }
        }
        result.timings.total_ms = elapsed_ms(total_started);
        return result;
    }
    catch (const HandModelError&)
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
        throw_resource("hand inference allocation failed");
    }
}

} // namespace kfcore::hand_models
