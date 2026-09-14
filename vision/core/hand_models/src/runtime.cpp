#include "kfcore/hand_models/runtime.hpp"

#include "decode.hpp"
#include "geometry.hpp"
#include "kfcore/image_processor/cpu.hpp"
#include "kfcore/runtime/error.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kfcore::hand_models
{
namespace
{

using Clock = std::chrono::steady_clock;
constexpr std::size_t kImageChannels = 3U;
constexpr std::size_t kHandOutputWidth = kHandLandmarkCount * 3U;
constexpr std::size_t kClassifierFeatureWidth = kHandLandmarkCount * 2U;

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw HandModelError(HandModelErrorCode::InvalidArgument,
                         "hand runtime model: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw HandModelError(HandModelErrorCode::ModelContractMismatch,
                         "hand runtime model contract: " + detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw HandModelError(HandModelErrorCode::ResourceLimitExceeded,
                         "hand runtime model resource limit: " + detail);
}

[[noreturn]] void throw_runtime(const std::string& detail)
{
    throw HandModelError(HandModelErrorCode::RuntimeFailure,
                         "hand runtime model execution: " + detail);
}

double elapsed_ms(Clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char* subject)
{
    if (left != 0U && right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_resource(std::string(subject) + " size overflow");
    }
    return left * right;
}

void validate_options(const HandRuntimeOptions& options)
{
    if (options.max_source_bytes == 0U || options.max_tensor_bytes == 0U ||
        options.max_output_bytes == 0U || options.max_palm_candidates == 0U ||
        options.max_hands == 0U || options.max_hands > options.max_palm_candidates ||
        !std::isfinite(options.palm_score_threshold) ||
        !std::isfinite(options.hand_score_threshold) ||
        options.palm_score_threshold < 0.0F || options.palm_score_threshold > 1.0F ||
        options.hand_score_threshold < 0.0F || options.hand_score_threshold > 1.0F)
    {
        throw_invalid("options contain invalid limits or score thresholds");
    }
}

void validate_host_image(const image::ImageView& image)
{
    if (image.memory_kind != image::MemoryKind::Host)
    {
        throw_invalid("v1 hand preprocessing accepts Host images only");
    }
}

std::size_t element_size(runtime::DataType type)
{
    switch (type)
    {
    case runtime::DataType::Float32: return sizeof(float);
    case runtime::DataType::Float16: return sizeof(std::uint16_t);
    case runtime::DataType::Int32: return sizeof(std::int32_t);
    case runtime::DataType::Int64: return sizeof(std::int64_t);
    default: throw_contract("unsupported hand tensor data type");
    }
}

std::uint16_t float_to_half(float value) noexcept
{
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23U) & 0xffU;
    const std::uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU)
        return static_cast<std::uint16_t>(sign | (mantissa == 0U ? 0x7c00U : 0x7e00U));
    const int adjusted = static_cast<int>(exponent) - 127 + 15;
    if (adjusted >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
    if (adjusted <= 0)
    {
        if (adjusted < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t normalized = mantissa | 0x800000U;
        const int shift = 14 - adjusted;
        return static_cast<std::uint16_t>(
            sign | ((normalized + (UINT32_C(1) << (shift - 1))) >> shift));
    }
    const std::uint32_t rounded = mantissa + 0x1000U;
    if ((rounded & 0x800000U) != 0U)
        return static_cast<std::uint16_t>(
            sign | (static_cast<std::uint32_t>(adjusted + 1) << 10U));
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(adjusted) << 10U) | (rounded >> 13U));
}

float half_to_float(std::uint16_t bits) noexcept
{
    const bool negative = (bits & UINT16_C(0x8000)) != 0U;
    const std::uint16_t exponent = static_cast<std::uint16_t>((bits >> 10U) & 0x1fU);
    const std::uint16_t fraction = static_cast<std::uint16_t>(bits & 0x03ffU);
    float value = 0.0F;
    if (exponent == 0U) value = std::ldexp(static_cast<float>(fraction), -24);
    else if (exponent == 0x1fU)
        value = fraction == 0U ? (std::numeric_limits<float>::infinity)()
                               : (std::numeric_limits<float>::quiet_NaN)();
    else
        value = std::ldexp(static_cast<float>(UINT16_C(0x0400) + fraction),
                           static_cast<int>(exponent) - 25);
    return negative ? -value : value;
}

std::size_t element_count(const runtime::TensorShape& shape, const char* subject)
{
    std::size_t result = 1U;
    for (const auto dimension : shape)
    {
        if (dimension <= 0 ||
            static_cast<std::uintmax_t>(dimension) >
                static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)()))
        {
            throw_contract(std::string(subject) + " contains unresolved dimensions");
        }
        result = checked_multiply(result, static_cast<std::size_t>(dimension), subject);
    }
    return result;
}

runtime::TensorShape resolve_image_input(const runtime::TensorDescriptor& descriptor,
                                         std::int32_t extent,
                                         bool dynamic_batch,
                                         const char* subject)
{
    if (descriptor.shape.size() != 4U)
        throw_contract(std::string(subject) + " input must be NCHW rank 4");
    runtime::TensorShape shape = descriptor.shape;
    if (shape[0] == -1) shape[0] = dynamic_batch ? -1 : 1;
    if (shape[1] == -1) shape[1] = 3;
    if (shape[2] == -1) shape[2] = extent;
    if (shape[3] == -1) shape[3] = extent;
    if ((!dynamic_batch && shape[0] != 1) ||
        (dynamic_batch && shape[0] != -1 && shape[0] <= 0) ||
        shape[1] != 3 || shape[2] != extent || shape[3] != extent)
    {
        throw_contract(std::string(subject) + " image input has incompatible shape");
    }
    if (descriptor.data_type != runtime::DataType::Float32 &&
        descriptor.data_type != runtime::DataType::Float16)
    {
        throw_contract(std::string(subject) + " input must use FP32 or FP16");
    }
    return shape;
}

runtime::TensorShape batch_shape(const runtime::TensorDescriptor& descriptor,
                                 std::size_t batch,
                                 std::int64_t width,
                                 const char* subject)
{
    if (descriptor.shape.size() != 2U)
        throw_contract(std::string(subject) + " output must have rank 2");
    runtime::TensorShape shape = descriptor.shape;
    if (shape[0] == -1) shape[0] = static_cast<std::int64_t>(batch);
    if (shape[1] == -1) shape[1] = width;
    if (shape[0] != static_cast<std::int64_t>(batch) || shape[1] != width)
        throw_contract(std::string(subject) + " output shape does not match runtime batch");
    return shape;
}

struct HostBuffer
{
    runtime::TensorDescriptor descriptor;
    std::vector<std::max_align_t> storage;
    std::size_t capacity_bytes = 0U;

    void allocate(std::size_t elements, std::size_t max_bytes)
    {
        capacity_bytes = checked_multiply(elements, element_size(descriptor.data_type),
                                          descriptor.name.c_str());
        if (capacity_bytes > max_bytes)
            throw_resource("output tensor exceeds max_output_bytes");
        storage.resize((capacity_bytes + sizeof(std::max_align_t) - 1U) /
                       sizeof(std::max_align_t));
    }

    void* data() noexcept { return storage.empty() ? nullptr : storage.data(); }
    const void* data() const noexcept { return storage.empty() ? nullptr : storage.data(); }

    runtime::MutableTensorView view(const runtime::TensorShape& shape)
    {
        const std::size_t required = checked_multiply(
            element_count(shape, descriptor.name.c_str()), element_size(descriptor.data_type),
            descriptor.name.c_str());
        if (required > capacity_bytes) throw_resource("runtime output exceeds allocated capacity");
        return {descriptor.name, descriptor.data_type, shape, data(), capacity_bytes,
                runtime::MemoryKind::Host, {}};
    }

    std::vector<float> floats(std::size_t count) const
    {
        if (descriptor.data_type != runtime::DataType::Float32 &&
            descriptor.data_type != runtime::DataType::Float16)
            throw_contract("requested floating values from non-floating tensor");
        std::vector<float> result(count);
        if (descriptor.data_type == runtime::DataType::Float32)
        {
            std::memcpy(result.data(), data(), count * sizeof(float));
        }
        else
        {
            const auto* input = static_cast<const std::uint16_t*>(data());
            std::transform(input, input + count, result.begin(), half_to_float);
        }
        return result;
    }

    std::int64_t integer(std::size_t index) const
    {
        if (descriptor.data_type == runtime::DataType::Int64)
            return static_cast<const std::int64_t*>(data())[index];
        if (descriptor.data_type == runtime::DataType::Int32)
            return static_cast<const std::int32_t*>(data())[index];
        throw_contract("gesture classifier output must use Int32 or Int64");
    }
};

struct PreparedInput
{
    std::vector<float> fp32;
    std::vector<std::uint16_t> fp16;
    const void* data = nullptr;
    std::size_t bytes = 0U;
};

PreparedInput prepare_input(std::vector<float> values, runtime::DataType type,
                            std::size_t max_bytes)
{
    PreparedInput result;
    result.fp32 = std::move(values);
    if (type == runtime::DataType::Float32)
    {
        result.bytes = checked_multiply(result.fp32.size(), sizeof(float), "input tensor");
        result.data = result.fp32.data();
    }
    else if (type == runtime::DataType::Float16)
    {
        result.fp16.resize(result.fp32.size());
        std::transform(result.fp32.begin(), result.fp32.end(), result.fp16.begin(),
                       float_to_half);
        result.bytes = checked_multiply(result.fp16.size(), sizeof(std::uint16_t),
                                        "input tensor");
        result.data = result.fp16.data();
    }
    else
    {
        throw_contract("hand model input must use FP32 or FP16");
    }
    if (result.bytes > max_bytes) throw_resource("input tensor exceeds max_tensor_bytes");
    return result;
}

image::PreprocessOptions rgb_unit_options(float border = 0.0F)
{
    image::PreprocessOptions options;
    options.output_format = image::PixelFormat::Rgb8;
    options.border_value = border;
    return options;
}

class UseGuard final
{
public:
    explicit UseGuard(std::atomic_flag& flag) : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
            throw HandModelError(HandModelErrorCode::ConcurrentExecution,
                                 "hand runtime backend is already in use");
    }
    ~UseGuard() { flag_.clear(std::memory_order_release); }
private:
    std::atomic_flag& flag_;
};

const runtime::TensorDescriptor& one_input(
    const std::vector<runtime::TensorDescriptor>& tensors, const char* subject)
{
    const runtime::TensorDescriptor* result = nullptr;
    for (const auto& tensor : tensors)
    {
        if (!tensor.is_input) continue;
        if (result != nullptr) throw_contract(std::string(subject) + " requires one input");
        result = &tensor;
    }
    if (result == nullptr) throw_contract(std::string(subject) + " has no input");
    return *result;
}

std::vector<runtime::TensorDescriptor> outputs(
    const std::vector<runtime::TensorDescriptor>& tensors)
{
    std::vector<runtime::TensorDescriptor> result;
    for (const auto& tensor : tensors) if (!tensor.is_input) result.push_back(tensor);
    return result;
}

std::string lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

} // namespace

struct HandBackend::Impl final
{
    Impl(runtime::ResolvedModel palm_value,
         runtime::ResolvedModel landmark_value,
         runtime::ResolvedModel classifier_value,
         HandRuntimeOptions options_value)
        : palm(std::move(palm_value)), landmark(std::move(landmark_value)),
          classifier(std::move(classifier_value)), options(options_value),
          palm_context(palm.model->create_context()),
          landmark_context(landmark.model->create_context()),
          classifier_context(classifier.model->create_context())
    {
        configure_palm();
        configure_landmark();
        configure_classifier();
    }

    void configure_palm()
    {
        const auto tensors = palm.model->tensors();
        palm_input = one_input(tensors, "palm detector");
        palm_input_shape = resolve_image_input(palm_input, kPalmInputExtent, false,
                                               "palm detector");
        const auto out = outputs(tensors);
        if (out.size() != 1U || out[0].shape.size() != 2U)
            throw_contract("palm detector requires one [N,8] output");
        palm_output.descriptor = out[0];
        if (palm_output.descriptor.data_type != runtime::DataType::Float32 &&
            palm_output.descriptor.data_type != runtime::DataType::Float16)
            throw_contract("palm output must use FP32 or FP16");
        std::size_t rows = options.max_palm_candidates;
        if (out[0].shape[0] > 0) rows = static_cast<std::size_t>(out[0].shape[0]);
        if (out[0].shape[1] != -1 && out[0].shape[1] != static_cast<std::int64_t>(kPalmRowWidth))
            throw_contract("palm output width must be 8");
        palm_output.allocate(checked_multiply(rows, kPalmRowWidth, "palm output"),
                             options.max_output_bytes);
        palm_rows_capacity = rows;
    }

    void configure_landmark()
    {
        const auto tensors = landmark.model->tensors();
        landmark_input = one_input(tensors, "hand landmarker");
        landmark_input_shape = resolve_image_input(
            landmark_input, kHandLandmarkInputExtent, true, "hand landmarker");
        const auto out = outputs(tensors);
        if (out.size() != 3U) throw_contract("hand landmarker requires three outputs");
        for (const auto& descriptor : out)
        {
            const std::string name = lower(descriptor.name);
            if (descriptor.shape.size() != 2U) throw_contract("hand landmark outputs must have rank 2");
            const std::int64_t width = descriptor.shape[1];
            if ((width == -1 || width == static_cast<std::int64_t>(kHandOutputWidth)) &&
                (name.find("xyz") != std::string::npos || name.find("landmark") != std::string::npos))
            {
                landmark_xyz.descriptor = descriptor;
            }
            else if (width == -1 || width == 1)
            {
                if (name.find("score") != std::string::npos)
                    landmark_score.descriptor = descriptor;
                else if (name.find("left") != std::string::npos ||
                         name.find("right") != std::string::npos ||
                         name.find("handed") != std::string::npos)
                    handedness.descriptor = descriptor;
            }
        }
        if (landmark_xyz.descriptor.name.empty() || landmark_score.descriptor.name.empty() ||
            handedness.descriptor.name.empty())
            throw_contract("cannot identify landmark, score, and handedness outputs");
        for (auto* buffer : {&landmark_xyz, &landmark_score, &handedness})
        {
            if (buffer->descriptor.data_type != runtime::DataType::Float32 &&
                buffer->descriptor.data_type != runtime::DataType::Float16)
                throw_contract("hand landmark outputs must use FP32 or FP16");
        }
        landmark_xyz.allocate(checked_multiply(options.max_hands, kHandOutputWidth,
                                               "landmark output"),
                              options.max_output_bytes);
        landmark_score.allocate(options.max_hands, options.max_output_bytes);
        handedness.allocate(options.max_hands, options.max_output_bytes);
    }

    void configure_classifier()
    {
        const auto tensors = classifier.model->tensors();
        classifier_input = one_input(tensors, "gesture classifier");
        if (classifier_input.shape.size() != 2U ||
            (classifier_input.shape[1] != -1 &&
             classifier_input.shape[1] != static_cast<std::int64_t>(kClassifierFeatureWidth)) ||
            (classifier_input.data_type != runtime::DataType::Float32 &&
             classifier_input.data_type != runtime::DataType::Float16))
            throw_contract("gesture classifier input must be [B,42] FP32/FP16");
        const auto out = outputs(tensors);
        if (out.size() != 1U || out[0].shape.size() != 1U ||
            (out[0].data_type != runtime::DataType::Int32 &&
             out[0].data_type != runtime::DataType::Int64))
            throw_contract("gesture classifier output must be [B] Int32/Int64");
        classifier_output.descriptor = out[0];
        classifier_output.allocate(options.max_hands, options.max_output_bytes);
    }

    runtime::ResolvedModel palm;
    runtime::ResolvedModel landmark;
    runtime::ResolvedModel classifier;
    HandRuntimeOptions options;
    std::unique_ptr<runtime::ExecutionContext> palm_context;
    std::unique_ptr<runtime::ExecutionContext> landmark_context;
    std::unique_ptr<runtime::ExecutionContext> classifier_context;
    runtime::TensorDescriptor palm_input;
    runtime::TensorShape palm_input_shape;
    HostBuffer palm_output;
    std::size_t palm_rows_capacity = 0U;
    runtime::TensorDescriptor landmark_input;
    runtime::TensorShape landmark_input_shape;
    HostBuffer landmark_xyz;
    HostBuffer landmark_score;
    HostBuffer handedness;
    runtime::TensorDescriptor classifier_input;
    HostBuffer classifier_output;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

HandBackend::HandBackend(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
HandBackend::~HandBackend() = default;

std::unique_ptr<HandBackend> HandBackend::load(
    runtime::Runtime& runtime,
    const runtime::ModelPackage& palm_package,
    const runtime::ExecutionPolicy& palm_policy,
    const runtime::ModelPackage& landmark_package,
    const runtime::ExecutionPolicy& landmark_policy,
    const runtime::ModelPackage& classifier_package,
    const runtime::ExecutionPolicy& classifier_policy,
    const HandRuntimeOptions& options)
{
    validate_options(options);
    if (palm_package.model_type() != "hand.palm-detector" ||
        landmark_package.model_type() != "hand.landmarker" ||
        classifier_package.model_type() != "hand.gesture-classifier")
        throw_contract("hand packages must use canonical hand.* model_type values");
    try
    {
        return std::unique_ptr<HandBackend>(new HandBackend(std::make_unique<Impl>(
            runtime.load_model(palm_package, palm_policy),
            runtime.load_model(landmark_package, landmark_policy),
            runtime.load_model(classifier_package, classifier_policy), options)));
    }
    catch (const HandModelError&) { throw; }
    catch (const runtime::RuntimeError& error) { throw_runtime(error.what()); }
    catch (const std::bad_alloc&) { throw_resource("hand backend allocation failed"); }
}

HandFrame HandBackend::infer(const image::ImageView& source)
{
    if (!impl_) throw_invalid("hand backend state is unavailable");
    UseGuard guard(impl_->in_use);
    validate_host_image(source);
    try
    {
        const Clock::time_point total_started = Clock::now();
        HandFrame result;

        const Clock::time_point palm_preprocess_started = Clock::now();
        image::LetterboxTransform letterbox;
        PreparedInput palm_input = prepare_input(
            image::CpuImageProcessor::letterbox_nchw(
                source, kPalmInputExtent, kPalmInputExtent, rgb_unit_options(),
                impl_->options.max_source_bytes, impl_->options.max_tensor_bytes,
                &letterbox),
            impl_->palm_input.data_type, impl_->options.max_tensor_bytes);
        result.timings.preprocess_ms += elapsed_ms(palm_preprocess_started);

        const runtime::TensorView palm_input_view{
            impl_->palm_input.name, impl_->palm_input.data_type, impl_->palm_input_shape,
            palm_input.data, palm_input.bytes, runtime::MemoryKind::Host, {}};
        runtime::TensorShape palm_output_shape = impl_->palm_output.descriptor.shape;
        if (palm_output_shape[0] == -1)
            palm_output_shape[0] = static_cast<std::int64_t>(impl_->palm_rows_capacity);
        if (palm_output_shape[1] == -1)
            palm_output_shape[1] = static_cast<std::int64_t>(kPalmRowWidth);
        auto palm_output_view = impl_->palm_output.view(palm_output_shape);
        const Clock::time_point palm_inference_started = Clock::now();
        impl_->palm_context->run({palm_input_view}, {palm_output_view});
        result.timings.palm_inference_ms = elapsed_ms(palm_inference_started);
        const std::size_t palm_elements = element_count(palm_output_shape, "palm output");
        const std::vector<float> palm_values = impl_->palm_output.floats(palm_elements);
        std::vector<PalmDetection> palms = detail::decode_palms(
            palm_values.data(), palm_values.size(), impl_->options.palm_score_threshold,
            impl_->options.max_palm_candidates, impl_->options.max_hands,
            letterbox, kPalmInputExtent);
        if (palms.empty())
        {
            result.timings.total_ms = elapsed_ms(total_started);
            return result;
        }

        const Clock::time_point landmark_preprocess_started = Clock::now();
        const image::BgrImage packed = image::CpuImageProcessor::copy_bgr(
            source, impl_->options.max_source_bytes);
        const std::size_t one_hand_elements = kImageChannels *
            static_cast<std::size_t>(kHandLandmarkInputExtent) *
            static_cast<std::size_t>(kHandLandmarkInputExtent);
        std::vector<float> hand_batch;
        hand_batch.reserve(checked_multiply(palms.size(), one_hand_elements, "hand batch"));
        for (const PalmDetection& palm : palms)
        {
            const image::AffineTransform transform = detail::hand_roi_transform(
                palm.roi, kHandLandmarkInputExtent);
            const image::BgrImage crop = image::CpuImageProcessor::warp_affine_bgr(
                packed, kHandLandmarkInputExtent, kHandLandmarkInputExtent,
                transform, 0.0F, impl_->options.max_source_bytes);
            std::vector<float> tensor = image::CpuImageProcessor::to_nchw(
                crop, rgb_unit_options(), impl_->options.max_tensor_bytes);
            hand_batch.insert(hand_batch.end(), tensor.begin(), tensor.end());
        }
        PreparedInput landmark_input = prepare_input(
            std::move(hand_batch), impl_->landmark_input.data_type,
            impl_->options.max_tensor_bytes);
        result.timings.preprocess_ms += elapsed_ms(landmark_preprocess_started);

        const std::size_t hand_count = palms.size();
        runtime::TensorShape landmark_input_shape = impl_->landmark_input_shape;
        if (landmark_input_shape[0] == -1)
            landmark_input_shape[0] = static_cast<std::int64_t>(hand_count);
        if (landmark_input_shape[0] != static_cast<std::int64_t>(hand_count))
            throw_contract("hand landmarker static batch does not match detected hands");
        const runtime::TensorView landmark_input_view{
            impl_->landmark_input.name, impl_->landmark_input.data_type,
            landmark_input_shape, landmark_input.data, landmark_input.bytes,
            runtime::MemoryKind::Host, {}};
        const auto xyz_shape = batch_shape(impl_->landmark_xyz.descriptor, hand_count,
                                           static_cast<std::int64_t>(kHandOutputWidth),
                                           "hand landmarks");
        const auto score_shape = batch_shape(impl_->landmark_score.descriptor, hand_count, 1,
                                             "hand score");
        const auto handed_shape = batch_shape(impl_->handedness.descriptor, hand_count, 1,
                                              "handedness");
        auto xyz_view = impl_->landmark_xyz.view(xyz_shape);
        auto score_view = impl_->landmark_score.view(score_shape);
        auto handed_view = impl_->handedness.view(handed_shape);
        const Clock::time_point landmark_inference_started = Clock::now();
        impl_->landmark_context->run({landmark_input_view},
                                     {xyz_view, score_view, handed_view});
        result.timings.landmark_inference_ms = elapsed_ms(landmark_inference_started);
        const std::vector<float> xyz_values = impl_->landmark_xyz.floats(
            hand_count * kHandOutputWidth);
        const std::vector<float> score_values = impl_->landmark_score.floats(hand_count);
        const std::vector<float> handed_values = impl_->handedness.floats(hand_count);

        std::vector<std::array<float, kClassifierFeatureWidth>> classifier_features;
        classifier_features.reserve(hand_count);
        result.hands.reserve(hand_count);
        const Clock::time_point classifier_preprocess_started = Clock::now();
        for (std::size_t index = 0U; index < hand_count; ++index)
        {
            const float score = score_values[index];
            if (!std::isfinite(score)) throw_contract("hand landmark score is non-finite");
            if (score < impl_->options.hand_score_threshold) continue;
            HandResult hand_result;
            hand_result.palm = palms[index];
            hand_result.landmark_confidence = score;
            hand_result.handedness = detail::decode_handedness(handed_values[index]);
            hand_result.landmarks = detail::decode_hand_landmarks(
                xyz_values.data() + index * kHandOutputWidth,
                kHandOutputWidth, palms[index].roi, kHandLandmarkInputExtent);
            classifier_features.push_back(detail::make_keypoint_features(hand_result.landmarks));
            result.hands.push_back(std::move(hand_result));
        }
        result.timings.preprocess_ms += elapsed_ms(classifier_preprocess_started);

        if (!result.hands.empty())
        {
            std::vector<float> classifier_values;
            classifier_values.reserve(result.hands.size() * kClassifierFeatureWidth);
            for (const auto& feature : classifier_features)
                classifier_values.insert(classifier_values.end(), feature.begin(), feature.end());
            PreparedInput classifier_input = prepare_input(
                std::move(classifier_values), impl_->classifier_input.data_type,
                impl_->options.max_tensor_bytes);
            runtime::TensorShape classifier_input_shape = impl_->classifier_input.shape;
            if (classifier_input_shape[0] == -1)
                classifier_input_shape[0] = static_cast<std::int64_t>(result.hands.size());
            if (classifier_input_shape[1] == -1)
                classifier_input_shape[1] = static_cast<std::int64_t>(kClassifierFeatureWidth);
            if (classifier_input_shape[0] != static_cast<std::int64_t>(result.hands.size()) ||
                classifier_input_shape[1] != static_cast<std::int64_t>(kClassifierFeatureWidth))
                throw_contract("gesture classifier static input shape does not match runtime batch");
            const runtime::TensorView classifier_input_view{
                impl_->classifier_input.name, impl_->classifier_input.data_type,
                classifier_input_shape, classifier_input.data, classifier_input.bytes,
                runtime::MemoryKind::Host, {}};
            runtime::TensorShape classifier_output_shape = impl_->classifier_output.descriptor.shape;
            if (classifier_output_shape[0] == -1)
                classifier_output_shape[0] = static_cast<std::int64_t>(result.hands.size());
            if (classifier_output_shape[0] != static_cast<std::int64_t>(result.hands.size()))
                throw_contract("gesture classifier output batch does not match runtime batch");
            auto classifier_output_view = impl_->classifier_output.view(classifier_output_shape);
            const Clock::time_point classifier_inference_started = Clock::now();
            impl_->classifier_context->run({classifier_input_view}, {classifier_output_view});
            result.timings.classifier_inference_ms = elapsed_ms(classifier_inference_started);
            for (std::size_t index = 0U; index < result.hands.size(); ++index)
                result.hands[index].gesture = detail::decode_gesture(
                    impl_->classifier_output.integer(index));
        }

        result.timings.total_ms = elapsed_ms(total_started);
        return result;
    }
    catch (const HandModelError&) { throw; }
    catch (const image::ImageProcessorError& error)
    {
        if (error.code() == image::ImageProcessorErrorCode::ResourceLimitExceeded)
            throw_resource(error.what());
        throw_invalid(error.what());
    }
    catch (const runtime::RuntimeError& error) { throw_runtime(error.what()); }
    catch (const std::bad_alloc&) { throw_resource("hand inference allocation failed"); }
}

const runtime::ExecutionRoute& HandBackend::palm_execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty{};
    return impl_ ? impl_->palm.route : empty;
}

const runtime::ExecutionRoute& HandBackend::landmark_execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty{};
    return impl_ ? impl_->landmark.route : empty;
}

const runtime::ExecutionRoute& HandBackend::classifier_execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty{};
    return impl_ ? impl_->classifier.route : empty;
}

} // namespace kfcore::hand_models
