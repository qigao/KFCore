#include "kfcore/face_models/runtime.hpp"

#include "facemesh_decode.hpp"
#include "facemesh_geometry.hpp"
#include "kfcore/image_processor/cpu.hpp"
#include "kfcore/runtime/error.hpp"

#include <algorithm>
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
#include <utility>
#include <vector>

namespace kfcore::face_models
{
namespace
{

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::InvalidArgument,
                         "face runtime model: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::ModelContractMismatch,
                         "face runtime model contract: " + detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::ResourceLimitExceeded,
                         "face runtime model resource limit: " + detail);
}

[[noreturn]] void throw_runtime(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::RuntimeFailure,
                         "face runtime model execution: " + detail);
}

double elapsed_ms(Clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

void validate_options(const FaceRuntimeOptions& options)
{
    if (options.detector_input_extent <= 0 || options.landmarker_input_extent <= 0 ||
        options.max_detections == 0U || options.max_source_bytes == 0U ||
        options.max_tensor_bytes == 0U || options.max_output_bytes == 0U ||
        options.face_class_id < 0 ||
        !std::isfinite(options.face_detection_score_threshold) ||
        options.face_detection_score_threshold < 0.0F ||
        options.face_detection_score_threshold > 1.0F)
    {
        throw_invalid("options contain invalid dimensions, limits, class id, or threshold");
    }
}

void validate_host_image(const image::ImageView& image)
{
    if (image.memory_kind != image::MemoryKind::Host)
    {
        throw_invalid("v1 face preprocessing accepts Host images only");
    }
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

std::size_t element_size(runtime::DataType type)
{
    switch (type)
    {
    case runtime::DataType::Float32: return sizeof(float);
    case runtime::DataType::Float16: return sizeof(std::uint16_t);
    default: throw_contract("face tensors must use FP32 or FP16");
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
    {
        return static_cast<std::uint16_t>(sign | (mantissa == 0U ? 0x7c00U : 0x7e00U));
    }
    const int adjusted = static_cast<int>(exponent) - 127 + 15;
    if (adjusted >= 31)
    {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    if (adjusted <= 0)
    {
        if (adjusted < -10)
        {
            return static_cast<std::uint16_t>(sign);
        }
        const std::uint32_t normalized = mantissa | 0x800000U;
        const int shift = 14 - adjusted;
        const std::uint32_t rounded =
            (normalized + (UINT32_C(1) << (shift - 1))) >> shift;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const std::uint32_t rounded = mantissa + 0x1000U;
    if ((rounded & 0x800000U) != 0U)
    {
        return static_cast<std::uint16_t>(
            sign | (static_cast<std::uint32_t>(adjusted + 1) << 10U));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(adjusted) << 10U) | (rounded >> 13U));
}

float half_to_float(std::uint16_t bits) noexcept
{
    const bool negative = (bits & UINT16_C(0x8000)) != 0U;
    const std::uint16_t exponent = static_cast<std::uint16_t>((bits >> 10U) & 0x1fU);
    const std::uint16_t fraction = static_cast<std::uint16_t>(bits & 0x03ffU);
    float value = 0.0F;
    if (exponent == 0U)
    {
        value = std::ldexp(static_cast<float>(fraction), -24);
    }
    else if (exponent == 0x1fU)
    {
        value = fraction == 0U ? (std::numeric_limits<float>::infinity)()
                               : (std::numeric_limits<float>::quiet_NaN)();
    }
    else
    {
        value = std::ldexp(static_cast<float>(UINT16_C(0x0400) + fraction),
                           static_cast<int>(exponent) - 25);
    }
    return negative ? -value : value;
}

std::size_t element_count(const runtime::TensorShape& shape, const char* subject)
{
    std::size_t result = 1U;
    for (const std::int64_t dimension : shape)
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
                                         std::int32_t requested_extent,
                                         const char* subject)
{
    if (descriptor.shape.size() != 4U)
    {
        throw_contract(std::string(subject) + " input must be NCHW rank 4");
    }
    runtime::TensorShape shape = descriptor.shape;
    if (shape[0] == -1) shape[0] = 1;
    if (shape[1] == -1) shape[1] = 3;
    if (shape[2] == -1) shape[2] = requested_extent;
    if (shape[3] == -1) shape[3] = requested_extent;
    if (shape[0] != 1 || shape[1] != 3 ||
        shape[2] != requested_extent || shape[3] != requested_extent)
    {
        throw_contract(std::string(subject) + " input must resolve to [1,3,E,E]");
    }
    if (descriptor.data_type != runtime::DataType::Float32 &&
        descriptor.data_type != runtime::DataType::Float16)
    {
        throw_contract(std::string(subject) + " input must use FP32 or FP16");
    }
    return shape;
}

struct HostBuffer
{
    runtime::TensorDescriptor descriptor;
    runtime::TensorShape shape;
    std::vector<std::max_align_t> storage;
    std::size_t bytes = 0U;

    void allocate(std::size_t max_bytes)
    {
        const std::size_t count = element_count(shape, descriptor.name.c_str());
        bytes = checked_multiply(count, element_size(descriptor.data_type),
                                 descriptor.name.c_str());
        if (bytes > max_bytes)
        {
            throw_resource("output tensor exceeds max_output_bytes");
        }
        storage.resize((bytes + sizeof(std::max_align_t) - 1U) /
                       sizeof(std::max_align_t));
    }

    void* data() noexcept { return storage.empty() ? nullptr : storage.data(); }
    const void* data() const noexcept { return storage.empty() ? nullptr : storage.data(); }

    runtime::MutableTensorView view()
    {
        return {descriptor.name, descriptor.data_type, shape, data(), bytes,
                runtime::MemoryKind::Host, {}};
    }

    runtime::DynamicMutableTensorView dynamic_view()
    {
        return {descriptor.name, descriptor.data_type, data(), bytes,
                runtime::MemoryKind::Host, {}, {}, 0U};
    }

    std::vector<float> floats(std::size_t count) const
    {
        const std::size_t required = checked_multiply(
            count, element_size(descriptor.data_type), descriptor.name.c_str());
        if (required > bytes)
        {
            throw_resource("requested output values exceed allocated capacity");
        }
        std::vector<float> result(count);
        if (count == 0U)
        {
            return result;
        }
        if (descriptor.data_type == runtime::DataType::Float32)
        {
            std::memcpy(result.data(), data(), count * sizeof(float));
            return result;
        }
        const auto* input = static_cast<const std::uint16_t*>(data());
        std::transform(input, input + count, result.begin(), half_to_float);
        return result;
    }

    std::vector<float> floats() const
    {
        return floats(element_count(shape, descriptor.name.c_str()));
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
        throw_contract("input must use FP32 or FP16");
    }
    if (result.bytes > max_bytes)
    {
        throw_resource("input tensor exceeds max_tensor_bytes");
    }
    return result;
}

class UseGuard final
{
public:
    explicit UseGuard(std::atomic_flag& flag, const char* subject) : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw FaceModelError(FaceModelErrorCode::ConcurrentExecution,
                                 std::string(subject) + " calls must not overlap");
        }
    }
    ~UseGuard() { flag_.clear(std::memory_order_release); }
private:
    std::atomic_flag& flag_;
};

image::PreprocessOptions rgb_unit_options(float border = 0.0F)
{
    image::PreprocessOptions options;
    options.output_format = image::PixelFormat::Rgb8;
    options.border_value = border;
    return options;
}

} // namespace

struct FaceDetector::Impl final
{
    Impl(runtime::ResolvedModel value, FaceRuntimeOptions options_value)
        : resolved(std::move(value)), options(options_value),
          context(resolved.model->create_context())
    {
        const auto tensors = resolved.model->tensors();
        for (const auto& tensor : tensors)
        {
            (tensor.is_input ? inputs : outputs).push_back(tensor);
        }
        if (inputs.size() != 1U || outputs.size() != 1U)
        {
            throw_contract("face detector requires exactly one input and one output");
        }
        input_shape = resolve_image_input(inputs[0], options.detector_input_extent,
                                          "face detector");
        if (outputs[0].shape.size() != 3U)
        {
            throw_contract("face detector output must have rank 3");
        }
        output.descriptor = outputs[0];
        if ((output.descriptor.shape[0] != -1 && output.descriptor.shape[0] != 1) ||
            output.descriptor.shape[2] != 6)
        {
            throw_contract("face detector declared output must be [1,N,6]");
        }
        dynamic_output = output.descriptor.shape[1] == -1;
        output.shape = output.descriptor.shape;
        if (output.shape[0] == -1) output.shape[0] = 1;
        if (output.shape[1] == -1)
            output.shape[1] = static_cast<std::int64_t>(options.max_detections);
        if (output.shape[0] != 1 || output.shape[1] <= 0 || output.shape[2] != 6)
        {
            throw_contract("face detector output must resolve to [1,N,6]");
        }
        if (static_cast<std::uintmax_t>(output.shape[1]) >
            static_cast<std::uintmax_t>(options.max_detections))
        {
            throw_resource("face detector output rows exceed max_detections");
        }
        output.allocate(options.max_output_bytes);
    }

    runtime::ResolvedModel resolved;
    FaceRuntimeOptions options;
    std::unique_ptr<runtime::ExecutionContext> context;
    std::vector<runtime::TensorDescriptor> inputs;
    std::vector<runtime::TensorDescriptor> outputs;
    runtime::TensorShape input_shape;
    HostBuffer output;
    bool dynamic_output = false;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

FaceDetector::FaceDetector(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FaceDetector::~FaceDetector() = default;

std::unique_ptr<FaceDetector> FaceDetector::load(
    runtime::Runtime& runtime, const runtime::ModelPackage& package,
    const runtime::ExecutionPolicy& policy, const FaceRuntimeOptions& options)
{
    validate_options(options);
    if (package.model_type() != "face.detector")
    {
        throw_contract("ModelPackage model_type must be 'face.detector'");
    }
    try
    {
        return std::unique_ptr<FaceDetector>(new FaceDetector(std::make_unique<Impl>(
            runtime.load_model(package, policy), options)));
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("face detector allocation failed");
    }
}

FaceDetectionsResult FaceDetector::infer_all(const image::ImageView& source)
{
    if (!impl_) throw_invalid("face detector state is unavailable");
    UseGuard guard(impl_->in_use, "face detector");
    validate_host_image(source);
    try
    {
        const Clock::time_point total_started = Clock::now();
        FaceDetectionsResult result;
        image::LetterboxTransform letterbox;
        const Clock::time_point preprocess_started = Clock::now();
        PreparedInput input = prepare_input(
            image::CpuImageProcessor::letterbox_nchw(
                source, impl_->options.detector_input_extent,
                impl_->options.detector_input_extent, rgb_unit_options(114.0F),
                impl_->options.max_source_bytes, impl_->options.max_tensor_bytes,
                &letterbox),
            impl_->inputs[0].data_type, impl_->options.max_tensor_bytes);
        result.preprocess_ms = elapsed_ms(preprocess_started);

        const runtime::TensorView input_view{
            impl_->inputs[0].name, impl_->inputs[0].data_type, impl_->input_shape,
            input.data, input.bytes, runtime::MemoryKind::Host, {}};
        std::size_t detection_count = static_cast<std::size_t>(impl_->output.shape[1]);
        const Clock::time_point inference_started = Clock::now();
        if (impl_->dynamic_output)
        {
            std::vector<runtime::DynamicMutableTensorView> dynamic_outputs{
                impl_->output.dynamic_view()};
            impl_->context->run_dynamic({input_view}, dynamic_outputs);
            const auto& actual = dynamic_outputs.front();
            if (actual.shape.size() != 3U || actual.shape[0] != 1 || actual.shape[1] < 0 ||
                actual.shape[2] != 6)
            {
                throw_contract("face detector dynamic output must resolve to [1,N,6]");
            }
            if (static_cast<std::uintmax_t>(actual.shape[1]) >
                static_cast<std::uintmax_t>(impl_->options.max_detections))
            {
                throw_resource("face detector dynamic output exceeds max_detections");
            }
            detection_count = static_cast<std::size_t>(actual.shape[1]);
            const std::size_t actual_elements = checked_multiply(
                detection_count, 6U, "face detector dynamic output");
            const std::size_t expected_bytes = checked_multiply(
                actual_elements, element_size(impl_->output.descriptor.data_type),
                "face detector dynamic output");
            if (actual.byte_size != expected_bytes)
            {
                throw_contract("face detector dynamic output byte count does not match shape");
            }
        }
        else
        {
            auto output_view = impl_->output.view();
            impl_->context->run({input_view}, {output_view});
        }
        result.inference_ms = elapsed_ms(inference_started);

        const std::size_t value_count = checked_multiply(
            detection_count, 6U, "face detector output");
        const std::vector<float> values = impl_->output.floats(value_count);
        result.faces = detail::decode_yolo12_faces(
            values.data(), values.size(), impl_->options.face_class_id,
            impl_->options.face_detection_score_threshold, letterbox,
            source.width, source.height);
        result.total_ms = elapsed_ms(total_started);
        return result;
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const image::ImageProcessorError& error)
    {
        if (error.code() == image::ImageProcessorErrorCode::ResourceLimitExceeded)
            throw_resource(error.what());
        throw_invalid(error.what());
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("face detector inference allocation failed");
    }
}

FaceDetectionResult FaceDetector::infer(const image::ImageView& image)
{
    const FaceDetectionsResult all = infer_all(image);
    FaceDetectionResult result;
    result.preprocess_ms = all.preprocess_ms;
    result.inference_ms = all.inference_ms;
    result.total_ms = all.total_ms;
    const auto best = std::max_element(
        all.faces.begin(), all.faces.end(),
        [](const FaceDetection& left, const FaceDetection& right) {
            return left.confidence < right.confidence;
        });
    if (best != all.faces.end()) result.face = *best;
    return result;
}

const runtime::ExecutionRoute& FaceDetector::execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty{};
    return impl_ ? impl_->resolved.route : empty;
}

struct FaceLandmarker::Impl final
{
    Impl(runtime::ResolvedModel value, FaceRuntimeOptions options_value)
        : resolved(std::move(value)), options(options_value),
          context(resolved.model->create_context())
    {
        const auto tensors = resolved.model->tensors();
        for (const auto& tensor : tensors)
        {
            (tensor.is_input ? inputs : outputs).push_back(tensor);
        }
        if (inputs.size() != 1U || outputs.size() < 2U)
        {
            throw_contract("face landmarker requires one input and score/landmark outputs");
        }
        input_shape = resolve_image_input(inputs[0], options.landmarker_input_extent,
                                          "face landmarker");
        for (const auto& descriptor : outputs)
        {
            runtime::TensorShape shape = descriptor.shape;
            for (auto& dimension : shape)
            {
                if (dimension == -1) dimension = 1;
            }
            std::size_t count = 0U;
            try { count = element_count(shape, descriptor.name.c_str()); }
            catch (const FaceModelError&) { continue; }
            if (count == 1U && score.descriptor.name.empty())
            {
                score.descriptor = descriptor;
                score.shape = shape;
            }
            else if (count == kFaceMeshLandmarkCount * 3U && landmarks.descriptor.name.empty())
            {
                landmarks.descriptor = descriptor;
                landmarks.shape = shape;
            }
        }
        if (score.descriptor.name.empty() || landmarks.descriptor.name.empty())
        {
            throw_contract("cannot identify face confidence and 468x3 landmark outputs");
        }
        score.allocate(options.max_output_bytes);
        landmarks.allocate(options.max_output_bytes);
        if (score.bytes > options.max_output_bytes - landmarks.bytes)
        {
            throw_resource("aggregate face landmark outputs exceed max_output_bytes");
        }
    }

    runtime::ResolvedModel resolved;
    FaceRuntimeOptions options;
    std::unique_ptr<runtime::ExecutionContext> context;
    std::vector<runtime::TensorDescriptor> inputs;
    std::vector<runtime::TensorDescriptor> outputs;
    runtime::TensorShape input_shape;
    HostBuffer score;
    HostBuffer landmarks;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

FaceLandmarker::FaceLandmarker(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FaceLandmarker::~FaceLandmarker() = default;

std::unique_ptr<FaceLandmarker> FaceLandmarker::load(
    runtime::Runtime& runtime, const runtime::ModelPackage& package,
    const runtime::ExecutionPolicy& policy, const FaceRuntimeOptions& options)
{
    validate_options(options);
    if (package.model_type() != "face.landmarker")
    {
        throw_contract("ModelPackage model_type must be 'face.landmarker'");
    }
    try
    {
        return std::unique_ptr<FaceLandmarker>(new FaceLandmarker(std::make_unique<Impl>(
            runtime.load_model(package, policy), options)));
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("face landmarker allocation failed");
    }
}

FaceLandmarkResult FaceLandmarker::infer(const image::ImageView& source,
                                         const RectF& face_box)
{
    if (!impl_) throw_invalid("face landmarker state is unavailable");
    UseGuard guard(impl_->in_use, "face landmarker");
    validate_host_image(source);
    try
    {
        const Clock::time_point total_started = Clock::now();
        FaceLandmarkResult result;
        const Clock::time_point preprocess_started = Clock::now();
        const image::BgrImage packed = image::CpuImageProcessor::copy_bgr(
            source, impl_->options.max_source_bytes);
        const detail::FaceRoi roi = detail::make_face_roi(
            face_box, impl_->options.landmarker_input_extent);
        const image::BgrImage crop = image::CpuImageProcessor::warp_affine_bgr(
            packed, impl_->options.landmarker_input_extent,
            impl_->options.landmarker_input_extent, roi.destination_to_source,
            0.0F, impl_->options.max_source_bytes);
        PreparedInput input = prepare_input(
            image::CpuImageProcessor::to_nchw(
                crop, rgb_unit_options(), impl_->options.max_tensor_bytes),
            impl_->inputs[0].data_type, impl_->options.max_tensor_bytes);
        result.preprocess_ms = elapsed_ms(preprocess_started);

        const runtime::TensorView input_view{
            impl_->inputs[0].name, impl_->inputs[0].data_type, impl_->input_shape,
            input.data, input.bytes, runtime::MemoryKind::Host, {}};
        auto score_view = impl_->score.view();
        auto landmarks_view = impl_->landmarks.view();
        const Clock::time_point inference_started = Clock::now();
        impl_->context->run({input_view}, {score_view, landmarks_view});
        result.inference_ms = elapsed_ms(inference_started);

        const std::vector<float> score_values = impl_->score.floats();
        const std::vector<float> landmark_values = impl_->landmarks.floats();
        result.confidence = score_values.front();
        if (!std::isfinite(result.confidence))
        {
            throw_contract("face confidence is non-finite");
        }
        result.landmarks = detail::decode_face_landmarks(
            landmark_values.data(), landmark_values.size(), roi,
            impl_->options.landmarker_input_extent,
            impl_->options.face_coordinates_normalized);
        result.total_ms = elapsed_ms(total_started);
        return result;
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const image::ImageProcessorError& error)
    {
        if (error.code() == image::ImageProcessorErrorCode::ResourceLimitExceeded)
            throw_resource(error.what());
        throw_invalid(error.what());
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("face landmarker inference allocation failed");
    }
}

const runtime::ExecutionRoute& FaceLandmarker::execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty{};
    return impl_ ? impl_->resolved.route : empty;
}

} // namespace kfcore::face_models
