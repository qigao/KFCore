#include "kfcore/pose/rtmw.hpp"

#include "simcc_decode.hpp"
#include "rtmw_preprocess.hpp"

#include "kfcore/image_processor/cpu.hpp"
#include "kfcore/image_processor/error.hpp"
#include "kfcore/runtime/error.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::pose
{
namespace
{

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw PoseError(PoseErrorCode::InvalidArgument, "RTMW: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw PoseError(PoseErrorCode::ModelContractMismatch,
                    "RTMW model contract: " + detail);
}

[[noreturn]] void throw_runtime(const std::string& detail)
{
    throw PoseError(PoseErrorCode::RuntimeFailure, "RTMW runtime: " + detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw PoseError(PoseErrorCode::ResourceLimitExceeded,
                    "RTMW resource limit: " + detail);
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char* subject)
{
    if (left != 0U && right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_resource(std::string(subject) + " byte count overflow");
    }
    return left * right;
}

std::size_t element_size(runtime::DataType type)
{
    switch (type)
    {
    case runtime::DataType::Float32: return sizeof(float);
    case runtime::DataType::Float16: return sizeof(std::uint16_t);
    default: throw_contract("RTMW tensors must use FP32 or FP16");
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
        sign | (static_cast<std::uint32_t>(adjusted) << 10U) |
        (rounded >> 13U));
}

detail::SimccElementType simcc_element_type(runtime::DataType type)
{
    switch (type)
    {
    case runtime::DataType::Float32:
        return detail::SimccElementType::Float32;
    case runtime::DataType::Float16:
        return detail::SimccElementType::Float16;
    default:
        throw_contract("SimCC output must use FP32 or FP16");
    }
}

void validate_options(const RtmwOptions& options)
{
    if ((options.input_width < 0) || (options.input_height < 0) ||
        ((options.input_width == 0) != (options.input_height == 0)))
    {
        throw_invalid("input width and height must both be zero or both be positive");
    }
    if (!std::isfinite(options.bbox_padding) || options.bbox_padding <= 0.0F ||
        !std::isfinite(options.simcc_split_ratio) || options.simcc_split_ratio <= 0.0F)
    {
        throw_invalid("bbox padding and SimCC split ratio must be finite and positive");
    }
    if (!std::isfinite(options.border_value) || options.border_value < 0.0F ||
        options.border_value > 255.0F)
    {
        throw_invalid("border value must be finite within [0,255]");
    }
    for (std::size_t channel = 0U; channel < 3U; ++channel)
    {
        if (!std::isfinite(options.mean[channel]) ||
            !std::isfinite(options.stddev[channel]) || options.stddev[channel] <= 0.0F)
        {
            throw_invalid("normalization mean must be finite and stddev positive");
        }
    }
    if (options.max_source_bytes == 0U || options.max_tensor_bytes == 0U ||
        options.max_output_bytes == 0U)
    {
        throw_resource("configured byte limits must be positive");
    }
}

std::int32_t resolve_spatial(std::int64_t declared, std::int32_t requested,
                             const char* name)
{
    if (declared > 0)
    {
        if (declared > (std::numeric_limits<std::int32_t>::max)())
        {
            throw_contract(std::string(name) + " exceeds int32 range");
        }
        const auto value = static_cast<std::int32_t>(declared);
        if (requested > 0 && requested != value)
        {
            throw_contract(std::string(name) + " conflicts with static artifact shape");
        }
        return value;
    }
    if (declared == -1 && requested > 0)
    {
        return requested;
    }
    throw_contract(std::string(name) +
                   " is dynamic; RtmwOptions must provide an explicit size");
}

runtime::TensorShape resolve_input_shape(const runtime::TensorDescriptor& input,
                                         const RtmwOptions& options,
                                         std::int32_t& input_width,
                                         std::int32_t& input_height)
{
    if (input.shape.size() != 4U)
    {
        throw_contract("input tensor must be NCHW rank 4");
    }
    runtime::TensorShape shape = input.shape;
    if (shape[0] == -1)
    {
        shape[0] = 1;
    }
    if (shape[0] != 1)
    {
        throw_contract("RTMW v1 supports batch size 1 per execution context call");
    }
    if (shape[1] == -1)
    {
        shape[1] = 3;
    }
    if (shape[1] != 3)
    {
        throw_contract("input tensor must have three channels");
    }
    input_height = resolve_spatial(shape[2], options.input_height, "input height");
    input_width = resolve_spatial(shape[3], options.input_width, "input width");
    shape[2] = input_height;
    shape[3] = input_width;
    return shape;
}

std::size_t expected_simcc_extent(std::int32_t input_extent, float split_ratio,
                                  const char* axis)
{
    const double value = static_cast<double>(input_extent) *
                         static_cast<double>(split_ratio);
    const auto rounded = static_cast<std::int64_t>(std::llround(value));
    if (rounded <= 0 || std::fabs(value - static_cast<double>(rounded)) > 1.0e-4)
    {
        throw_contract(std::string("SimCC ") + axis +
                       " extent is not integral for the configured split ratio");
    }
    return static_cast<std::size_t>(rounded);
}

bool compatible_simcc_shape(const runtime::TensorDescriptor& tensor,
                            std::size_t expected_keypoints,
                            std::size_t expected_extent)
{
    if (tensor.shape.size() == 2U)
    {
        return (tensor.shape[0] == -1 ||
                tensor.shape[0] == static_cast<std::int64_t>(expected_keypoints)) &&
               (tensor.shape[1] == -1 ||
                tensor.shape[1] == static_cast<std::int64_t>(expected_extent));
    }
    if (tensor.shape.size() == 3U)
    {
        return (tensor.shape[0] == -1 || tensor.shape[0] == 1) &&
               (tensor.shape[1] == -1 ||
                tensor.shape[1] == static_cast<std::int64_t>(expected_keypoints)) &&
               (tensor.shape[2] == -1 ||
                tensor.shape[2] == static_cast<std::int64_t>(expected_extent));
    }
    return false;
}

const runtime::TensorDescriptor& choose_simcc_output(
    const std::vector<runtime::TensorDescriptor>& outputs,
    const std::string& requested_name,
    std::size_t expected_keypoints,
    std::size_t expected_extent,
    const char* axis)
{
    if (!requested_name.empty())
    {
        const auto named = std::find_if(outputs.begin(), outputs.end(),
            [&](const runtime::TensorDescriptor& tensor) {
                return tensor.name == requested_name;
            });
        if (named != outputs.end())
        {
            if (!compatible_simcc_shape(*named, expected_keypoints, expected_extent))
            {
                throw_contract(std::string("named SimCC ") + axis +
                               " tensor has an incompatible shape");
            }
            return *named;
        }
    }

    const runtime::TensorDescriptor* match = nullptr;
    for (const auto& tensor : outputs)
    {
        if (!compatible_simcc_shape(tensor, expected_keypoints, expected_extent))
        {
            continue;
        }
        if (match != nullptr)
        {
            throw_contract(std::string("multiple candidate SimCC ") + axis +
                           " outputs; configure the tensor name explicitly");
        }
        match = &tensor;
    }
    if (match == nullptr)
    {
        throw_contract(std::string("missing SimCC ") + axis + " output tensor");
    }
    return *match;
}

runtime::TensorShape resolved_simcc_shape(const runtime::TensorDescriptor& tensor,
                                          std::size_t keypoint_count,
                                          std::size_t extent)
{
    runtime::TensorShape shape = tensor.shape;
    if (shape.size() == 2U)
    {
        if (shape[0] == -1)
        {
            shape[0] = static_cast<std::int64_t>(keypoint_count);
        }
        if (shape[1] == -1)
        {
            shape[1] = static_cast<std::int64_t>(extent);
        }
        return shape;
    }
    if (shape.size() == 3U)
    {
        if (shape[0] == -1)
        {
            shape[0] = 1;
        }
        if (shape[1] == -1)
        {
            shape[1] = static_cast<std::int64_t>(keypoint_count);
        }
        if (shape[2] == -1)
        {
            shape[2] = static_cast<std::int64_t>(extent);
        }
        return shape;
    }
    throw_contract("SimCC output rank must be 2 or 3");
}

class UseGuard final
{
public:
    explicit UseGuard(std::atomic_flag& flag)
        : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw_invalid("calls on one RTMW instance must not overlap");
        }
    }

    ~UseGuard()
    {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag& flag_;
};

struct HostTensorBuffer
{
    runtime::TensorDescriptor descriptor;
    runtime::TensorShape shape;
    std::vector<std::max_align_t> storage;
    std::size_t bytes = 0U;

    void allocate(std::size_t keypoint_count,
                  std::size_t extent,
                  std::size_t max_output_bytes)
    {
        shape = resolved_simcc_shape(descriptor, keypoint_count, extent);
        const std::size_t elements = checked_multiply(
            keypoint_count, extent, descriptor.name.c_str());
        bytes = checked_multiply(elements, element_size(descriptor.data_type),
                                 descriptor.name.c_str());
        if (bytes > max_output_bytes)
        {
            throw_resource("SimCC output exceeds configured output byte limit");
        }
        const std::size_t units =
            (bytes + sizeof(std::max_align_t) - 1U) / sizeof(std::max_align_t);
        storage.resize(units);
    }

    void* data() noexcept
    {
        return storage.empty() ? nullptr : storage.data();
    }

    const void* data() const noexcept
    {
        return storage.empty() ? nullptr : storage.data();
    }

    runtime::MutableTensorView view()
    {
        return {descriptor.name, descriptor.data_type, shape, data(), bytes,
                runtime::MemoryKind::Host, {}};
    }
};

} // namespace

struct Rtmw::Impl final
{
    Impl(runtime::ResolvedModel resolved_value,
         RtmwOptions options_value,
         const PoseSchema& schema_value)
        : resolved(std::move(resolved_value))
        , options(std::move(options_value))
        , pose_schema(&schema_value)
        , keypoint_count(schema_value.output_map.size())
        , context(resolved.model->create_context())
    {
        if (keypoint_count == 0U)
        {
            throw_contract("selected PoseSchema has an empty output map");
        }
        validate_contract();
    }

    void validate_contract()
    {
        const auto tensors = resolved.model->tensors();
        std::vector<runtime::TensorDescriptor> inputs;
        std::vector<runtime::TensorDescriptor> outputs;
        for (const auto& tensor : tensors)
        {
            (tensor.is_input ? inputs : outputs).push_back(tensor);
        }
        if (inputs.size() != 1U)
        {
            throw_contract("RTMW requires exactly one image input tensor");
        }
        if (outputs.size() < 2U)
        {
            throw_contract("RTMW requires SimCC X and Y output tensors");
        }

        input_descriptor = inputs.front();
        if (!options.input_name.empty() && input_descriptor.name != options.input_name)
        {
            throw_contract("input tensor name does not match RtmwOptions::input_name");
        }
        if (input_descriptor.data_type != runtime::DataType::Float32 &&
            input_descriptor.data_type != runtime::DataType::Float16)
        {
            throw_contract("input tensor must use FP32 or FP16");
        }
        input_shape = resolve_input_shape(input_descriptor, options,
                                          input_width_value, input_height_value);

        simcc_x_extent = expected_simcc_extent(input_width_value,
                                               options.simcc_split_ratio, "X");
        simcc_y_extent = expected_simcc_extent(input_height_value,
                                               options.simcc_split_ratio, "Y");
        simcc_x.descriptor = choose_simcc_output(
            outputs, options.simcc_x_name,
            keypoint_count, simcc_x_extent, "X");
        simcc_y.descriptor = choose_simcc_output(
            outputs, options.simcc_y_name,
            keypoint_count, simcc_y_extent, "Y");
        if (simcc_x.descriptor.name == simcc_y.descriptor.name)
        {
            throw_contract("SimCC X and Y must be distinct output tensors");
        }
        if ((simcc_x.descriptor.data_type != runtime::DataType::Float32 &&
             simcc_x.descriptor.data_type != runtime::DataType::Float16) ||
            (simcc_y.descriptor.data_type != runtime::DataType::Float32 &&
             simcc_y.descriptor.data_type != runtime::DataType::Float16))
        {
            throw_contract("SimCC outputs must use FP32 or FP16");
        }
        simcc_x.allocate(keypoint_count, simcc_x_extent, options.max_output_bytes);
        simcc_y.allocate(keypoint_count, simcc_y_extent, options.max_output_bytes);
    }

    PoseResult infer_pose_one(const image::BgrImage& source, const RectF& box)
    {
        detail::RtmwPreprocessResult preprocess = detail::preprocess_rtmw(
            source, box, options, input_width_value, input_height_value);
        const std::vector<float>& input_float = preprocess.nchw;

        const std::size_t input_bytes = checked_multiply(
            input_float.size(), element_size(input_descriptor.data_type), "input tensor");
        if (input_bytes > options.max_tensor_bytes)
        {
            throw_resource("input tensor exceeds configured byte limit");
        }

        std::vector<std::uint16_t> input_half;
        const void* input_data = input_float.data();
        if (input_descriptor.data_type == runtime::DataType::Float16)
        {
            input_half.resize(input_float.size());
            std::transform(input_float.begin(), input_float.end(), input_half.begin(),
                           [](float value) { return float_to_half(value); });
            input_data = input_half.data();
        }

        runtime::TensorView input_view {
            input_descriptor.name,
            input_descriptor.data_type,
            input_shape,
            input_data,
            input_bytes,
            runtime::MemoryKind::Host,
            {},
        };
        std::vector<runtime::MutableTensorView> outputs {
            simcc_x.view(), simcc_y.view()
        };
        context->run({input_view}, outputs);

        std::vector<detail::DecodedSimccKeypoint> decoded(keypoint_count);
        detail::decode_simcc(
            {simcc_x.data(), simcc_element_type(simcc_x.descriptor.data_type),
             keypoint_count, simcc_x_extent},
            {simcc_y.data(), simcc_element_type(simcc_y.descriptor.data_type),
             keypoint_count, simcc_y_extent},
            options.simcc_split_ratio, decoded.data(), decoded.size());

        PoseResult result;
        result.source_box = box;
        result.schema_id = pose_schema->id;
        result.capabilities = kPoseCapabilityConfidence;
        result.keypoints.reserve(keypoint_count);

        for (std::size_t channel = 0U; channel < keypoint_count; ++channel)
        {
            const detail::DecodedSimccKeypoint& decoded_point = decoded[channel];
            PoseKeypoint point;
            point.id = pose_schema->output_map[channel];
            point.confidence = decoded_point.confidence;
            if (point.confidence > 0.0F)
            {
                const auto source_point = detail::rtmw_model_to_source(
                    preprocess.geometry, decoded_point.x, decoded_point.y,
                    input_width_value, input_height_value);
                point.x = source_point.first;
                point.y = source_point.second;
                point.flags |= kPoseKeypointFlagValid;
            }
            result.keypoints.push_back(point);
        }
        return result;
    }

    runtime::ResolvedModel resolved;
    RtmwOptions options;
    const PoseSchema* pose_schema = nullptr;
    std::size_t keypoint_count = 0U;
    std::unique_ptr<runtime::ExecutionContext> context;
    runtime::TensorDescriptor input_descriptor;
    runtime::TensorShape input_shape;
    std::int32_t input_width_value = 0;
    std::int32_t input_height_value = 0;
    std::size_t simcc_x_extent = 0U;
    std::size_t simcc_y_extent = 0U;
    HostTensorBuffer simcc_x;
    HostTensorBuffer simcc_y;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

Rtmw::Rtmw(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

Rtmw::~Rtmw() = default;

std::unique_ptr<Rtmw> Rtmw::load(runtime::Runtime& runtime,
                                 const runtime::ModelPackage& package,
                                 const runtime::ExecutionPolicy& policy,
                                 const RtmwOptions& options)
{
    validate_options(options);
    if (package.model_type() != kRtmwModelType)
    {
        throw_contract("ModelPackage model_type must be 'pose.rtmw'");
    }
    const PoseSchema& selected_schema = pose_schema_for_semantic_contract(
        package.semantic_contract(), package.semantic_version());
    try
    {
        runtime::ResolvedModel resolved = runtime.load_model(package, policy);
        return std::unique_ptr<Rtmw>(
            new Rtmw(std::make_unique<Impl>(
                std::move(resolved), options, selected_schema)));
    }
    catch (const PoseError&)
    {
        throw;
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
    catch (const image::ImageProcessorError& error)
    {
        if (error.code() == image::ImageProcessorErrorCode::ResourceLimitExceeded)
        {
            throw_resource(error.what());
        }
        throw_invalid(error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("model allocation failed");
    }
}

WholeBodyPose Rtmw::infer(const image::ImageView& image, const RectF& person_box)
{
    if (!impl_)
    {
        throw_invalid("model state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    try
    {
        const image::BgrImage source = image::CpuImageProcessor::copy_bgr(
            image, impl_->options.max_source_bytes);
        return impl_->infer_one(source, person_box);
    }
    catch (const PoseError&)
    {
        throw;
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
    catch (const image::ImageProcessorError& error)
    {
        if (error.code() == image::ImageProcessorErrorCode::ResourceLimitExceeded)
        {
            throw_resource(error.what());
        }
        throw_invalid(error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("inference allocation failed");
    }
}

std::vector<WholeBodyPose> Rtmw::infer(const image::ImageView& image,
                                       const std::vector<RectF>& person_boxes)
{
    if (!impl_)
    {
        throw_invalid("model state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    try
    {
        const image::BgrImage source = image::CpuImageProcessor::copy_bgr(
            image, impl_->options.max_source_bytes);
        std::vector<WholeBodyPose> result;
        result.reserve(person_boxes.size());
        for (const RectF& box : person_boxes)
        {
            result.push_back(impl_->infer_one(source, box));
        }
        return result;
    }
    catch (const PoseError&)
    {
        throw;
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
    catch (const image::ImageProcessorError& error)
    {
        if (error.code() == image::ImageProcessorErrorCode::ResourceLimitExceeded)
        {
            throw_resource(error.what());
        }
        throw_invalid(error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("batch inference allocation failed");
    }
}

std::int32_t Rtmw::input_width() const noexcept
{
    return impl_ ? impl_->input_width_value : 0;
}

std::int32_t Rtmw::input_height() const noexcept
{
    return impl_ ? impl_->input_height_value : 0;
}

const runtime::ExecutionRoute& Rtmw::execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty {};
    return impl_ ? impl_->resolved.route : empty;
}

} // namespace kfcore::pose
