#include "tensor_validation.hpp"

#include "kfcore/tensorrt/error.hpp"

#include <limits>
#include <string>

namespace kfcore::tensorrt::detail
{
namespace
{

    [[noreturn]] void throw_contract(const std::string& stage, const std::string& detail)
    {
        throw TensorRtError(TensorRtErrorCode::EngineContractMismatch, stage + " stage: " + detail);
    }

    [[noreturn]] void throw_view(const std::string& stage, const std::string& detail)
    {
        throw TensorRtError(TensorRtErrorCode::InvalidTensorView, stage + " stage: " + detail);
    }

    [[noreturn]] void throw_resource(const std::string& stage, const std::string& detail)
    {
        throw TensorRtError(TensorRtErrorCode::ResourceLimitExceeded, stage + " stage: " + detail);
    }

    std::size_t checked_add(std::size_t left, std::size_t right, const char* stage)
    {
        if (right > (std::numeric_limits<std::size_t>::max)() - left)
        {
            throw_resource(stage, "aggregate byte count overflow");
        }
        return left + right;
    }

    std::size_t checked_dimension(std::int64_t value, const char* stage)
    {
        if (value <= 0)
        {
            throw_contract(stage, "shape dimensions must be positive");
        }
        const auto maximum = static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)());
        if (static_cast<std::uintmax_t>(value) > maximum)
        {
            throw_resource(stage, "shape dimension cannot be represented as size_t");
        }
        return static_cast<std::size_t>(value);
    }

    std::size_t checked_multiply(std::size_t left, std::size_t right, const char* stage)
    {
        if (left != 0 && right > (std::numeric_limits<std::size_t>::max)() / left)
        {
            throw_resource(stage, "shape byte count overflow");
        }
        return left * right;
    }

    void validate_options(const EngineOptions& options)
    {
        if (options.max_serialized_engine_bytes == 0 || options.max_tensor_count == 0 ||
            options.max_input_bytes == 0 || options.max_output_bytes == 0)
        {
            throw_resource("engine options", "all resource limits must be positive");
        }
    }

    void validate_expected_descriptors(const std::vector<TensorDescriptor>& expected,
                                       TensorIoMode mode, const char* stage)
    {
        for (std::size_t index = 0; index < expected.size(); ++index)
        {
            const TensorDescriptor& descriptor = expected[index];
            if (descriptor.name.empty())
            {
                throw_contract(stage, "tensor name must not be empty");
            }
            if (descriptor.mode != mode)
            {
                throw_contract(descriptor.name,
                               "tensor I/O mode does not match validation direction");
            }
            validate_profile_bounds(descriptor.profile, descriptor.name);
            for (std::size_t previous = 0; previous < index; ++previous)
            {
                if (expected[previous].name == descriptor.name)
                {
                    throw_contract(descriptor.name, "duplicate tensor name");
                }
            }
        }
    }

    template <typename View>
    void validate_views_impl(const std::vector<TensorDescriptor>& expected,
                             const std::vector<View>& views, TensorIoMode mode,
                             std::size_t max_aggregate_bytes, const char* stage)
    {
        validate_expected_descriptors(expected, mode, stage);
        std::size_t aggregate_bytes = 0;

        for (std::size_t index = 0; index < views.size(); ++index)
        {
            const View& view = views[index];
            if (view.name.empty())
            {
                throw_view(stage, "tensor view name must not be empty");
            }
            for (std::size_t previous = 0; previous < index; ++previous)
            {
                if (views[previous].name == view.name)
                {
                    throw_view(view.name, "duplicate tensor view");
                }
            }

            const TensorDescriptor* descriptor = nullptr;
            for (const TensorDescriptor& candidate : expected)
            {
                if (candidate.name == view.name)
                {
                    descriptor = &candidate;
                    break;
                }
            }
            if (descriptor == nullptr)
            {
                throw_view(view.name, "unexpected tensor view");
            }
            if (view.data == nullptr)
            {
                throw_view(view.name, "tensor view data must not be null");
            }
            if (view.data_type != descriptor->data_type)
            {
                throw_view(view.name, "tensor view data type does not match engine metadata");
            }
            if (view.shape.size() != descriptor->profile.minimum.size())
            {
                throw_view(view.name, "tensor view rank does not match engine metadata");
            }
            for (std::size_t dimension = 0; dimension < view.shape.size(); ++dimension)
            {
                const std::int64_t value = view.shape[dimension];
                if (value < descriptor->profile.minimum[dimension] ||
                    value > descriptor->profile.maximum[dimension])
                {
                    throw_view(view.name, "tensor view shape is outside profile 0 bounds");
                }
            }
            const std::size_t required_bytes =
                checked_shape_byte_size(view.shape, view.data_type, stage);
            if (view.byte_size < required_bytes)
            {
                throw_view(view.name, "tensor view capacity is smaller than required bytes");
            }
            aggregate_bytes = checked_add(aggregate_bytes, required_bytes, stage);
            if (aggregate_bytes > max_aggregate_bytes)
            {
                throw_resource(stage, "aggregate tensor view bytes exceed configured limit");
            }
        }

        for (const TensorDescriptor& descriptor : expected)
        {
            bool found = false;
            for (const View& view : views)
            {
                if (view.name == descriptor.name)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                throw_view(descriptor.name, "required tensor view is missing");
            }
        }
    }

} // namespace

std::size_t scalar_byte_size(DataType data_type)
{
    switch (data_type)
    {
    case DataType::Float32:
    case DataType::Int32:
        return sizeof(std::uint32_t);
    case DataType::Float16:
    case DataType::BFloat16:
        return sizeof(std::uint16_t);
    case DataType::Int8:
    case DataType::Bool:
    case DataType::UInt8:
        return sizeof(std::uint8_t);
    case DataType::Int64:
        return sizeof(std::int64_t);
    }
    throw_contract("tensor metadata", "data type is unsupported");
}

std::size_t checked_shape_byte_size(const TensorShape& shape, DataType data_type, const char* stage)
{
    std::size_t element_count = 1;
    for (const std::int64_t dimension : shape)
    {
        element_count = checked_multiply(element_count, checked_dimension(dimension, stage), stage);
    }
    return checked_multiply(element_count, scalar_byte_size(data_type), stage);
}

void validate_profile_bounds(const TensorProfile& profile, const std::string& tensor_name)
{
    const std::size_t rank = profile.minimum.size();
    if (profile.optimum.size() != rank || profile.maximum.size() != rank)
    {
        throw_contract(tensor_name, "profile ranks must match");
    }
    for (std::size_t index = 0; index < rank; ++index)
    {
        const std::int64_t minimum = profile.minimum[index];
        const std::int64_t optimum = profile.optimum[index];
        const std::int64_t maximum = profile.maximum[index];
        if (minimum <= 0 || optimum <= 0 || maximum <= 0)
        {
            throw_contract(tensor_name, "profile dimensions must be positive");
        }
        if (minimum > optimum || optimum > maximum)
        {
            throw_contract(tensor_name, "profile dimensions must satisfy min <= opt <= max");
        }
    }
}

void validate_tensor_metadata(const std::vector<TensorDescriptor>& tensors,
                              const EngineOptions&                 options)
{
    validate_options(options);
    if (tensors.empty())
    {
        throw_contract("tensor metadata", "engine must expose at least one I/O tensor");
    }
    if (tensors.size() > options.max_tensor_count)
    {
        throw_resource("tensor metadata", "tensor count exceeds configured limit");
    }

    std::size_t input_bytes  = 0;
    std::size_t output_bytes = 0;
    for (std::size_t index = 0; index < tensors.size(); ++index)
    {
        const TensorDescriptor& tensor = tensors[index];
        if (tensor.name.empty())
        {
            throw_contract("tensor metadata", "tensor name must not be empty");
        }
        for (std::size_t previous = 0; previous < index; ++previous)
        {
            if (tensors[previous].name == tensor.name)
            {
                throw_contract(tensor.name, "duplicate tensor name");
            }
        }
        validate_profile_bounds(tensor.profile, tensor.name);
        const std::size_t bytes =
            checked_shape_byte_size(tensor.profile.maximum, tensor.data_type, tensor.name.c_str());
        switch (tensor.mode)
        {
        case TensorIoMode::Input:
            input_bytes = checked_add(input_bytes, bytes, tensor.name.c_str());
            break;
        case TensorIoMode::Output:
            output_bytes = checked_add(output_bytes, bytes, tensor.name.c_str());
            break;
        default:
            throw_contract(tensor.name, "tensor I/O mode is unsupported");
        }
    }
    if (input_bytes > options.max_input_bytes)
    {
        throw_resource("tensor metadata", "aggregate input bytes exceed configured limit");
    }
    if (output_bytes > options.max_output_bytes)
    {
        throw_resource("tensor metadata", "aggregate output bytes exceed configured limit");
    }
}

void validate_input_views(const std::vector<TensorDescriptor>& expected,
                          const std::vector<TensorView>& views, std::size_t max_aggregate_bytes)
{
    validate_views_impl(expected, views, TensorIoMode::Input, max_aggregate_bytes,
                        "input view validation");
}

void validate_output_views(const std::vector<TensorDescriptor>&  expected,
                           const std::vector<MutableTensorView>& views,
                           std::size_t                           max_aggregate_bytes)
{
    validate_views_impl(expected, views, TensorIoMode::Output, max_aggregate_bytes,
                        "output view validation");
}

} // namespace kfcore::tensorrt::detail
