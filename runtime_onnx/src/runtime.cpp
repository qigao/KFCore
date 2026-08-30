#include "kfcore/runtime_onnx/runtime.hpp"

#include <onnxruntime_cxx_api.h>

#include <limits>
#include <new>
#include <system_error>
#include <utility>

namespace kfcore::runtime_onnx
{
namespace
{

[[noreturn]] void throw_contract(const std::string& model, const std::string& detail)
{
    throw Error(ErrorCode::ModelContractMismatch,
                model + " ONNX contract validation stage: " + detail);
}

[[noreturn]] void throw_view(const std::string& model, const std::string& detail)
{
    throw Error(ErrorCode::InvalidTensorView, model + " input validation stage: " + detail);
}

[[noreturn]] void throw_runtime(const std::string& model, const std::string& detail)
{
    throw Error(ErrorCode::RuntimeFailure, model + " ONNX Runtime stage: " + detail);
}

[[noreturn]] void throw_resource(const std::string& model, const std::string& detail)
{
    throw Error(ErrorCode::ResourceLimitExceeded,
                model + " resource validation stage: " + detail);
}

std::size_t checked_elements(const std::vector<std::int64_t>& dimensions,
                             const std::string& model, const std::string& tensor,
                             bool output)
{
    std::size_t count = 1U;
    for (const std::int64_t dimension : dimensions)
    {
        if (dimension <= 0)
        {
            if (output)
            {
                throw_contract(model, tensor + " runtime output dimensions must be positive");
            }
            throw_view(model, tensor + " runtime input dimensions must be positive");
        }
        const auto value = static_cast<std::uintmax_t>(dimension);
        if (value > static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)()) ||
            static_cast<std::size_t>(value) >
                (std::numeric_limits<std::size_t>::max)() / count)
        {
            throw_resource(model, tensor + " element count overflow");
        }
        count *= static_cast<std::size_t>(value);
    }
    return count;
}

void validate_declared_shape(const std::vector<std::int64_t>& actual,
                             const std::vector<std::int64_t>& expected,
                             const std::string& model, const std::string& tensor)
{
    if (actual.size() != expected.size())
    {
        throw_contract(model, tensor + " rank does not match the expected contract");
    }
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        if (actual[index] == 0 || actual[index] < -1 || expected[index] == 0 ||
            expected[index] < -1 ||
            (expected[index] > 0 && actual[index] != expected[index]))
        {
            throw_contract(model, tensor + " declared dimensions do not match the contract");
        }
    }
}

void validate_runtime_shape(const std::vector<std::int64_t>& actual,
                            const std::vector<std::int64_t>& expected,
                            const std::string& model, const std::string& tensor,
                            bool output)
{
    if (actual.size() != expected.size())
    {
        throw_contract(model, tensor + " runtime rank does not match the contract");
    }
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        if (actual[index] <= 0 || (expected[index] > 0 && actual[index] != expected[index]))
        {
            if (output)
            {
                throw_contract(model, tensor + " runtime output dimensions do not match contract");
            }
            throw_view(model, tensor + " runtime input dimensions do not match contract");
        }
    }
}

void validate_model_file(const std::filesystem::path& path, std::size_t limit,
                         const std::string& model)
{
    if (path.empty())
    {
        throw Error(ErrorCode::InvalidModelAsset, model + " model path must not be empty");
    }
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
    {
        throw Error(ErrorCode::InvalidModelAsset,
                    model + " model is not a readable regular file: " + path.u8string());
    }
    const std::uintmax_t bytes = std::filesystem::file_size(path, error);
    if (error)
    {
        throw Error(ErrorCode::InvalidModelAsset,
                    model + " model size cannot be read: " + path.u8string());
    }
    if (limit == 0U || bytes == 0U || bytes > static_cast<std::uintmax_t>(limit))
    {
        throw_resource(model, "model bytes are zero or exceed max_model_bytes");
    }
}

ONNXTensorElementDataType native_type(ElementType type)
{
    switch (type)
    {
    case ElementType::Float32:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case ElementType::Int64:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
    throw Error(ErrorCode::InvalidArgument, "unsupported ONNX element type");
}

std::size_t element_bytes(ElementType type)
{
    switch (type)
    {
    case ElementType::Float32:
        return sizeof(float);
    case ElementType::Int64:
        return sizeof(std::int64_t);
    }
    throw Error(ErrorCode::InvalidArgument, "unsupported ONNX element type");
}

void validate_contract(const ModelContract& contract)
{
    if (contract.model_name.empty() || contract.inputs.empty() || contract.outputs.empty())
    {
        throw Error(ErrorCode::InvalidArgument,
                    "ONNX model contract requires a name, inputs, and outputs");
    }
    for (const TensorContract& input : contract.inputs)
    {
        if (input.name.empty() || input.dimensions.empty())
        {
            throw Error(ErrorCode::InvalidArgument,
                        contract.model_name + " input contract is incomplete");
        }
        if (input.element_type != ElementType::Float32)
        {
            throw Error(ErrorCode::InvalidArgument,
                        contract.model_name + " runtime accepts FP32 inputs only");
        }
    }
    for (const TensorContract& output : contract.outputs)
    {
        if (output.name.empty() || output.dimensions.empty())
        {
            throw Error(ErrorCode::InvalidArgument,
                        contract.model_name + " output contract is incomplete");
        }
    }
}

} // namespace

struct Environment::Impl final
{
    explicit Impl(const char* log_id) : value(ORT_LOGGING_LEVEL_ERROR, log_id) {}

    Ort::Env value;
};

Environment::Environment(const char* log_id)
{
    if (log_id == nullptr || *log_id == '\0')
    {
        throw Error(ErrorCode::InvalidArgument, "ONNX Runtime log id must not be empty");
    }
    try
    {
        impl_ = std::make_unique<Impl>(log_id);
    }
    catch (const Ort::Exception& error)
    {
        throw_runtime("environment", error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("environment", "allocation failed");
    }
}

Environment::~Environment() = default;

struct Session::Impl final
{
    Impl(Ort::Env& environment, const std::filesystem::path& model_path,
         ModelContract expected, const SessionOptions& options)
        : contract(std::move(expected))
        , max_output_bytes(options.max_output_bytes)
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        validate_contract(contract);
        validate_model_file(model_path, options.max_model_bytes, contract.model_name);
        if (options.intra_op_threads < 0 || options.inter_op_threads < 0)
        {
            throw Error(ErrorCode::InvalidArgument,
                        "ONNX Runtime thread counts must not be negative");
        }
        if (options.max_output_bytes == 0U)
        {
            throw_resource(contract.model_name, "max_output_bytes must be positive");
        }
        if (options.intra_op_threads > 0)
        {
            session_options.SetIntraOpNumThreads(options.intra_op_threads);
        }
        if (options.inter_op_threads > 0)
        {
            session_options.SetInterOpNumThreads(options.inter_op_threads);
        }
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        session = std::make_unique<Ort::Session>(environment, model_path.c_str(), session_options);
#else
        session = std::make_unique<Ort::Session>(environment, model_path.string().c_str(),
                                                 session_options);
#endif
        validate_metadata();
    }

    void validate_metadata()
    {
        if (session->GetInputCount() != contract.inputs.size() ||
            session->GetOutputCount() != contract.outputs.size())
        {
            throw_contract(contract.model_name, "input or output tensor count is unexpected");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        input_names.reserve(contract.inputs.size());
        output_names.reserve(contract.outputs.size());
        declared_input_shapes.reserve(contract.inputs.size());
        declared_output_shapes.reserve(contract.outputs.size());
        for (std::size_t index = 0; index < contract.inputs.size(); ++index)
        {
            auto name = session->GetInputNameAllocated(index, allocator);
            const TensorContract& expected = contract.inputs[index];
            if (expected.name != name.get())
            {
                throw_contract(contract.model_name,
                               "unexpected input tensor: " + std::string(name.get()));
            }
            const auto info = session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != native_type(expected.element_type))
            {
                throw_contract(contract.model_name, expected.name + " input type is unexpected");
            }
            std::vector<std::int64_t> declared_shape = info.GetShape();
            validate_declared_shape(declared_shape, expected.dimensions,
                                    contract.model_name, expected.name);
            declared_input_shapes.push_back(std::move(declared_shape));
            input_names.push_back(expected.name.c_str());
        }
        for (std::size_t index = 0; index < contract.outputs.size(); ++index)
        {
            auto name = session->GetOutputNameAllocated(index, allocator);
            const TensorContract& expected = contract.outputs[index];
            if (expected.name != name.get())
            {
                throw_contract(contract.model_name,
                               "unexpected output tensor: " + std::string(name.get()));
            }
            const auto info = session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != native_type(expected.element_type))
            {
                throw_contract(contract.model_name, expected.name + " output type is unexpected");
            }
            std::vector<std::int64_t> declared_shape = info.GetShape();
            validate_declared_shape(declared_shape, expected.dimensions,
                                    contract.model_name, expected.name);
            declared_output_shapes.push_back(std::move(declared_shape));
            output_names.push_back(expected.name.c_str());
        }
    }

    ModelContract                 contract;
    std::size_t                   max_output_bytes;
    Ort::SessionOptions           session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo               memory_info;
    std::vector<const char*>      input_names;
    std::vector<const char*>      output_names;
    std::vector<std::vector<std::int64_t>> declared_input_shapes;
    std::vector<std::vector<std::int64_t>> declared_output_shapes;
};

Session::Session(Environment& environment, const std::filesystem::path& model_path,
                 ModelContract contract, const SessionOptions& options)
{
    if (!environment.impl_)
    {
        throw Error(ErrorCode::InvalidArgument, "ONNX Runtime environment has been moved from");
    }
    const std::string model_name = contract.model_name;
    try
    {
        impl_ = std::make_unique<Impl>(environment.impl_->value, model_path,
                                       std::move(contract), options);
    }
    catch (const Error&)
    {
        throw;
    }
    catch (const Ort::Exception& error)
    {
        throw_runtime(model_name, error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource(model_name, "session allocation failed");
    }
}

Session::~Session() = default;
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;

struct OwnedSession::Impl final
{
    Impl(const std::string& log_id, const std::filesystem::path& model_path,
         ModelContract contract, const SessionOptions& options)
        : environment(log_id.c_str())
        , session(environment, model_path, std::move(contract), options)
    {
    }

    Environment environment;
    Session     session;
};

OwnedSession::OwnedSession(std::string log_id,
                           const std::filesystem::path& model_path,
                           ModelContract contract, const SessionOptions& options)
    : impl_(std::make_unique<Impl>(log_id, model_path, std::move(contract), options))
{
}

OwnedSession::~OwnedSession() = default;
OwnedSession::OwnedSession(OwnedSession&&) noexcept = default;
OwnedSession& OwnedSession::operator=(OwnedSession&&) noexcept = default;

Session& OwnedSession::session() noexcept
{
    return impl_->session;
}

const Session& OwnedSession::session() const noexcept
{
    return impl_->session;
}

const std::vector<std::int64_t>& Session::declared_input_dimensions(std::size_t index) const
{
    if (!impl_ || index >= impl_->declared_input_shapes.size())
    {
        throw Error(ErrorCode::InvalidArgument,
                    "ONNX Runtime input metadata index is out of range");
    }
    return impl_->declared_input_shapes[index];
}

const std::vector<std::int64_t>& Session::declared_output_dimensions(std::size_t index) const
{
    if (!impl_ || index >= impl_->declared_output_shapes.size())
    {
        throw Error(ErrorCode::InvalidArgument,
                    "ONNX Runtime output metadata index is out of range");
    }
    return impl_->declared_output_shapes[index];
}

std::vector<HostTensor> Session::run(const std::vector<FloatTensorView>& inputs) const
{
    if (!impl_)
    {
        throw Error(ErrorCode::InvalidArgument, "ONNX Runtime session has been moved from");
    }
    try
    {
        if (inputs.size() != impl_->contract.inputs.size())
        {
            throw_view(impl_->contract.model_name, "run input count is unexpected");
        }
        std::vector<Ort::Value> input_values;
        input_values.reserve(inputs.size());
        for (std::size_t index = 0; index < inputs.size(); ++index)
        {
            const FloatTensorView& input = inputs[index];
            const TensorContract& expected = impl_->contract.inputs[index];
            validate_runtime_shape(input.dimensions, expected.dimensions,
                                   impl_->contract.model_name, expected.name, false);
            const std::size_t count = checked_elements(input.dimensions,
                                                       impl_->contract.model_name,
                                                       expected.name, false);
            if (input.data == nullptr || input.element_count != count)
            {
                throw_view(impl_->contract.model_name,
                           expected.name + " input storage does not match shape");
            }
            input_values.emplace_back(Ort::Value::CreateTensor<float>(
                impl_->memory_info, const_cast<float*>(input.data), input.element_count,
                input.dimensions.data(), input.dimensions.size()));
        }

        Ort::RunOptions run_options;
        std::vector<Ort::Value> outputs = impl_->session->Run(
            run_options, impl_->input_names.data(), input_values.data(), input_values.size(),
            impl_->output_names.data(), impl_->output_names.size());
        std::vector<HostTensor> result;
        result.reserve(outputs.size());
        std::size_t aggregate_bytes = 0U;
        for (std::size_t index = 0; index < outputs.size(); ++index)
        {
            if (!outputs[index].IsTensor())
            {
                throw_contract(impl_->contract.model_name, "output is not a tensor");
            }
            const TensorContract& expected = impl_->contract.outputs[index];
            const auto info = outputs[index].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != native_type(expected.element_type))
            {
                throw_contract(impl_->contract.model_name,
                               expected.name + " runtime output type is unexpected");
            }
            HostTensor tensor;
            tensor.element_type = expected.element_type;
            tensor.dimensions   = info.GetShape();
            validate_runtime_shape(tensor.dimensions, expected.dimensions,
                                   impl_->contract.model_name, expected.name, true);
            const std::size_t count = checked_elements(tensor.dimensions,
                                                       impl_->contract.model_name,
                                                       expected.name, true);
            const std::size_t scalar_bytes = element_bytes(expected.element_type);
            if (count != 0U && scalar_bytes >
                                   (std::numeric_limits<std::size_t>::max)() / count)
            {
                throw_resource(impl_->contract.model_name,
                               expected.name + " output byte count overflow");
            }
            const std::size_t bytes = count * scalar_bytes;
            if (aggregate_bytes > impl_->max_output_bytes ||
                bytes > impl_->max_output_bytes - aggregate_bytes)
            {
                throw_resource(impl_->contract.model_name,
                               "aggregate output bytes exceed configured limit");
            }
            aggregate_bytes += bytes;
            if (expected.element_type == ElementType::Float32)
            {
                const float* data = outputs[index].GetTensorData<float>();
                tensor.float_values.assign(data, data + count);
            }
            else
            {
                const std::int64_t* data = outputs[index].GetTensorData<std::int64_t>();
                tensor.int64_values.assign(data, data + count);
            }
            result.push_back(std::move(tensor));
        }
        return result;
    }
    catch (const Error&)
    {
        throw;
    }
    catch (const Ort::Exception& error)
    {
        throw_runtime(impl_->contract.model_name, error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource(impl_->contract.model_name, "run output allocation failed");
    }
}

} // namespace kfcore::runtime_onnx
