#include "onnx_session.hpp"

#include "kfcore/vision_models/error.hpp"

#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <utility>

namespace kfcore::vision_models::detail
{
namespace
{

[[noreturn]] void throw_contract(const std::string& model, const std::string& detail)
{
    throw VisionModelError(VisionModelErrorCode::ModelContractMismatch,
                           model + " ONNX contract stage: " + detail);
}

[[noreturn]] void throw_runtime(const std::string& model, const std::string& detail)
{
    throw VisionModelError(VisionModelErrorCode::RuntimeFailure,
                           model + " ONNX Runtime stage: " + detail);
}

[[noreturn]] void throw_resource(const std::string& model, const std::string& detail)
{
    throw VisionModelError(VisionModelErrorCode::ResourceLimitExceeded,
                           model + " resource stage: " + detail);
}

ONNXTensorElementDataType native_type(OnnxElementType type)
{
    switch (type)
    {
    case OnnxElementType::Float32:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case OnnxElementType::Int64:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
    throw_contract("ONNX", "unsupported tensor element type");
}

std::size_t element_bytes(OnnxElementType type)
{
    return type == OnnxElementType::Float32 ? sizeof(float) : sizeof(std::int64_t);
}

std::size_t checked_elements(const std::vector<std::int64_t>& dimensions,
                             const std::string& model, const char* tensor,
                             bool allow_zero)
{
    std::size_t result = 1U;
    for (std::int64_t dimension : dimensions)
    {
        if (dimension < 0 || (!allow_zero && dimension == 0))
        {
            throw_contract(model, std::string(tensor) +
                                       " runtime dimensions are invalid");
        }
        if (dimension == 0)
        {
            result = 0U;
            continue;
        }
        const auto value = static_cast<std::uintmax_t>(dimension);
        if (value > static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)()) ||
            static_cast<std::size_t>(value) >
                (std::numeric_limits<std::size_t>::max)() / result)
        {
            throw_resource(model, std::string(tensor) + " element count overflow");
        }
        result *= static_cast<std::size_t>(value);
    }
    return result;
}

void validate_declared_shape(const std::vector<std::int64_t>& actual,
                             const std::vector<std::int64_t>& expected,
                             const std::string& model, const std::string& tensor)
{
    if (actual.size() != expected.size())
    {
        throw_contract(model, tensor + " rank does not match expected contract");
    }
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        if (actual[index] == 0 || actual[index] < -1)
        {
            throw_contract(model, tensor + " has an invalid declared dimension");
        }
        if (expected[index] > 0 && actual[index] != expected[index])
        {
            throw_contract(model, tensor + " dimension " + std::to_string(index) +
                                      " does not match expected contract");
        }
    }
}

void validate_runtime_shape(const std::vector<std::int64_t>& actual,
                            const std::vector<std::int64_t>& expected,
                            const std::string& model, const std::string& tensor,
                            bool allow_zero)
{
    if (actual.size() != expected.size())
    {
        throw_contract(model, tensor + " runtime rank does not match expected contract");
    }
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        if (actual[index] < 0 || (!allow_zero && actual[index] == 0) ||
            (expected[index] > 0 && actual[index] != expected[index]))
        {
            throw_contract(model, tensor + " runtime dimensions do not match expected contract");
        }
    }
}

void validate_model_file(const std::filesystem::path& path, std::size_t limit,
                         const std::string& model)
{
    if (path.empty())
    {
        throw VisionModelError(VisionModelErrorCode::InvalidModelAsset,
                               model + " model path must not be empty");
    }
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
    {
        throw VisionModelError(VisionModelErrorCode::InvalidModelAsset,
                               model + " model is not a readable regular file: " +
                                   path.u8string());
    }
    const std::uintmax_t bytes = std::filesystem::file_size(path, error);
    if (error)
    {
        throw VisionModelError(VisionModelErrorCode::InvalidModelAsset,
                               model + " model size cannot be read: " + path.u8string());
    }
    if (limit == 0U || bytes == 0U || bytes > static_cast<std::uintmax_t>(limit))
    {
        throw_resource(model, "model bytes are zero or exceed configured limit");
    }
}

} // namespace

struct OnnxSession::Impl final
{
    Impl(Ort::Env& environment, const std::filesystem::path& model_path,
         OnnxModelContract expected, const OnnxSessionOptions& options)
        : contract(std::move(expected))
        , max_output_bytes(options.max_output_bytes)
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        validate_model_file(model_path, options.max_model_bytes, contract.model_name);
        if (options.max_output_bytes == 0U)
        {
            throw_resource(contract.model_name, "output byte limit must be positive");
        }
        if (options.intra_op_threads < 0 || options.inter_op_threads < 0)
        {
            throw VisionModelError(VisionModelErrorCode::InvalidArgument,
                                   "ONNX Runtime thread counts must not be negative");
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
        for (std::size_t index = 0; index < contract.inputs.size(); ++index)
        {
            auto name = session->GetInputNameAllocated(index, allocator);
            const OnnxTensorContract& expected = contract.inputs[index];
            if (expected.name != name.get())
            {
                throw_contract(contract.model_name, "unexpected input tensor: " +
                                                        std::string(name.get()));
            }
            const auto info = session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != native_type(expected.element_type))
            {
                throw_contract(contract.model_name, expected.name + " input type is unexpected");
            }
            validate_declared_shape(info.GetShape(), expected.dimensions,
                                    contract.model_name, expected.name);
            input_names.push_back(expected.name.c_str());
        }
        for (std::size_t index = 0; index < contract.outputs.size(); ++index)
        {
            auto name = session->GetOutputNameAllocated(index, allocator);
            const OnnxTensorContract& expected = contract.outputs[index];
            if (expected.name != name.get())
            {
                throw_contract(contract.model_name, "unexpected output tensor: " +
                                                        std::string(name.get()));
            }
            const auto info = session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != native_type(expected.element_type))
            {
                throw_contract(contract.model_name, expected.name + " output type is unexpected");
            }
            validate_declared_shape(info.GetShape(), expected.dimensions,
                                    contract.model_name, expected.name);
            output_names.push_back(expected.name.c_str());
        }
    }

    OnnxModelContract             contract;
    std::size_t                   max_output_bytes;
    Ort::SessionOptions           session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo               memory_info;
    std::vector<const char*>      input_names;
    std::vector<const char*>      output_names;
};

OnnxSession::OnnxSession(Ort::Env& environment, const std::filesystem::path& model_path,
                         OnnxModelContract contract, const OnnxSessionOptions& options)
{
    const std::string model_name = contract.model_name;
    try
    {
        impl_ = std::make_unique<Impl>(environment, model_path, std::move(contract), options);
    }
    catch (const VisionModelError&)
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

OnnxSession::~OnnxSession() = default;
OnnxSession::OnnxSession(OnnxSession&&) noexcept = default;
OnnxSession& OnnxSession::operator=(OnnxSession&&) noexcept = default;

std::vector<OnnxHostTensor> OnnxSession::run(
    const std::vector<OnnxFloatTensorView>& inputs) const
{
    try
    {
        if (inputs.size() != impl_->contract.inputs.size())
        {
            throw_contract(impl_->contract.model_name, "run input count is unexpected");
        }
        std::vector<Ort::Value> input_values;
        input_values.reserve(inputs.size());
        for (std::size_t index = 0; index < inputs.size(); ++index)
        {
            const OnnxFloatTensorView& input = inputs[index];
            const OnnxTensorContract& expected = impl_->contract.inputs[index];
            if (expected.element_type != OnnxElementType::Float32)
            {
                throw_contract(impl_->contract.model_name,
                               "only FP32 inputs are supported by this adapter");
            }
            validate_runtime_shape(input.dimensions, expected.dimensions,
                                   impl_->contract.model_name, expected.name, false);
            const std::size_t count = checked_elements(input.dimensions,
                                                       impl_->contract.model_name,
                                                       expected.name.c_str(), false);
            if (input.data == nullptr || input.element_count != count)
            {
                throw_contract(impl_->contract.model_name,
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
        std::vector<OnnxHostTensor> result;
        result.reserve(outputs.size());
        std::size_t aggregate_bytes = 0U;
        for (std::size_t index = 0; index < outputs.size(); ++index)
        {
            if (!outputs[index].IsTensor())
            {
                throw_contract(impl_->contract.model_name, "output is not a tensor");
            }
            const OnnxTensorContract& expected = impl_->contract.outputs[index];
            const auto info = outputs[index].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != native_type(expected.element_type))
            {
                throw_contract(impl_->contract.model_name,
                               expected.name + " runtime output type is unexpected");
            }
            OnnxHostTensor tensor;
            tensor.element_type = expected.element_type;
            tensor.dimensions   = info.GetShape();
            validate_runtime_shape(tensor.dimensions, expected.dimensions,
                                   impl_->contract.model_name, expected.name, true);
            const std::size_t count = checked_elements(tensor.dimensions,
                                                       impl_->contract.model_name,
                                                       expected.name.c_str(), true);
            const std::size_t scalar_bytes = element_bytes(expected.element_type);
            if (count != 0U && scalar_bytes >
                                   (std::numeric_limits<std::size_t>::max)() / count)
            {
                throw_resource(impl_->contract.model_name,
                               expected.name + " output byte count overflow");
            }
            const std::size_t bytes = count * scalar_bytes;
            if (bytes > impl_->max_output_bytes - aggregate_bytes)
            {
                throw_resource(impl_->contract.model_name,
                               "aggregate output bytes exceed configured limit");
            }
            aggregate_bytes += bytes;
            if (expected.element_type == OnnxElementType::Float32)
            {
                if (count != 0U)
                {
                    const float* data = outputs[index].GetTensorData<float>();
                    tensor.float_values.assign(data, data + count);
                }
            }
            else
            {
                if (count != 0U)
                {
                    const std::int64_t* data = outputs[index].GetTensorData<std::int64_t>();
                    tensor.int64_values.assign(data, data + count);
                }
            }
            result.push_back(std::move(tensor));
        }
        return result;
    }
    catch (const VisionModelError&)
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

} // namespace kfcore::vision_models::detail
