#include "onnx_session.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace kfcore::face_applications::detail
{
namespace
{

[[noreturn]] void throw_contract(const std::string& model, const std::string& detail)
{
    throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::ModelContractMismatch,
                                  model + " ONNX contract validation stage: " + detail);
}

[[noreturn]] void throw_runtime(const std::string& model, const std::string& detail)
{
    throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::RuntimeFailure,
                                  model + " ONNX Runtime stage: " + detail);
}

[[noreturn]] void throw_resource(const std::string& model, const std::string& detail)
{
    throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::ResourceLimitExceeded,
                                  model + " resource validation stage: " + detail);
}

std::size_t checked_elements(const std::vector<std::int64_t>& dimensions,
                             const std::string& model, const char* subject)
{
    std::size_t result = 1U;
    for (std::int64_t dimension : dimensions)
    {
        if (dimension <= 0)
        {
            throw_contract(model, std::string(subject) + " dimensions must be positive at run time");
        }
        const auto value = static_cast<std::uintmax_t>(dimension);
        if (value > static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)()) ||
            static_cast<std::size_t>(value) >
                (std::numeric_limits<std::size_t>::max)() / result)
        {
            throw_resource(model, std::string(subject) + " element count overflow");
        }
        result *= static_cast<std::size_t>(value);
    }
    return result;
}

void validate_shape(const std::vector<std::int64_t>& actual,
                    const std::vector<std::int64_t>& expected,
                    const std::string& model, const std::string& tensor)
{
    if (actual.size() != expected.size())
    {
        throw_contract(model, tensor + " rank does not match the expected contract");
    }
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        if (expected[index] > 0 && actual[index] != expected[index])
        {
            throw_contract(model, tensor + " dimension " + std::to_string(index) +
                                      " does not match the expected contract");
        }
        if (actual[index] == 0 || actual[index] < -1)
        {
            throw_contract(model, tensor + " has an invalid dimension declaration");
        }
    }
}

void validate_model_file(const std::filesystem::path& path, std::size_t limit,
                         const std::string& model)
{
    std::error_code error;
    if (path.empty())
    {
        throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::InvalidModelAsset,
                                      model + " model path must not be empty");
    }
    if (!std::filesystem::is_regular_file(path, error) || error)
    {
        throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::InvalidModelAsset,
                                      model + " model is not a readable regular file: " +
                                          path.u8string());
    }
    const std::uintmax_t bytes = std::filesystem::file_size(path, error);
    if (error)
    {
        throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::InvalidModelAsset,
                                      model + " model size cannot be read: " + path.u8string());
    }
    if (limit == 0U || bytes == 0U || bytes > static_cast<std::uintmax_t>(limit))
    {
        throw_resource(model, "model bytes are zero or exceed the configured limit");
    }
}

} // namespace

struct OnnxSession::Impl final
{
    Impl(Ort::Env& environment, const std::filesystem::path& model_path,
         const ModelContract& expected, const CpuFaceSwapOptions& options)
        : contract(expected)
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        validate_model_file(model_path, options.max_model_bytes, contract.model_name);
        if (options.intra_op_threads < 0 || options.inter_op_threads < 0)
        {
            throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::InvalidArgument,
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
            if (contract.inputs[index].name != name.get())
            {
                throw_contract(contract.model_name, "input tensor name is unexpected: " +
                                                        std::string(name.get()));
            }
            const auto info = session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                throw_contract(contract.model_name, contract.inputs[index].name +
                                                        " must use FP32 elements");
            }
            validate_shape(info.GetShape(), contract.inputs[index].dimensions,
                           contract.model_name, contract.inputs[index].name);
            input_names.push_back(contract.inputs[index].name.c_str());
        }
        for (std::size_t index = 0; index < contract.outputs.size(); ++index)
        {
            auto name = session->GetOutputNameAllocated(index, allocator);
            if (contract.outputs[index].name != name.get())
            {
                throw_contract(contract.model_name, "output tensor name is unexpected: " +
                                                        std::string(name.get()));
            }
            const auto info = session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                throw_contract(contract.model_name, contract.outputs[index].name +
                                                        " must use FP32 elements");
            }
            validate_shape(info.GetShape(), contract.outputs[index].dimensions,
                           contract.model_name, contract.outputs[index].name);
            output_names.push_back(contract.outputs[index].name.c_str());
        }
    }

    ModelContract                    contract;
    Ort::SessionOptions              session_options;
    std::unique_ptr<Ort::Session>    session;
    Ort::MemoryInfo                  memory_info;
    std::vector<const char*>         input_names;
    std::vector<const char*>         output_names;
};

OnnxSession::OnnxSession(Ort::Env& environment, const std::filesystem::path& model_path,
                         const ModelContract& contract, const CpuFaceSwapOptions& options)
{
    try
    {
        impl_ = std::make_unique<Impl>(environment, model_path, contract, options);
    }
    catch (const CpuFaceApplicationError&)
    {
        throw;
    }
    catch (const Ort::Exception& error)
    {
        throw_runtime(contract.model_name, error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource(contract.model_name, "session allocation failed");
    }
}

OnnxSession::~OnnxSession() = default;
OnnxSession::OnnxSession(OnnxSession&&) noexcept = default;
OnnxSession& OnnxSession::operator=(OnnxSession&&) noexcept = default;

std::vector<HostTensor> OnnxSession::run(const std::vector<HostTensorView>& inputs) const
{
    try
    {
        if (inputs.size() != impl_->contract.inputs.size())
        {
            throw_contract(impl_->contract.model_name, "run input count is unexpected");
        }
        std::vector<Ort::Value> values;
        values.reserve(inputs.size());
        for (std::size_t index = 0; index < inputs.size(); ++index)
        {
            const HostTensorView& input = inputs[index];
            validate_shape(input.dimensions, impl_->contract.inputs[index].dimensions,
                           impl_->contract.model_name, impl_->contract.inputs[index].name);
            const std::size_t expected = checked_elements(
                input.dimensions, impl_->contract.model_name,
                impl_->contract.inputs[index].name.c_str());
            if (input.data == nullptr || input.element_count != expected)
            {
                throw_contract(impl_->contract.model_name,
                               impl_->contract.inputs[index].name +
                                   " storage does not match its run-time shape");
            }
            values.emplace_back(Ort::Value::CreateTensor<float>(
                impl_->memory_info, const_cast<float*>(input.data), input.element_count,
                input.dimensions.data(), input.dimensions.size()));
        }
        Ort::RunOptions run_options;
        std::vector<Ort::Value> outputs = impl_->session->Run(
            run_options, impl_->input_names.data(), values.data(), values.size(),
            impl_->output_names.data(), impl_->output_names.size());
        std::vector<HostTensor> result;
        result.reserve(outputs.size());
        for (std::size_t index = 0; index < outputs.size(); ++index)
        {
            if (!outputs[index].IsTensor())
            {
                throw_contract(impl_->contract.model_name, "output is not a tensor");
            }
            auto info = outputs[index].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                throw_contract(impl_->contract.model_name, "output does not use FP32 elements");
            }
            HostTensor tensor;
            tensor.dimensions = info.GetShape();
            validate_shape(tensor.dimensions, impl_->contract.outputs[index].dimensions,
                           impl_->contract.model_name, impl_->contract.outputs[index].name);
            const std::size_t count = checked_elements(tensor.dimensions,
                                                       impl_->contract.model_name, "output");
            const float* data = outputs[index].GetTensorData<float>();
            tensor.values.assign(data, data + count);
            result.push_back(std::move(tensor));
        }
        return result;
    }
    catch (const CpuFaceApplicationError&)
    {
        throw;
    }
    catch (const Ort::Exception& error)
    {
        throw_runtime(impl_->contract.model_name, error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource(impl_->contract.model_name, "output allocation failed");
    }
}

} // namespace kfcore::face_applications::detail
