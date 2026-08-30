#include "support.hpp"

#include <new>
#include <utility>

namespace kfcore::face_models::detail
{
namespace
{

FaceModelErrorCode map_error_code(runtime_onnx::ErrorCode code) noexcept
{
    switch (code)
    {
    case runtime_onnx::ErrorCode::InvalidArgument:
        return FaceModelErrorCode::InvalidArgument;
    case runtime_onnx::ErrorCode::InvalidModelAsset:
        return FaceModelErrorCode::InvalidModelAsset;
    case runtime_onnx::ErrorCode::ModelContractMismatch:
        return FaceModelErrorCode::ModelContractMismatch;
    case runtime_onnx::ErrorCode::InvalidTensorView:
        return FaceModelErrorCode::InvalidTensorView;
    case runtime_onnx::ErrorCode::ResourceLimitExceeded:
        return FaceModelErrorCode::ResourceLimitExceeded;
    case runtime_onnx::ErrorCode::RuntimeFailure:
        return FaceModelErrorCode::RuntimeFailure;
    }
    return FaceModelErrorCode::RuntimeFailure;
}

[[noreturn]] void throw_runtime_error(const runtime_onnx::Error& error)
{
    throw FaceModelError(map_error_code(error.code()), error.what());
}

runtime_onnx::SessionOptions runtime_options(const CpuFaceModelOptions& options)
{
    runtime_onnx::SessionOptions result;
    result.intra_op_threads = options.intra_op_threads;
    result.inter_op_threads = options.inter_op_threads;
    result.max_model_bytes  = options.max_model_bytes;
    result.max_output_bytes = options.max_output_bytes;
    return result;
}

} // namespace

runtime_onnx::TensorContract fp32_contract(
    std::string name, std::vector<std::int64_t> dimensions)
{
    return { std::move(name), runtime_onnx::ElementType::Float32,
             std::move(dimensions) };
}

runtime_onnx::OwnedSession make_session(
    const char* model_name, const std::filesystem::path& model_path,
    CpuModelContract contract, const CpuFaceModelOptions& options)
{
    try
    {
        return runtime_onnx::OwnedSession(
            std::string("KFCoreFaceModelsCpu") + model_name, model_path,
            std::move(contract), runtime_options(options));
    }
    catch (const runtime_onnx::Error& error)
    {
        throw_runtime_error(error);
    }
}

std::vector<CpuSessionOutput> run(
    const runtime_onnx::OwnedSession& session,
    const std::vector<runtime_onnx::FloatTensorView>& inputs)
{
    try
    {
        return session.session().run(inputs);
    }
    catch (const runtime_onnx::Error& error)
    {
        throw_runtime_error(error);
    }
}

} // namespace kfcore::face_models::detail
