#include "runtime_support.hpp"

#include "kfcore/hand_models/error.hpp"

#include <utility>

namespace kfcore::hand_models::detail
{
namespace
{

HandModelErrorCode map_error_code(runtime_onnx::ErrorCode code) noexcept
{
    switch (code)
    {
    case runtime_onnx::ErrorCode::InvalidArgument:
        return HandModelErrorCode::InvalidArgument;
    case runtime_onnx::ErrorCode::InvalidModelAsset:
        return HandModelErrorCode::InvalidModelAsset;
    case runtime_onnx::ErrorCode::ModelContractMismatch:
    case runtime_onnx::ErrorCode::InvalidTensorView:
        return HandModelErrorCode::ModelContractMismatch;
    case runtime_onnx::ErrorCode::ResourceLimitExceeded:
        return HandModelErrorCode::ResourceLimitExceeded;
    case runtime_onnx::ErrorCode::RuntimeFailure:
        return HandModelErrorCode::RuntimeFailure;
    }
    return HandModelErrorCode::RuntimeFailure;
}

[[noreturn]] void throw_runtime_error(const runtime_onnx::Error& error)
{
    throw HandModelError(map_error_code(error.code()), error.what());
}

} // namespace

runtime_onnx::Session make_session(
    runtime_onnx::Environment& environment,
    const std::filesystem::path& model_path,
    OnnxModelContract contract, const OnnxSessionOptions& options)
{
    try
    {
        return runtime_onnx::Session(environment, model_path, std::move(contract), options);
    }
    catch (const runtime_onnx::Error& error)
    {
        throw_runtime_error(error);
    }
}

std::vector<OnnxHostTensor> run(
    const runtime_onnx::Session& session,
    const std::vector<OnnxFloatTensorView>& inputs)
{
    try
    {
        return session.run(inputs);
    }
    catch (const runtime_onnx::Error& error)
    {
        throw_runtime_error(error);
    }
}

} // namespace kfcore::hand_models::detail
