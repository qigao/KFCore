#pragma once

#include <stdexcept>
#include <string>

namespace kfcore::runtime_onnx
{

enum class ErrorCode
{
    InvalidArgument,
    InvalidModelAsset,
    ModelContractMismatch,
    InvalidTensorView,
    ResourceLimitExceeded,
    RuntimeFailure
};

class Error final : public std::runtime_error
{
public:
    Error(ErrorCode code, std::string message);

    [[nodiscard]] ErrorCode code() const noexcept;

private:
    ErrorCode code_;
};

} // namespace kfcore::runtime_onnx
