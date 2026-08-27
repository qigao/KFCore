#pragma once

#include <stdexcept>
#include <string>

namespace kfcore::tensorrt
{

enum class TensorRtErrorCode
{
    InvalidArgument,
    FileIo,
    EngineDeserialize,
    EngineContractMismatch,
    InvalidTensorView,
    TensorRtFailure,
    CudaFailure,
    ResourceLimitExceeded,
    ConcurrentExecution,
};

class TensorRtError final : public std::runtime_error
{
public:
    TensorRtError(TensorRtErrorCode code, std::string message);

    TensorRtErrorCode code() const noexcept;

private:
    TensorRtErrorCode code_;
};

} // namespace kfcore::tensorrt
