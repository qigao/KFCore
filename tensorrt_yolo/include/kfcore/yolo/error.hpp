#pragma once

#include <stdexcept>
#include <string>

namespace kfcore::yolo {

enum class YoloErrorCode {
    InvalidArgument,
    FileIo,
    EngineDeserialize,
    EngineContractMismatch,
    TensorRtFailure,
    CudaFailure,
    TrackerAllocationFailure,
    ResourceLimitExceeded,
};

class YoloError final : public std::runtime_error {
public:
    YoloError(YoloErrorCode code, std::string message);
    YoloErrorCode code() const noexcept;

private:
    YoloErrorCode code_;
};

}  // namespace kfcore::yolo
