#pragma once

#include <stdexcept>
#include <string>

namespace kfcore::vision_models
{

enum class VisionModelErrorCode
{
    InvalidArgument,
    ModelContractMismatch,
    ResourceLimitExceeded,
    RuntimeFailure,
    ConcurrentExecution,
    TrackerFailure,
};

class VisionModelError final : public std::runtime_error
{
public:
    VisionModelError(VisionModelErrorCode code, std::string message);

    [[nodiscard]] VisionModelErrorCode code() const noexcept;

private:
    VisionModelErrorCode code_;
};

} // namespace kfcore::vision_models
