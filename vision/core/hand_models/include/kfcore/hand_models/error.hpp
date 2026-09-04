#pragma once

#include <stdexcept>
#include <string>

namespace kfcore::hand_models
{

enum class HandModelErrorCode
{
    InvalidArgument,
    InvalidModelAsset,
    ModelContractMismatch,
    ResourceLimitExceeded,
    RuntimeFailure,
    ConcurrentExecution,
    TrackerFailure,
};

class HandModelError final : public std::runtime_error
{
public:
    HandModelError(HandModelErrorCode code, std::string message);

    [[nodiscard]] HandModelErrorCode code() const noexcept;

private:
    HandModelErrorCode code_;
};

} // namespace kfcore::hand_models
