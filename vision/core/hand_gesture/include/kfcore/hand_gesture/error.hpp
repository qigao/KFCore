#pragma once

#include <stdexcept>
#include <string>

namespace kfcore::hand_gesture
{

enum class HandGestureErrorCode
{
    InvalidArgument,
    ModelContractMismatch,
    ResourceLimitExceeded,
    RuntimeFailure,
    ConcurrentExecution,
};

class HandGestureError final : public std::runtime_error
{
public:
    HandGestureError(HandGestureErrorCode code, std::string message);

    [[nodiscard]] HandGestureErrorCode code() const noexcept;

private:
    HandGestureErrorCode code_;
};

} // namespace kfcore::hand_gesture
