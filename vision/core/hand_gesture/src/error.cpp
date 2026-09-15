#include "kfcore/hand_gesture/error.hpp"

#include <utility>

namespace kfcore::hand_gesture
{

HandGestureError::HandGestureError(HandGestureErrorCode code, std::string message)
    : std::runtime_error(std::move(message))
    , code_(code)
{
}

HandGestureErrorCode HandGestureError::code() const noexcept
{
    return code_;
}

} // namespace kfcore::hand_gesture
