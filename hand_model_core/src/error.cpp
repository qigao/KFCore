#include "kfcore/hand_models/error.hpp"

#include <utility>

namespace kfcore::hand_models
{

HandModelError::HandModelError(HandModelErrorCode code, std::string message)
    : std::runtime_error(std::move(message))
    , code_(code)
{
}

HandModelErrorCode HandModelError::code() const noexcept
{
    return code_;
}

} // namespace kfcore::hand_models
