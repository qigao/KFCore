#include "kfcore/tensorrt/error.hpp"

#include <utility>

namespace kfcore::tensorrt
{

TensorRtError::TensorRtError(TensorRtErrorCode code, std::string message)
    : std::runtime_error(std::move(message))
    , code_(code)
{
}

TensorRtErrorCode TensorRtError::code() const noexcept
{
    return code_;
}

} // namespace kfcore::tensorrt
