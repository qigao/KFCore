#include "kfcore/vision_models/error.hpp"

#include <utility>

namespace kfcore::vision_models
{

VisionModelError::VisionModelError(VisionModelErrorCode code, std::string message)
    : std::runtime_error(std::move(message))
    , code_(code)
{
}

VisionModelErrorCode VisionModelError::code() const noexcept
{
    return code_;
}

} // namespace kfcore::vision_models
