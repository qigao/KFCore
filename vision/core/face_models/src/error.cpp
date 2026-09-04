#include "kfcore/face_models/error.hpp"

#include <utility>

namespace kfcore::face_models
{

FaceModelError::FaceModelError(FaceModelErrorCode code, std::string message)
    : std::runtime_error(std::move(message))
    , code_(code)
{
}

FaceModelErrorCode FaceModelError::code() const noexcept
{
    return code_;
}

} // namespace kfcore::face_models
