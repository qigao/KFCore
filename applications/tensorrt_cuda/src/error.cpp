#include "kfcore/face_applications/error.hpp"

#include <utility>

namespace kfcore::face_applications
{

FaceApplicationError::FaceApplicationError(FaceApplicationErrorCode code, std::string message)
    : std::runtime_error(std::move(message))
    , code_(code)
{
}

FaceApplicationErrorCode FaceApplicationError::code() const noexcept
{
    return code_;
}

} // namespace kfcore::face_applications
