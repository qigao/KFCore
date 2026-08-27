#pragma once

#include <stdexcept>
#include <string>

namespace kfcore::face_applications
{

enum class FaceApplicationErrorCode
{
    InvalidArgument,
    NoFaceDetected,
    InvalidLandmarks,
    InvalidModelMatrix,
    ModelContractMismatch,
    ResourceLimitExceeded,
    RuntimeFailure,
    ImageProcessingFailure,
};

class FaceApplicationError final : public std::runtime_error
{
public:
    FaceApplicationError(FaceApplicationErrorCode code, std::string message);

    [[nodiscard]] FaceApplicationErrorCode code() const noexcept;

private:
    FaceApplicationErrorCode code_;
};

} // namespace kfcore::face_applications
