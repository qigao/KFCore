#pragma once

#include <stdexcept>
#include <string>

namespace kfcore::face_models
{

enum class FaceModelErrorCode
{
    InvalidArgument,
    ModelContractMismatch,
    InvalidTensorView,
    ResourceLimitExceeded,
    RuntimeFailure,
    InvalidModelAsset,
};

class FaceModelError final : public std::runtime_error
{
public:
    FaceModelError(FaceModelErrorCode code, std::string message);

    [[nodiscard]] FaceModelErrorCode code() const noexcept;

private:
    FaceModelErrorCode code_;
};

} // namespace kfcore::face_models
