#pragma once

#include <stdexcept>
#include <string>

namespace kfcore::image
{

enum class ImageProcessorErrorCode
{
    InvalidArgument,
    ResourceLimitExceeded,
    CudaFailure,
};

class ImageProcessorError final : public std::runtime_error
{
public:
    ImageProcessorError(ImageProcessorErrorCode code, std::string message);
    ImageProcessorErrorCode code() const noexcept;

private:
    ImageProcessorErrorCode code_;
};

} // namespace kfcore::image
