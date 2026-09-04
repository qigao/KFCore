#include "kfcore/image_processor/error.hpp"

#include <string>
#include <utility>

namespace kfcore::image
{

ImageProcessorError::ImageProcessorError(ImageProcessorErrorCode code, std::string message)
    : std::runtime_error(std::move(message))
    , code_(code)
{
}

ImageProcessorErrorCode ImageProcessorError::code() const noexcept
{
    return code_;
}

} // namespace kfcore::image
