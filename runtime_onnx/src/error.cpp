#include "kfcore/runtime_onnx/error.hpp"

#include <utility>

namespace kfcore::runtime_onnx
{

Error::Error(ErrorCode code, std::string message)
    : std::runtime_error(std::move(message)), code_(code)
{
}

ErrorCode Error::code() const noexcept
{
    return code_;
}

} // namespace kfcore::runtime_onnx
