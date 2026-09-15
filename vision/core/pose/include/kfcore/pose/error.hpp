#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace kfcore::pose
{

enum class PoseErrorCode
{
    InvalidArgument,
    ModelContractMismatch,
    RuntimeFailure,
    ResourceLimitExceeded,
};

class PoseError final : public std::runtime_error
{
public:
    PoseError(PoseErrorCode code, std::string message)
        : std::runtime_error(std::move(message))
        , code_(code)
    {
    }

    [[nodiscard]] PoseErrorCode code() const noexcept
    {
        return code_;
    }

private:
    PoseErrorCode code_;
};

} // namespace kfcore::pose
