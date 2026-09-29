#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace kfcore::relation
{

enum class RelationErrorCode
{
    InvalidArgument,
    ModelContractMismatch,
    RuntimeFailure,
    ResourceLimitExceeded,
};

class RelationError final : public std::runtime_error
{
public:
    RelationError(RelationErrorCode code, std::string message)
        : std::runtime_error(std::move(message))
        , code_(code)
    {
    }

    [[nodiscard]] RelationErrorCode code() const noexcept
    {
        return code_;
    }

private:
    RelationErrorCode code_;
};

} // namespace kfcore::relation
