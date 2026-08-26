#pragma once

#include <stdexcept>
#include <string>

namespace kfcore::sift
{

enum class SiftErrorCode
{
    InvalidArgument,
    ResourceLimitExceeded,
    BackendFailure,
};

class SiftError : public std::runtime_error
{
public:
    SiftError(SiftErrorCode code, std::string message);

    [[nodiscard]] SiftErrorCode code() const noexcept;

private:
    SiftErrorCode code_;
};

} // namespace kfcore::sift
