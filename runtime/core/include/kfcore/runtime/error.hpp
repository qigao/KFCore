#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace kfcore::runtime
{

enum class RuntimeErrorCode
{
    InvalidArgument,
    FileIo,
    ModuleLoad,
    SymbolLookup,
    AbiMismatch,
    BackendFailure,
    DuplicateBackend,
    NotFound,
};

class RuntimeError final : public std::runtime_error
{
public:
    RuntimeError(RuntimeErrorCode code, std::string message)
        : std::runtime_error(std::move(message))
        , code_(code)
    {
    }

    [[nodiscard]] RuntimeErrorCode code() const noexcept
    {
        return code_;
    }

private:
    RuntimeErrorCode code_;
};

} // namespace kfcore::runtime
