#pragma once

#include "kfcore/tensorrt/error.hpp"

namespace kfcore::tensorrt::detail
{

enum class TensorRtAliasPolicy
{
    NoAliasFeature,
    RejectUnavailableQuery,
    QueryEngine,
};

constexpr bool tensorrt_version_supported(int major, int minor) noexcept
{
    return (major == 8 && minor == 6) || major == 10 || major == 11;
}

constexpr TensorRtAliasPolicy tensorrt_alias_policy(int major, int minor) noexcept
{
    if (major > 10 || (major == 10 && minor >= 11))
    {
        return TensorRtAliasPolicy::QueryEngine;
    }
    if (major == 10 && minor >= 3)
    {
        return TensorRtAliasPolicy::RejectUnavailableQuery;
    }
    return TensorRtAliasPolicy::NoAliasFeature;
}

constexpr bool tensorrt_supports_alias_query(int major, int minor) noexcept
{
    return tensorrt_alias_policy(major, minor) == TensorRtAliasPolicy::QueryEngine;
}

inline void validate_runtime_tensorrt_version(int major, int minor)
{
    if (!tensorrt_version_supported(major, minor))
    {
        throw TensorRtError(TensorRtErrorCode::EngineContractMismatch,
                            "engine version gate stage: expected TensorRT 8.6, 10.x, or 11.x");
    }
    if (tensorrt_alias_policy(major, minor) ==
        TensorRtAliasPolicy::RejectUnavailableQuery)
    {
        throw TensorRtError(
            TensorRtErrorCode::EngineContractMismatch,
            "engine version gate stage: TensorRT 10.3 through 10.10 can expose aliased "
            "engine I/O but cannot report the alias contract required by this runtime");
    }
}

} // namespace kfcore::tensorrt::detail
