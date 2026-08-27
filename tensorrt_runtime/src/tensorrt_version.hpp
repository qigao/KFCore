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

inline void validate_tensorrt_runtime_version(int major, int minor)
{
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
