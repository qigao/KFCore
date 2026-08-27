#pragma once

namespace kfcore::tensorrt::detail
{

constexpr bool tensorrt_supports_alias_query(int major, int minor) noexcept
{
    return major > 10 || (major == 10 && minor >= 11);
}

} // namespace kfcore::tensorrt::detail
