#pragma once

#include "kfcore/sift/error.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace kfcore::sift::detail
{

inline std::size_t descriptor_storage_index(const void* base, std::size_t count,
                                            std::size_t element_size, const void* pointer)
{
    if (count == 0 || element_size == 0 || base == nullptr || pointer == nullptr)
    {
        throw SiftError(SiftErrorCode::BackendFailure,
                        "PopSift result stage: descriptor storage is invalid");
    }
    if (count > (std::numeric_limits<std::size_t>::max)() / element_size)
    {
        throw SiftError(SiftErrorCode::BackendFailure,
                        "PopSift result stage: descriptor storage range overflows");
    }

    const auto begin = reinterpret_cast<std::uintptr_t>(base);
    const auto value = reinterpret_cast<std::uintptr_t>(pointer);
    const std::size_t bytes = count * element_size;
    if (begin > (std::numeric_limits<std::uintptr_t>::max)() - bytes)
    {
        throw SiftError(SiftErrorCode::BackendFailure,
                        "PopSift result stage: descriptor storage range overflows");
    }
    const auto end = begin + bytes;
    if (value < begin || value >= end || (value - begin) % element_size != 0)
    {
        throw SiftError(SiftErrorCode::BackendFailure,
                        "PopSift result stage: descriptor pointer is outside contiguous storage");
    }
    return static_cast<std::size_t>((value - begin) / element_size);
}

} // namespace kfcore::sift::detail
