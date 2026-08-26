#pragma once

#include "kfcore/sift/error.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace kfcore::sift
{

enum class PopSiftDescriptorNormalization
{
    Classic,
    RootSift,
};

struct PopSiftOptions
{
    static constexpr std::size_t kDefaultMaxImageBytes = 64U * 1024U * 1024U;
    static constexpr std::size_t kDefaultMaxFeatures   = 100000U;
    static constexpr std::size_t kDefaultMaxPendingJobs = 8U;
    static constexpr std::size_t kMaximumMaxPendingJobs = 1024U;
    // PopSift performs signed-int expansions of this value, including a 1.1x filter bound.
    static constexpr std::size_t kMaximumMaxFeatures =
        static_cast<std::size_t>((std::numeric_limits<int>::max)()) / 2U;

    std::int32_t device          = 0;
    std::size_t  max_image_bytes = kDefaultMaxImageBytes;
    std::size_t  max_features    = kDefaultMaxFeatures;
    std::size_t  max_pending_jobs = kDefaultMaxPendingJobs;
    PopSiftDescriptorNormalization normalization = PopSiftDescriptorNormalization::RootSift;

    /** Validate values accepted by the native PopSift adapter. Throws SiftError on failure. */
    void validate() const;
};

} // namespace kfcore::sift
