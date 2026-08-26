#pragma once

#include "kfcore/sift/error.hpp"

#include <cstddef>
#include <cstdint>

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

    std::int32_t device          = 0;
    std::size_t  max_image_bytes = kDefaultMaxImageBytes;
    std::size_t  max_features    = kDefaultMaxFeatures;
    PopSiftDescriptorNormalization normalization = PopSiftDescriptorNormalization::RootSift;

    /** Validate values accepted by the native PopSift adapter. Throws SiftError on failure. */
    void validate() const;
};

} // namespace kfcore::sift
