#include "kfcore/sift/error.hpp"
#include "kfcore/sift/popsift_options.hpp"
#include "kfcore/sift/sift_extractor.hpp"

#include <utility>

namespace kfcore::sift
{

SiftError::SiftError(SiftErrorCode code, std::string message)
    : std::runtime_error(std::move(message))
    , code_(code)
{
}

SiftErrorCode SiftError::code() const noexcept
{
    return code_;
}

SiftExtractor::~SiftExtractor() = default;

void PopSiftOptions::validate() const
{
    if (device < 0)
    {
        throw SiftError(SiftErrorCode::InvalidArgument,
                        "PopSift option validation stage: device must not be negative");
    }
    if (max_image_bytes == 0)
    {
        throw SiftError(SiftErrorCode::ResourceLimitExceeded,
                        "PopSift option validation stage: image byte limit must be positive");
    }
    if (max_features == 0)
    {
        throw SiftError(SiftErrorCode::ResourceLimitExceeded,
                        "PopSift option validation stage: feature limit must be positive");
    }
    if (max_features > kMaximumMaxFeatures)
    {
        throw SiftError(SiftErrorCode::ResourceLimitExceeded,
                        "PopSift option validation stage: feature limit exceeds the safe PopSift range");
    }
    if (max_pending_jobs == 0 || max_pending_jobs > kMaximumMaxPendingJobs)
    {
        throw SiftError(SiftErrorCode::ResourceLimitExceeded,
                        "PopSift option validation stage: pending job limit must be within the supported range");
    }
    switch (normalization)
    {
    case PopSiftDescriptorNormalization::Classic:
    case PopSiftDescriptorNormalization::RootSift:
        return;
    }
    throw SiftError(SiftErrorCode::InvalidArgument,
                    "PopSift option validation stage: descriptor normalization is unsupported");
}

} // namespace kfcore::sift
