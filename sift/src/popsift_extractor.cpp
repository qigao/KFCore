#include "kfcore/sift/popsift_extractor.hpp"

#include "kfcore/image_processor/error.hpp"
#include "kfcore/image_processor/image_processor.hpp"
#include "kfcore/sift/error.hpp"

#include <popsift/features.h>
#include <popsift/popsift.h>
#include <popsift/sift_conf.h>
#include <popsift/sift_extremum.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::sift
{
namespace
{

[[noreturn]] void throw_resource(std::string message)
{
    throw SiftError(SiftErrorCode::ResourceLimitExceeded, std::move(message));
}

[[noreturn]] void throw_backend(std::string message)
{
    throw SiftError(SiftErrorCode::BackendFailure, std::move(message));
}

popsift::Config make_config(const PopSiftOptions& options)
{
    popsift::Config config;
    config.setFilterMaxExtrema(static_cast<int>(options.max_features));
    switch (options.normalization)
    {
    case PopSiftDescriptorNormalization::Classic:
        config.setNormMode(popsift::Config::Classic);
        break;
    case PopSiftDescriptorNormalization::RootSift:
        config.setNormMode(popsift::Config::RootSift);
        break;
    }
    return config;
}

SiftError map_image_error(const image::ImageProcessorError& error)
{
    SiftErrorCode code = SiftErrorCode::BackendFailure;
    switch (error.code())
    {
    case image::ImageProcessorErrorCode::InvalidArgument:
        code = SiftErrorCode::InvalidArgument;
        break;
    case image::ImageProcessorErrorCode::ResourceLimitExceeded:
        code = SiftErrorCode::ResourceLimitExceeded;
        break;
    case image::ImageProcessorErrorCode::CudaFailure:
        code = SiftErrorCode::BackendFailure;
        break;
    }
    return SiftError(code, std::string("PopSift input preparation failed: ") + error.what());
}

} // namespace

class PopSiftExtractor::Impl
{
public:
    explicit Impl(PopSiftOptions requested_options)
        : options(std::move(requested_options))
        , backend(make_config(options), popsift::Config::ExtractingMode, PopSift::ByteImages,
                  options.device)
    {
    }

    PopSiftOptions options;
    PopSift        backend;
    std::mutex     mutex;
};

PopSiftExtractor::PopSiftExtractor(PopSiftOptions options)
{
    options.validate();
    try
    {
        impl_ = std::make_unique<Impl>(std::move(options));
    }
    catch (const SiftError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("PopSift initialization stage: allocation failed");
    }
    catch (const std::exception& error)
    {
        throw_backend(std::string("PopSift initialization stage: ") + error.what());
    }
}

PopSiftExtractor::~PopSiftExtractor() = default;

FeatureSet PopSiftExtractor::extract(const image::ImageView& image)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);

    try
    {
        const std::size_t grayscale_bytes = image::ImageProcessor::packed_grayscale_bytes(
            image, impl_->options.max_image_bytes);
        if (grayscale_bytes > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        {
            throw_resource(
                "PopSift input preparation stage: image exceeds PopSift int byte range");
        }
        std::vector<std::uint8_t> grayscale(grayscale_bytes);
        image::ImageProcessor::stage_host_grayscale(
            image, { grayscale.data(), grayscale.size() }, impl_->options.max_image_bytes);

        std::unique_ptr<SiftJob> job(
            impl_->backend.enqueue(image.width, image.height, grayscale.data()));
        if (!job)
        {
            throw_backend("PopSift extraction stage: backend rejected the image dimensions");
        }

        std::unique_ptr<popsift::FeaturesHost> native_features(job->getHost());
        if (!native_features)
        {
            throw_backend("PopSift extraction stage: backend returned no host result");
        }

        const int feature_count    = native_features->getFeatureCount();
        const int descriptor_count = native_features->getDescriptorCount();
        if (feature_count < 0 || descriptor_count < 0)
        {
            throw_backend("PopSift result stage: backend returned negative counts");
        }
        if (static_cast<std::size_t>(descriptor_count) > impl_->options.max_features)
        {
            throw_resource(
                "PopSift result stage: descriptor count exceeds configured feature limit");
        }
        if (static_cast<std::size_t>(feature_count) > impl_->options.max_features)
        {
            throw_resource(
                "PopSift result stage: extrema count exceeds configured feature limit");
        }

        FeatureSet result;
        result.features.reserve(static_cast<std::size_t>(descriptor_count));
        const popsift::Feature* native_data = native_features->getFeatures();
        if (feature_count > 0 && native_data == nullptr)
        {
            throw_backend("PopSift result stage: feature storage is null");
        }

        for (int feature_index = 0; feature_index < feature_count; ++feature_index)
        {
            const popsift::Feature& native_feature = native_data[feature_index];
            if (native_feature.num_ori < 0 || native_feature.num_ori > ORIENTATION_MAX_COUNT)
            {
                throw_backend("PopSift result stage: orientation count is invalid");
            }
            for (int orientation_index = 0; orientation_index < native_feature.num_ori;
                 ++orientation_index)
            {
                const popsift::Descriptor* native_descriptor =
                    native_feature.desc[orientation_index];
                if (native_descriptor == nullptr)
                {
                    throw_backend("PopSift result stage: descriptor storage is null");
                }
                if (result.features.size() >= static_cast<std::size_t>(descriptor_count))
                {
                    throw_backend("PopSift result stage: descriptor count is inconsistent");
                }

                Feature feature;
                feature.x           = native_feature.xpos;
                feature.y           = native_feature.ypos;
                feature.scale       = native_feature.sigma;
                feature.orientation_radians = native_feature.orientation[orientation_index];
                feature.octave = native_feature.debug_octave;
                if (!std::isfinite(feature.x) || !std::isfinite(feature.y) ||
                    !std::isfinite(feature.scale) || feature.scale <= 0.0f ||
                    !std::isfinite(feature.orientation_radians))
                {
                    throw_backend("PopSift result stage: feature geometry is invalid");
                }
                for (std::size_t descriptor_index = 0;
                     descriptor_index < kSiftDescriptorLength; ++descriptor_index)
                {
                    const float value = native_descriptor->features[descriptor_index];
                    if (!std::isfinite(value))
                    {
                        throw_backend("PopSift result stage: descriptor value is invalid");
                    }
                    feature.descriptor[descriptor_index] = value;
                }
                result.features.push_back(std::move(feature));
            }
        }
        if (result.features.size() != static_cast<std::size_t>(descriptor_count))
        {
            throw_backend("PopSift result stage: descriptor count is inconsistent");
        }
        return result;
    }
    catch (const image::ImageProcessorError& error)
    {
        throw map_image_error(error);
    }
    catch (const SiftError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("PopSift extraction stage: allocation failed");
    }
    catch (const std::length_error&)
    {
        throw_resource("PopSift extraction stage: container capacity exceeded");
    }
    catch (const std::exception& error)
    {
        throw_backend(std::string("PopSift extraction stage: ") + error.what());
    }
}

} // namespace kfcore::sift
