#include "kfcore/sift/popsift_extractor.hpp"

#include "popsift_result_validation.hpp"

#include "kfcore/image_processor/error.hpp"
#include "kfcore/image_processor/image_processor.hpp"
#include "kfcore/sift/error.hpp"

#include <popsift/features.h>
#include <popsift/popsift.h>
#include <popsift/sift_conf.h>
#include <popsift/sift_extremum.h>

#include <cuda_runtime_api.h>

#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
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

std::string cuda_error_message(const char* stage, cudaError_t error)
{
    return std::string(stage) + ": " + cudaGetErrorString(error);
}

class CudaDeviceScope
{
public:
    explicit CudaDeviceScope(std::int32_t requested_device)
    {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess)
        {
            throw_backend(cuda_error_message("PopSift CUDA device count stage", error));
        }
        if (requested_device >= device_count)
        {
            throw SiftError(SiftErrorCode::InvalidArgument,
                            "PopSift CUDA device selection stage: device is outside the "
                            "available CUDA device range");
        }

        error = cudaGetDevice(&previous_device_);
        if (error != cudaSuccess)
        {
            throw_backend(cuda_error_message("PopSift current CUDA device stage", error));
        }
        device_count_ = device_count;
        if (previous_device_ != requested_device)
        {
            error = cudaSetDevice(requested_device);
            if (error != cudaSuccess)
            {
                throw_backend(cuda_error_message("PopSift CUDA device selection stage", error));
            }
            restore_required_ = true;
        }
    }

    CudaDeviceScope(const CudaDeviceScope&)            = delete;
    CudaDeviceScope& operator=(const CudaDeviceScope&) = delete;

    ~CudaDeviceScope()
    {
        if (restore_required_)
        {
            (void)cudaSetDevice(previous_device_);
        }
    }

    int device_count() const noexcept
    {
        return device_count_;
    }

    void restore()
    {
        if (!restore_required_)
        {
            return;
        }
        const cudaError_t error = cudaSetDevice(previous_device_);
        if (error != cudaSuccess)
        {
            throw_backend(cuda_error_message("PopSift CUDA device restore stage", error));
        }
        restore_required_ = false;
    }

private:
    int  previous_device_ = 0;
    int  device_count_ = 0;
    bool restore_required_ = false;
};

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

struct BackendState
{
    explicit BackendState(const PopSiftOptions& options)
        : device(options.device)
        , max_features(options.max_features)
        , normalization(options.normalization)
        , backend(std::make_unique<PopSift>(make_config(options),
                                            popsift::Config::ExtractingMode,
                                            PopSift::ByteImages, options.device))
    {
    }

    ~BackendState()
    {
        int previous_device = device;
        const bool have_previous = cudaGetDevice(&previous_device) == cudaSuccess;
        const bool selected = cudaSetDevice(device) == cudaSuccess;
        backend.reset();
        if (selected && have_previous && previous_device != device)
        {
            (void)cudaSetDevice(previous_device);
        }
    }

    BackendState(const BackendState&)            = delete;
    BackendState& operator=(const BackendState&) = delete;

    std::int32_t device;
    std::size_t max_features;
    PopSiftDescriptorNormalization normalization;
    std::unique_ptr<PopSift> backend;
};

struct BackendSlot
{
    std::shared_ptr<BackendState> backend;
    std::size_t users = 0;
    bool creating = false;
    bool destroying = false;
};

struct BackendRegistry
{
    explicit BackendRegistry(int device_count)
        : slots(static_cast<std::size_t>(device_count))
    {
    }

    std::mutex mutex;
    std::condition_variable changed;
    std::vector<BackendSlot> slots;
};

BackendRegistry& backend_registry(int device_count)
{
    static BackendRegistry registry(device_count);
    if (registry.slots.size() != static_cast<std::size_t>(device_count))
    {
        throw_backend("PopSift CUDA device registry stage: CUDA device count changed");
    }
    return registry;
}

bool compatible_backend(const BackendState& backend, const PopSiftOptions& options)
{
    return backend.max_features == options.max_features &&
           backend.normalization == options.normalization;
}

void release_backend(BackendRegistry& registry, std::size_t slot_index) noexcept
{
    std::shared_ptr<BackendState> retiring;
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        BackendSlot& slot = registry.slots[slot_index];
        if (slot.users == 0)
        {
            std::terminate();
        }
        --slot.users;
        if (slot.users != 0)
        {
            return;
        }
        slot.destroying = true;
        retiring = std::move(slot.backend);
    }

    // PopSift joins worker threads and frees CUDA allocations outside the registry lock.
    retiring.reset();

    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        registry.slots[slot_index].destroying = false;
    }
    registry.changed.notify_all();
}

class BackendLease
{
public:
    BackendLease(BackendRegistry& registry, std::size_t slot_index,
                 std::shared_ptr<BackendState> backend)
        : registry_(&registry)
        , slot_index_(slot_index)
        , backend_(std::move(backend))
    {
    }

    BackendLease(const BackendLease&)            = delete;
    BackendLease& operator=(const BackendLease&) = delete;

    BackendLease(BackendLease&& other) noexcept
        : registry_(std::exchange(other.registry_, nullptr))
        , slot_index_(other.slot_index_)
        , backend_(std::move(other.backend_))
    {
    }

    ~BackendLease()
    {
        if (registry_ != nullptr)
        {
            backend_.reset();
            release_backend(*registry_, slot_index_);
        }
    }

    BackendState* operator->() const noexcept
    {
        return backend_.get();
    }

private:
    BackendRegistry* registry_ = nullptr;
    std::size_t slot_index_ = 0;
    std::shared_ptr<BackendState> backend_;
};

BackendLease acquire_backend(const PopSiftOptions& options, int device_count)
{
    BackendRegistry& registry = backend_registry(device_count);
    const std::size_t slot_index = static_cast<std::size_t>(options.device);
    BackendSlot& slot = registry.slots[slot_index];

    std::unique_lock<std::mutex> lock(registry.mutex);
    while (slot.creating || slot.destroying)
    {
        registry.changed.wait(lock);
    }
    if (slot.backend)
    {
        if (!compatible_backend(*slot.backend, options))
        {
            lock.unlock();
            throw_resource("PopSift initialization stage: CUDA device already has an active "
                           "configuration");
        }
        if (slot.users == (std::numeric_limits<std::size_t>::max)())
        {
            lock.unlock();
            throw_resource("PopSift initialization stage: logical extractor count overflowed");
        }
        ++slot.users;
        return BackendLease(registry, slot_index, slot.backend);
    }

    slot.creating = true;
    lock.unlock();
    try
    {
        std::shared_ptr<BackendState> created = std::make_shared<BackendState>(options);
        lock.lock();
        slot.backend = created;
        slot.users = 1;
        slot.creating = false;
        lock.unlock();
        registry.changed.notify_all();
        return BackendLease(registry, slot_index, std::move(created));
    }
    catch (...)
    {
        lock.lock();
        slot.creating = false;
        lock.unlock();
        registry.changed.notify_all();
        throw;
    }
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
    Impl(PopSiftOptions requested_options, BackendLease shared_backend)
        : options(std::move(requested_options))
        , backend(std::move(shared_backend))
    {
    }

    PopSiftOptions options;
    BackendLease backend;
};

PopSiftExtractor::PopSiftExtractor(PopSiftOptions options)
{
    options.validate();
    try
    {
        CudaDeviceScope device_scope(options.device);
        BackendLease backend = acquire_backend(options, device_scope.device_count());
        impl_ = std::make_unique<Impl>(std::move(options), std::move(backend));
        device_scope.restore();
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

        CudaDeviceScope device_scope(impl_->options.device);
        std::unique_ptr<SiftJob> job(
            impl_->backend->backend->enqueue(image.width, image.height, grayscale.data()));
        if (!job)
        {
            device_scope.restore();
            throw_backend("PopSift extraction stage: backend rejected the image dimensions");
        }

        std::unique_ptr<popsift::FeaturesHost> native_features(job->getHost());
        // A submitted SiftJob must remain alive until the worker has fulfilled its future.
        device_scope.restore();
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
        const popsift::Descriptor* descriptor_base = native_features->getDescriptors();
        if (feature_count > 0 && native_data == nullptr)
        {
            throw_backend("PopSift result stage: feature storage is null");
        }
        if (descriptor_count > 0 && descriptor_base == nullptr)
        {
            throw_backend("PopSift result stage: descriptor storage is null");
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
                const std::size_t storage_index = detail::descriptor_storage_index(
                    descriptor_base, static_cast<std::size_t>(descriptor_count),
                    sizeof(popsift::Descriptor), native_descriptor);
                if (storage_index != result.features.size())
                {
                    throw_backend("PopSift result stage: descriptor storage order is inconsistent");
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
