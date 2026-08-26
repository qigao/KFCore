#include "kfcore/sift/error.hpp"
#include "kfcore/sift/popsift_extractor.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tinytest.hpp"

using namespace kfcore::sift;

namespace
{

constexpr std::int32_t kImageWidth  = 128;
constexpr std::int32_t kImageHeight = 128;
constexpr std::int32_t kTileSize    = 8;
constexpr std::uint8_t kDarkPixel   = 16;
constexpr std::uint8_t kLightPixel  = 240;
constexpr std::size_t  kMaxFeatures = 2048;

std::vector<std::uint8_t> checkerboard(std::int32_t width = kImageWidth,
                                       std::int32_t height = kImageHeight)
{
    std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    for (std::int32_t row = 0; row < height; ++row)
    {
        for (std::int32_t column = 0; column < width; ++column)
        {
            const bool light = ((row / kTileSize) + (column / kTileSize)) % 2 == 0;
            pixels[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(column)] = light ? kLightPixel : kDarkPixel;
        }
    }
    return pixels;
}

} // namespace

spec("PopSift adapter")
{
    it("rejects a CUDA device index outside the runtime range")
    {
        PopSiftOptions options;
        options.device = (std::numeric_limits<std::int32_t>::max)();
        try
        {
            PopSiftExtractor extractor(options);
            check(false);
        }
        catch (const SiftError& error)
        {
            check(error.code() == SiftErrorCode::InvalidArgument);
            check(std::string(error.what()).find("available CUDA device range") !=
                  std::string::npos);
        }
    }

    it("shares one native backend between compatible extractors on the same device")
    {
        PopSiftExtractor first;
        PopSiftExtractor second;
        const std::vector<std::uint8_t> pixels = checkerboard();
        const kfcore::image::ImageView image = {
            pixels.data(), pixels.size(), kImageWidth, kImageHeight,
            static_cast<std::size_t>(kImageWidth), kfcore::image::PixelFormat::Gray8,
            kfcore::image::MemoryKind::Host,
        };

        std::future<FeatureSet> first_result =
            std::async(std::launch::async, [&] { return first.extract(image); });
        std::future<FeatureSet> second_result =
            std::async(std::launch::async, [&] { return second.extract(image); });

        check(!first_result.get().features.empty());
        check(!second_result.get().features.empty());
    }

    it("completes concurrent jobs with a single pending slot")
    {
        constexpr std::size_t kConcurrentJobs = 6;
        PopSiftOptions options;
        options.max_pending_jobs = 1;
        PopSiftExtractor extractor(options);
        const std::vector<std::uint8_t> pixels = checkerboard();
        const kfcore::image::ImageView image = {
            pixels.data(), pixels.size(), kImageWidth, kImageHeight,
            static_cast<std::size_t>(kImageWidth), kfcore::image::PixelFormat::Gray8,
            kfcore::image::MemoryKind::Host,
        };

        std::vector<std::future<FeatureSet>> results;
        results.reserve(kConcurrentJobs);
        for (std::size_t index = 0; index < kConcurrentJobs; ++index)
        {
            results.emplace_back(
                std::async(std::launch::async, [&] { return extractor.extract(image); }));
        }
        for (std::future<FeatureSet>& result : results)
        {
            check(!result.get().features.empty());
        }
    }

    it("completes concurrent jobs with different image dimensions")
    {
        constexpr std::int32_t kSmallWidth = 96;
        constexpr std::int32_t kSmallHeight = 80;
        PopSiftExtractor extractor;
        const std::vector<std::uint8_t> small_pixels =
            checkerboard(kSmallWidth, kSmallHeight);
        const std::vector<std::uint8_t> large_pixels = checkerboard();
        const kfcore::image::ImageView small_image = {
            small_pixels.data(), small_pixels.size(), kSmallWidth, kSmallHeight,
            static_cast<std::size_t>(kSmallWidth), kfcore::image::PixelFormat::Gray8,
            kfcore::image::MemoryKind::Host,
        };
        const kfcore::image::ImageView large_image = {
            large_pixels.data(), large_pixels.size(), kImageWidth, kImageHeight,
            static_cast<std::size_t>(kImageWidth), kfcore::image::PixelFormat::Gray8,
            kfcore::image::MemoryKind::Host,
        };

        std::future<FeatureSet> small_result =
            std::async(std::launch::async, [&] { return extractor.extract(small_image); });
        std::future<FeatureSet> large_result =
            std::async(std::launch::async, [&] { return extractor.extract(large_image); });

        check(!small_result.get().features.empty());
        check(!large_result.get().features.empty());
    }

    it("coordinates concurrent first construction and final release")
    {
        constexpr int kRaceIterations = 8;
        for (int iteration = 0; iteration < kRaceIterations; ++iteration)
        {
            std::promise<void> construction_start;
            const std::shared_future<void> construction_signal =
                construction_start.get_future().share();
            auto construct = [construction_signal] {
                construction_signal.wait();
                return std::make_unique<PopSiftExtractor>();
            };
            std::future<std::unique_ptr<PopSiftExtractor>> first_future =
                std::async(std::launch::async, construct);
            std::future<std::unique_ptr<PopSiftExtractor>> second_future =
                std::async(std::launch::async, construct);
            construction_start.set_value();
            std::unique_ptr<PopSiftExtractor> first = first_future.get();
            std::unique_ptr<PopSiftExtractor> second = second_future.get();

            std::promise<void> replacement_start;
            const std::shared_future<void> replacement_signal =
                replacement_start.get_future().share();
            std::future<void> destruction = std::async(
                std::launch::async,
                [replacement_signal, first = std::move(first), second = std::move(second)]
                    () mutable {
                    replacement_signal.wait();
                    first.reset();
                    second.reset();
                });
            std::future<std::unique_ptr<PopSiftExtractor>> replacement = std::async(
                std::launch::async, [replacement_signal] {
                    replacement_signal.wait();
                    return std::make_unique<PopSiftExtractor>();
                });
            replacement_start.set_value();
            destruction.get();
            check(replacement.get() != nullptr);
        }
    }

    it("restores the caller CUDA device after successful construction and extraction")
    {
        int device_count = 0;
        check(cudaGetDeviceCount(&device_count) == cudaSuccess);
        if (device_count < 2)
        {
            check(true);
        }
        else
        {
            int original_device = 0;
            check(cudaGetDevice(&original_device) == cudaSuccess);
            PopSiftOptions options;
            options.device = original_device == 0 ? 1 : 0;
            PopSiftExtractor extractor(options);

            const std::vector<std::uint8_t> pixels = checkerboard();
            const kfcore::image::ImageView image = {
                pixels.data(), pixels.size(), kImageWidth, kImageHeight,
                static_cast<std::size_t>(kImageWidth), kfcore::image::PixelFormat::Gray8,
                kfcore::image::MemoryKind::Host,
            };
            check(!extractor.extract(image).features.empty());

            int current_device = -1;
            check(cudaGetDevice(&current_device) == cudaSuccess);
            check(current_device == original_device);
        }
    }

    it("allows a new configuration after the previous backend is released")
    {
        {
            PopSiftExtractor root_sift;
        }
        PopSiftOptions options;
        options.normalization = PopSiftDescriptorNormalization::Classic;
        PopSiftExtractor classic(options);
        check(true);
    }

    it("rejects conflicting native configurations on the same active device")
    {
        PopSiftExtractor first;
        PopSiftOptions options;
        options.normalization = PopSiftDescriptorNormalization::Classic;
        try
        {
            PopSiftExtractor second(options);
            check(false);
        }
        catch (const SiftError& error)
        {
            check(error.code() == SiftErrorCode::ResourceLimitExceeded);
            check(std::string(error.what()).find("active configuration") != std::string::npos);
        }
    }

    it("rejects conflicting pending-job capacities on the same active device")
    {
        PopSiftExtractor first;
        PopSiftOptions options;
        options.max_pending_jobs = PopSiftOptions::kDefaultMaxPendingJobs + 1U;
        try
        {
            PopSiftExtractor second(options);
            check(false);
        }
        catch (const SiftError& error)
        {
            check(error.code() == SiftErrorCode::ResourceLimitExceeded);
            check(std::string(error.what()).find("active configuration") != std::string::npos);
        }
    }

    it("extracts a bounded owned result from a Host Gray8 image")
    {
        const std::vector<std::uint8_t> pixels = checkerboard();
        const kfcore::image::ImageView image = {
            pixels.data(), pixels.size(), kImageWidth, kImageHeight,
            static_cast<std::size_t>(kImageWidth), kfcore::image::PixelFormat::Gray8,
            kfcore::image::MemoryKind::Host,
        };
        PopSiftOptions options;
        options.max_image_bytes = pixels.size();
        options.max_features    = kMaxFeatures;
        PopSiftExtractor extractor(options);

        const FeatureSet result = extractor.extract(image);

        check(!result.features.empty());
        check(result.features.size() <= options.max_features);
        for (const Feature& feature : result.features)
        {
            check(feature.descriptor.size() == kSiftDescriptorLength);
        }
    }

    it("rejects images that overflow PopSift's signed byte calculation")
    {
        constexpr std::int32_t kOverflowDimension = 46341;
        constexpr std::size_t kOverflowBytes =
            static_cast<std::size_t>(kOverflowDimension) *
            static_cast<std::size_t>(kOverflowDimension);
        const std::uint8_t pixel = 0;
        const kfcore::image::ImageView image = {
            &pixel, kOverflowBytes, kOverflowDimension, kOverflowDimension,
            static_cast<std::size_t>(kOverflowDimension), kfcore::image::PixelFormat::Gray8,
            kfcore::image::MemoryKind::Host,
        };
        PopSiftOptions options;
        options.max_image_bytes = kOverflowBytes;
        PopSiftExtractor extractor(options);
        try
        {
            (void)extractor.extract(image);
            check(false);
        }
        catch (const SiftError& error)
        {
            check(error.code() == SiftErrorCode::ResourceLimitExceeded);
            check(std::string(error.what()).find("PopSift int byte range") != std::string::npos);
        }
    }

    it("rejects CUDA-device input without an implicit transfer")
    {
        const std::vector<std::uint8_t> pixels = checkerboard();
        const kfcore::image::ImageView image = {
            pixels.data(), pixels.size(), kImageWidth, kImageHeight,
            static_cast<std::size_t>(kImageWidth), kfcore::image::PixelFormat::Gray8,
            kfcore::image::MemoryKind::CudaDevice,
        };
        PopSiftExtractor extractor;
        try
        {
            (void)extractor.extract(image);
            check(false);
        }
        catch (const SiftError& error)
        {
            check(error.code() == SiftErrorCode::InvalidArgument);
            check(std::string(error.what()).find("Host memory") != std::string::npos);
        }
    }
}
