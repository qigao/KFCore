#include "kfcore/sift/error.hpp"
#include "kfcore/sift/popsift_extractor.hpp"
#include "tinytest.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using namespace kfcore::sift;

namespace
{

constexpr std::int32_t kImageWidth  = 128;
constexpr std::int32_t kImageHeight = 128;
constexpr std::int32_t kTileSize    = 8;
constexpr std::uint8_t kDarkPixel   = 16;
constexpr std::uint8_t kLightPixel  = 240;
constexpr std::size_t  kMaxFeatures = 2048;

std::vector<std::uint8_t> checkerboard()
{
    std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(kImageWidth) * static_cast<std::size_t>(kImageHeight));
    for (std::int32_t row = 0; row < kImageHeight; ++row)
    {
        for (std::int32_t column = 0; column < kImageWidth; ++column)
        {
            const bool light = ((row / kTileSize) + (column / kTileSize)) % 2 == 0;
            pixels[static_cast<std::size_t>(row) * static_cast<std::size_t>(kImageWidth) +
                   static_cast<std::size_t>(column)] = light ? kLightPixel : kDarkPixel;
        }
    }
    return pixels;
}

} // namespace

spec("PopSift adapter")
{
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
