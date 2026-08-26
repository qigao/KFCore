#include "kfcore/sift/popsift_extractor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tinytest.hpp"

namespace
{

constexpr std::int32_t kImageWidth  = 128;
constexpr std::int32_t kImageHeight = 128;
constexpr std::int32_t kTileSize    = 8;
constexpr std::uint8_t kDarkPixel   = 16;
constexpr std::uint8_t kLightPixel  = 240;
constexpr std::size_t kWarmupIterations = 3;
constexpr std::size_t kMeasuredSamples  = 50;

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

suite("PopSift benchmarks")
{
    bench("persistent extractor")
    {
        const std::vector<std::uint8_t> pixels = checkerboard();
        const kfcore::image::ImageView image = {
            pixels.data(), pixels.size(), kImageWidth, kImageHeight,
            static_cast<std::size_t>(kImageWidth), kfcore::image::PixelFormat::Gray8,
            kfcore::image::MemoryKind::Host,
        };
        kfcore::sift::PopSiftOptions options;
        options.max_image_bytes = pixels.size();
        options.max_features = 2048;
        kfcore::sift::PopSiftExtractor extractor(options);

        std::size_t feature_count = 0;
        for (std::size_t iteration = 0; iteration < kWarmupIterations; ++iteration)
        {
            feature_count = extractor.extract(image).features.size();
        }
        check(feature_count != 0);

        benchmark_batch("extract 128x128 checkerboard", kMeasuredSamples)
        {
            feature_count = extractor.extract(image).features.size();
        }

        check(feature_count != 0);
    }
}
