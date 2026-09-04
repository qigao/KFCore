#include "kfcore/face_applications/composer.hpp"

#include "kfcore/face_applications/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::face_applications
{
namespace
{

constexpr int kMaximumMaskExtent = 4096;

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw FaceApplicationError(FaceApplicationErrorCode::InvalidArgument,
                               "face composition validation stage: " + detail);
}

std::size_t pixel_count(int width, int height)
{
    if (width <= 0 || height <= 0 || width > kMaximumMaskExtent ||
        height > kMaximumMaskExtent)
    {
        throw_invalid("mask extent must be within [1,4096]");
    }
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

void gaussian_blur(FloatMask& mask, float sigma)
{
    if (sigma <= 0.0F)
    {
        return;
    }
    const int radius = static_cast<int>(std::ceil(sigma * 3.0F));
    std::vector<float> kernel(static_cast<std::size_t>(radius * 2 + 1));
    float sum = 0.0F;
    for (int offset = -radius; offset <= radius; ++offset)
    {
        const float value = std::exp(-static_cast<float>(offset * offset) /
                                     (2.0F * sigma * sigma));
        kernel[static_cast<std::size_t>(offset + radius)] = value;
        sum += value;
    }
    for (float& value : kernel)
    {
        value /= sum;
    }

    std::vector<float> horizontal(mask.values.size(), 0.0F);
    for (int y = 0; y < mask.height; ++y)
    {
        for (int x = 0; x < mask.width; ++x)
        {
            float value = 0.0F;
            for (int offset = -radius; offset <= radius; ++offset)
            {
                const int sample_x = (std::clamp)(x + offset, 0, mask.width - 1);
                value += mask.values[static_cast<std::size_t>(y) * mask.width + sample_x] *
                         kernel[static_cast<std::size_t>(offset + radius)];
            }
            horizontal[static_cast<std::size_t>(y) * mask.width + x] = value;
        }
    }

    std::vector<float> output(mask.values.size(), 0.0F);
    for (int y = 0; y < mask.height; ++y)
    {
        for (int x = 0; x < mask.width; ++x)
        {
            float value = 0.0F;
            for (int offset = -radius; offset <= radius; ++offset)
            {
                const int sample_y = (std::clamp)(y + offset, 0, mask.height - 1);
                value += horizontal[static_cast<std::size_t>(sample_y) * mask.width + x] *
                         kernel[static_cast<std::size_t>(offset + radius)];
            }
            output[static_cast<std::size_t>(y) * mask.width + x] =
                (std::clamp)(value, 0.0F, 1.0F);
        }
    }
    mask.values = std::move(output);
}

} // namespace

FloatMask create_static_box_mask(int width, int height, const FaceMaskOptions& options)
{
    const std::size_t count = pixel_count(width, height);
    if (!std::isfinite(options.blur_fraction) || options.blur_fraction < 0.0F ||
        options.blur_fraction > 1.0F)
    {
        throw_invalid("mask blur_fraction must be finite within [0,1]");
    }
    for (int padding : options.padding_percent)
    {
        if (padding < 0 || padding > 100)
        {
            throw_invalid("mask padding percentages must be within [0,100]");
        }
    }

    const float blur_amount = static_cast<float>(width) * 0.5F * options.blur_fraction;
    const int blur_area = (std::max)(static_cast<int>(blur_amount / 2.0F), 1);
    const int top = (std::max)(blur_area, height * options.padding_percent[0] / 100);
    const int right = (std::max)(blur_area, width * options.padding_percent[1] / 100);
    const int bottom = (std::max)(blur_area, height * options.padding_percent[2] / 100);
    const int left = (std::max)(blur_area, width * options.padding_percent[3] / 100);
    FloatMask mask;
    mask.width = width;
    mask.height = height;
    mask.values.assign(count, 1.0F);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (y < top || y >= height - bottom || x < left || x >= width - right)
            {
                mask.values[static_cast<std::size_t>(y) * width + x] = 0.0F;
            }
        }
    }
    gaussian_blur(mask, blur_amount * 0.25F);
    return mask;
}

} // namespace kfcore::face_applications
