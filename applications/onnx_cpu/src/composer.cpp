#include "composer.hpp"

#include "kfcore/image_processor/cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace kfcore::face_applications::detail
{
namespace
{

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::InvalidArgument,
                                  "CPU face composition validation stage: " + detail);
}

[[noreturn]] void throw_runtime(const std::string& detail)
{
    throw CpuFaceApplicationError(CpuFaceApplicationErrorCode::RuntimeFailure,
                                  "CPU face output decoding stage: " + detail);
}

std::size_t pixel_count(int width, int height)
{
    if (width <= 0 || height <= 0 || width > 4096 || height > 4096)
    {
        throw_invalid("image or mask extent must be within [1,4096]");
    }
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

void validate_image(const kfcore::image::BgrImage& image, const char* subject,
                    std::size_t max_image_bytes)
{
    const std::size_t count = pixel_count(image.width, image.height);
    if (count > (std::numeric_limits<std::size_t>::max)() / 3U ||
        image.pixels.size() != count * 3U || image.pixels.size() > max_image_bytes)
    {
        throw_invalid(std::string(subject) + " BGR storage is malformed or exceeds its limit");
    }
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

float sample_mask(const FloatMask& mask, int x, int y)
{
    if (x < 0 || y < 0 || x >= mask.width || y >= mask.height)
    {
        return 0.0F;
    }
    return mask.values[static_cast<std::size_t>(y) * mask.width + x];
}

FloatMask warp_mask(const FloatMask& source, int destination_width,
                    int destination_height, const AffineMatrix& destination_to_source)
{
    FloatMask result;
    result.width = destination_width;
    result.height = destination_height;
    result.values.assign(pixel_count(destination_width, destination_height), 0.0F);
    const auto& matrix = destination_to_source.values;
    for (int y = 0; y < destination_height; ++y)
    {
        for (int x = 0; x < destination_width; ++x)
        {
            const float sx = matrix[0] * x + matrix[1] * y + matrix[2];
            const float sy = matrix[3] * x + matrix[4] * y + matrix[5];
            const int x0 = static_cast<int>(std::floor(sx));
            const int y0 = static_cast<int>(std::floor(sy));
            const float fx = sx - x0;
            const float fy = sy - y0;
            const float top = sample_mask(source, x0, y0) * (1.0F - fx) +
                              sample_mask(source, x0 + 1, y0) * fx;
            const float bottom = sample_mask(source, x0, y0 + 1) * (1.0F - fx) +
                                 sample_mask(source, x0 + 1, y0 + 1) * fx;
            result.values[static_cast<std::size_t>(y) * destination_width + x] =
                (std::clamp)(top * (1.0F - fy) + bottom * fy, 0.0F, 1.0F);
        }
    }
    return result;
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

kfcore::image::BgrImage decode_rgb_chw(const std::vector<float>& values,
                                        int extent, bool signed_unit)
{
    const std::size_t plane = pixel_count(extent, extent);
    if (values.size() != plane * 3U)
    {
        throw_runtime("tensor element count does not match the image contract");
    }
    kfcore::image::BgrImage result;
    result.width = extent;
    result.height = extent;
    result.pixels.resize(plane * 3U);
    for (std::size_t pixel = 0; pixel < plane; ++pixel)
    {
        for (std::size_t rgb_channel = 0; rgb_channel < 3U; ++rgb_channel)
        {
            float value = values[rgb_channel * plane + pixel];
            if (!std::isfinite(value))
            {
                throw_runtime("tensor values must be finite");
            }
            if (signed_unit)
            {
                value = (value + 1.0F) * 0.5F;
            }
            result.pixels[pixel * 3U + (2U - rgb_channel)] =
                static_cast<std::uint8_t>((std::clamp)(value, 0.0F, 1.0F) * 255.0F);
        }
    }
    return result;
}

kfcore::image::BgrImage paste_back(const kfcore::image::BgrImage& target,
                                   const kfcore::image::BgrImage& aligned,
                                   const FloatMask& aligned_mask,
                                   const AffineMatrix& aligned_to_target,
                                   std::size_t max_image_bytes)
{
    validate_image(target, "target", max_image_bytes);
    validate_image(aligned, "aligned", max_image_bytes);
    if (aligned_mask.width != aligned.width || aligned_mask.height != aligned.height ||
        aligned_mask.values.size() != pixel_count(aligned.width, aligned.height))
    {
        throw_invalid("aligned mask storage must match the aligned image");
    }
    const AffineMatrix target_to_aligned = inverse_affine(aligned_to_target);
    const kfcore::image::AffineTransform image_matrix { target_to_aligned.values };
    const kfcore::image::BgrImage warped =
        kfcore::image::CpuImageProcessor::warp_affine_bgr(
            aligned, target.width, target.height, image_matrix, 0.0F, max_image_bytes);
    const FloatMask mask = warp_mask(aligned_mask, target.width, target.height,
                                     target_to_aligned);
    kfcore::image::BgrImage result = target;
    for (std::size_t pixel = 0; pixel < mask.values.size(); ++pixel)
    {
        const float alpha = mask.values[pixel];
        if (!std::isfinite(alpha))
        {
            throw_invalid("warped mask values must be finite");
        }
        for (std::size_t channel = 0; channel < 3U; ++channel)
        {
            const float value = alpha * warped.pixels[pixel * 3U + channel] +
                                (1.0F - alpha) * target.pixels[pixel * 3U + channel];
            result.pixels[pixel * 3U + channel] =
                static_cast<std::uint8_t>((std::clamp)(value, 0.0F, 255.0F));
        }
    }
    return result;
}

kfcore::image::BgrImage blend_images(const kfcore::image::BgrImage& base,
                                     const kfcore::image::BgrImage& enhanced,
                                     float enhanced_strength,
                                     std::size_t max_image_bytes)
{
    validate_image(base, "base", max_image_bytes);
    validate_image(enhanced, "enhanced", max_image_bytes);
    if (base.width != enhanced.width || base.height != enhanced.height)
    {
        throw_invalid("blend image extents must match");
    }
    if (!std::isfinite(enhanced_strength) || enhanced_strength < 0.0F ||
        enhanced_strength > 1.0F)
    {
        throw_invalid("enhanced strength must be finite within [0,1]");
    }
    kfcore::image::BgrImage result = base;
    for (std::size_t index = 0; index < result.pixels.size(); ++index)
    {
        const float value = (1.0F - enhanced_strength) * base.pixels[index] +
                            enhanced_strength * enhanced.pixels[index];
        result.pixels[index] =
            static_cast<std::uint8_t>((std::clamp)(value, 0.0F, 255.0F));
    }
    return result;
}

} // namespace kfcore::face_applications::detail
