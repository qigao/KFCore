#pragma once

#include "geometry.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace kfcore::face_applications::detail
{

struct FloatMask
{
    int                width  = 0;
    int                height = 0;
    std::vector<float> values;
};

struct FaceMaskOptions
{
    float              blur_fraction = 0.3F;
    std::array<int, 4> padding_percent { 0, 0, 0, 0 };
};

[[nodiscard]] FloatMask create_static_box_mask(
    int width, int height, const FaceMaskOptions& options = {});
[[nodiscard]] kfcore::image::BgrImage decode_rgb_chw(
    const std::vector<float>& values, int extent, bool signed_unit);
[[nodiscard]] kfcore::image::BgrImage paste_back(
    const kfcore::image::BgrImage& target, const kfcore::image::BgrImage& aligned,
    const FloatMask& aligned_mask, const AffineMatrix& aligned_to_target,
    std::size_t max_image_bytes);
[[nodiscard]] kfcore::image::BgrImage blend_images(
    const kfcore::image::BgrImage& base, const kfcore::image::BgrImage& enhanced,
    float enhanced_strength, std::size_t max_image_bytes);

} // namespace kfcore::face_applications::detail

