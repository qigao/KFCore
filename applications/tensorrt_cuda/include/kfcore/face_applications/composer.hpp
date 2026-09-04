#pragma once

#include <array>
#include <vector>

namespace kfcore::face_applications
{

struct FloatMask
{
    int width = 0;
    int height = 0;
    std::vector<float> values;
};

struct FaceMaskOptions
{
    float blur_fraction = 0.3F;
    std::array<int, 4> padding_percent { 0, 0, 0, 0 };
};

[[nodiscard]] FloatMask create_static_box_mask(
    int width, int height, const FaceMaskOptions& options = {});

} // namespace kfcore::face_applications
