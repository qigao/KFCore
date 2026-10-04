#pragma once

#include "heatmap_decode.hpp"

#include "kfcore/image_processor/types.hpp"
#include "kfcore/pose/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kfcore::pose::detail
{

struct TopdownUdpPreprocessDesc
{
    float bbox_padding = 1.25F;
    float border_value = 0.0F;
    std::array<float, 3> mean {0.485F, 0.456F, 0.406F};
    std::array<float, 3> stddev {0.229F, 0.224F, 0.225F};
    std::size_t max_source_bytes = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes = 32U * 1024U * 1024U;
};

struct TopdownUdpPreprocessResult
{
    HeatmapSourceGeometry geometry;
    image::AffineTransform destination_to_source {};
    std::vector<float> nchw;
};

[[nodiscard]] TopdownUdpPreprocessResult
preprocess_topdown_udp(
    const image::BgrImage& source,
    const RectF& box,
    const TopdownUdpPreprocessDesc& desc,
    std::int32_t input_width,
    std::int32_t input_height);

[[nodiscard]] HeatmapSourceGeometry
topdown_udp_geometry(
    const RectF& box,
    float bbox_padding,
    std::int32_t input_width,
    std::int32_t input_height);

[[nodiscard]] image::AffineTransform
topdown_udp_destination_to_source(
    const HeatmapSourceGeometry& geometry,
    std::int32_t input_width,
    std::int32_t input_height);

} // namespace kfcore::pose::detail
