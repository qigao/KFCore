#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace kfcore::pose::detail
{

enum class HeatmapElementType
{
    Float32,
    Float16,
};

struct HeatmapView
{
    const void* data = nullptr;
    HeatmapElementType data_type = HeatmapElementType::Float32;
    std::size_t keypoints = 0U;
    std::size_t height = 0U;
    std::size_t width = 0U;
};

struct HeatmapDecodeDesc
{
    std::size_t dark_kernel = 11U;
};

struct DecodedHeatmapKeypoint
{
    float x = -1.0F;
    float y = -1.0F;
    float confidence = 0.0F;
};

struct HeatmapSourceGeometry
{
    float center_x = 0.0F;
    float center_y = 0.0F;
    float scale_width = 0.0F;
    float scale_height = 0.0F;
};

void decode_gaussian_heatmap_udp(
    const HeatmapView& heatmap,
    const HeatmapDecodeDesc& desc,
    DecodedHeatmapKeypoint* output,
    std::size_t output_count);

[[nodiscard]] std::pair<float, float>
heatmap_udp_to_source(
    const HeatmapSourceGeometry& geometry,
    float heatmap_x,
    float heatmap_y,
    std::size_t heatmap_width,
    std::size_t heatmap_height);

} // namespace kfcore::pose::detail
