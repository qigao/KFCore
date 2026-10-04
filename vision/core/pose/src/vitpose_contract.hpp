#pragma once

#include "heatmap_decode.hpp"

#include "kfcore/pose/schema.hpp"
#include "kfcore/pose/types.hpp"
#include "kfcore/runtime/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kfcore::pose::detail
{

struct VitPoseTensorContract
{
    runtime::TensorDescriptor input;
    runtime::TensorShape input_shape;
    std::int32_t input_width = 0;
    std::int32_t input_height = 0;

    runtime::TensorDescriptor heatmap;
    runtime::TensorShape heatmap_shape;
    std::size_t heatmap_width = 0U;
    std::size_t heatmap_height = 0U;
};

[[nodiscard]] VitPoseTensorContract
resolve_vitpose_tensor_contract(
    const std::vector<runtime::TensorDescriptor>& tensors,
    const PoseSchema& schema,
    std::int32_t requested_input_width,
    std::int32_t requested_input_height,
    const std::string& requested_input_name,
    const std::string& requested_heatmap_name);

[[nodiscard]] PoseResult
decode_vitpose_pose(
    const HeatmapView& heatmap,
    const HeatmapSourceGeometry& geometry,
    const RectF& source_box,
    const PoseSchema& schema);

} // namespace kfcore::pose::detail
