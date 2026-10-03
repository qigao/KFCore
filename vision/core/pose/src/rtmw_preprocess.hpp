#pragma once

#include "kfcore/image_processor/types.hpp"
#include "kfcore/pose/rtmw.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace kfcore::pose::detail
{

struct RtmwCropGeometry
{
    float center_x = 0.0F;
    float center_y = 0.0F;
    float scale_width = 0.0F;
    float scale_height = 0.0F;
};

struct RtmwPreprocessResult
{
    RtmwCropGeometry geometry;
    std::vector<float> nchw;
};

[[nodiscard]] RtmwPreprocessResult
preprocess_rtmw(const image::BgrImage& source,
                const RectF& box,
                const RtmwOptions& options,
                std::int32_t input_width,
                std::int32_t input_height);

[[nodiscard]] std::pair<float, float>
rtmw_model_to_source(const RtmwCropGeometry& geometry,
                     float model_x,
                     float model_y,
                     std::int32_t input_width,
                     std::int32_t input_height);

} // namespace kfcore::pose::detail
