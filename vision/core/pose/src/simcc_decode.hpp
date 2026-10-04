#pragma once

#include <cstddef>

namespace kfcore::pose::detail
{

enum class SimccElementType
{
    Float32,
    Float16,
};

struct SimccAxisView
{
    const void* data = nullptr;
    SimccElementType data_type = SimccElementType::Float32;
    std::size_t keypoints = 0U;
    std::size_t bins = 0U;
};

struct DecodedSimccKeypoint
{
    float x = -1.0F;
    float y = -1.0F;
    float confidence = 0.0F;
};

struct SimccVisibilityDecodeDesc
{
    float beta = 0.0F;
    float sigma_x = 0.0F;
    float sigma_y = 0.0F;
};

void decode_simcc(const SimccAxisView& x_axis,
                  const SimccAxisView& y_axis,
                  float split_ratio,
                  DecodedSimccKeypoint* output,
                  std::size_t output_count);

void decode_simcc_visibility(const SimccAxisView& x_axis,
                             const SimccAxisView& y_axis,
                             const SimccVisibilityDecodeDesc& desc,
                             float* output,
                             std::size_t output_count);

} // namespace kfcore::pose::detail
