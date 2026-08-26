#pragma once

#include "engine_contract.hpp"
#include "kfcore/yolo/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace kfcore::yolo::detail
{

inline constexpr std::uint32_t kLetterboxThreadsPerBlock = 256;
inline constexpr std::uint32_t kLetterboxMaxBlocks = 4096;

struct LetterboxTransform
{
    float        scale;
    float        pad_x;
    float        pad_y;
    std::int32_t source_width;
    std::int32_t source_height;
};

struct LetterboxLaunchPlan
{
    std::size_t   total_pixels;
    std::uint32_t block_count;
};

LetterboxLaunchPlan plan_letterbox_launch(std::size_t destination_width,
                                          std::size_t destination_height);

LetterboxTransform compute_letterbox_transform(std::int32_t source_width,
                                               std::int32_t source_height,
                                               std::int32_t destination_width,
                                               std::int32_t destination_height);

void launch_letterbox(const std::uint8_t* source, std::size_t source_stride,
                      PixelFormat source_format, void* destination,
                      std::int32_t destination_width, std::int32_t destination_height,
                      TensorDataType destination_type, const LetterboxTransform& transform,
                      const std::array<float, 3>& mean, const std::array<float, 3>& stddev,
                      float border_value, void* stream);

} // namespace kfcore::yolo::detail
