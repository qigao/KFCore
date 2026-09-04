#pragma once

#include "kfcore/image_processor/types.hpp"
#include "kfcore/yolo/types.hpp"

#include <cstddef>
#include <vector>

namespace kfcore::yolo
{

enum class TensorDataType
{
    Float32,
    Float16,
    Int32
};

namespace detail
{

using LetterboxTransform = kfcore::image::LetterboxTransform;

struct CompactNmsOutputView
{
    const void*    detections;
    std::size_t    detections_count;
    std::size_t    max_detections;
    TensorDataType output_type;
};

std::vector<DetectionFrame> decode_compact_nms(
    const std::vector<ImageView>& images,
    const std::vector<LetterboxTransform>& transforms,
    const CompactNmsOutputView& outputs);

} // namespace detail
} // namespace kfcore::yolo
