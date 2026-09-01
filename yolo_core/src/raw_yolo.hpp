#pragma once

#include "compact_nms.hpp"

#include <cstddef>
#include <vector>

namespace kfcore::yolo::detail
{

struct RawYoloOutputView
{
    const void*    predictions;
    std::size_t    prediction_count;
    std::size_t    class_count;
    std::size_t    candidate_count;
    TensorDataType output_type;
    float          score_threshold;
    float          iou_threshold;
    std::size_t    max_detections;
};

std::vector<DetectionFrame> decode_raw_yolo(
    const std::vector<ImageView>& images,
    const std::vector<LetterboxTransform>& transforms,
    const RawYoloOutputView& outputs);

} // namespace kfcore::yolo::detail
