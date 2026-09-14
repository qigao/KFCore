#pragma once

#include "compact_nms.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kfcore::yolo::detail
{

struct EfficientNmsOutputView
{
    const std::int32_t* num_dets = nullptr;
    std::size_t num_dets_count = 0U;
    const void* boxes = nullptr;
    std::size_t boxes_count = 0U;
    const void* scores = nullptr;
    std::size_t scores_count = 0U;
    const std::int32_t* labels = nullptr;
    std::size_t labels_count = 0U;
    std::size_t max_detections = 0U;
    TensorDataType output_type = TensorDataType::Float32;
};

std::vector<DetectionFrame> decode_efficient_nms(
    const std::vector<ImageView>& images,
    const std::vector<LetterboxTransform>& transforms,
    const EfficientNmsOutputView& outputs);

} // namespace kfcore::yolo::detail
