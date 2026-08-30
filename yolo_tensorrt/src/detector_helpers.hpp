#pragma once

#include "compact_nms.hpp"
#include "engine_contract.hpp"
#include "kfcore/image_processor/types.hpp"
#include "kfcore/yolo/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kfcore::yolo::detail
{

struct BatchInputPlan
{
    std::vector<kfcore::image::ImageView> source_images;
    kfcore::image::BatchPlan              processor;
    std::size_t                           input_elements;
    std::size_t                           input_bytes;
};

struct DetectionBufferLayout
{
    std::size_t num_dets_bytes;
    std::size_t boxes_bytes;
    std::size_t scores_bytes;
    std::size_t labels_bytes;
    std::size_t total_output_bytes;
};

struct CompactNmsBufferLayout
{
    std::size_t detections_bytes;
};

struct EfficientNmsOutputView
{
    const std::int32_t* num_dets;
    std::size_t         num_dets_count;
    const void*         boxes;
    std::size_t         boxes_count;
    const void*         scores;
    std::size_t         scores_count;
    const std::int32_t* labels;
    std::size_t         labels_count;
    std::size_t         max_detections;
    TensorDataType      output_type;
};

BatchInputPlan prepare_batch(const std::vector<ImageView>& images, std::size_t min_batch,
                             std::size_t max_batch, std::int32_t input_width,
                             std::int32_t input_height, TensorDataType input_type,
                             std::size_t max_input_bytes);

LetterboxTransform compute_letterbox_transform(std::int32_t source_width,
                                               std::int32_t source_height,
                                               std::int32_t destination_width,
                                               std::int32_t destination_height);

DetectionBufferLayout compute_detection_buffer_layout(std::size_t batch, std::size_t max_detections,
                                                      TensorDataType output_type,
                                                      std::size_t    max_output_bytes);

CompactNmsBufferLayout compute_compact_nms_buffer_layout(std::size_t batch,
                                                         std::size_t max_detections,
                                                         TensorDataType output_type,
                                                         std::size_t max_output_bytes);

std::vector<DetectionFrame> decode_efficient_nms(const std::vector<ImageView>&          images,
                                                 const std::vector<LetterboxTransform>& transforms,
                                                 const EfficientNmsOutputView&          outputs);

} // namespace kfcore::yolo::detail
