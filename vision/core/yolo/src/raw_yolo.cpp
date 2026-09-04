#include "raw_yolo.hpp"

#include "checked_size.hpp"
#include "kfcore/yolo/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <utility>

namespace kfcore::yolo::detail
{
namespace
{

constexpr std::size_t kBoxValueCount = 4U;

[[noreturn]] void throw_resource(std::string message)
{
    throw YoloError(YoloErrorCode::ResourceLimitExceeded, std::move(message));
}

[[noreturn]] void throw_inference(std::string message)
{
    throw YoloError(YoloErrorCode::TensorRtFailure, std::move(message));
}

float half_to_float(std::uint16_t bits) noexcept
{
    const bool negative = (bits & UINT16_C(0x8000)) != 0;
    const std::uint16_t exponent = static_cast<std::uint16_t>((bits >> 10) & 0x1fU);
    const std::uint16_t fraction = static_cast<std::uint16_t>(bits & 0x03ffU);
    float value = 0.0F;
    if (exponent == 0U)
    {
        value = std::ldexp(static_cast<float>(fraction), -24);
    }
    else if (exponent == 0x1fU)
    {
        value = fraction == 0U ? (std::numeric_limits<float>::infinity)()
                               : (std::numeric_limits<float>::quiet_NaN)();
    }
    else
    {
        value = std::ldexp(static_cast<float>(UINT16_C(0x0400) + fraction),
                           static_cast<int>(exponent) - 25);
    }
    return negative ? -value : value;
}

float value_at(const RawYoloOutputView& outputs, std::size_t index)
{
    if (outputs.output_type == TensorDataType::Float32)
    {
        return static_cast<const float*>(outputs.predictions)[index];
    }
    if (outputs.output_type == TensorDataType::Float16)
    {
        return half_to_float(static_cast<const std::uint16_t*>(outputs.predictions)[index]);
    }
    throw_inference("raw YOLO output type must be Float16 or Float32");
}

float intersection_over_union(const BoxF& left, const BoxF& right) noexcept
{
    const float overlap_width =
        (std::max)(0.0F, (std::min)(left.right, right.right) -
                              (std::max)(left.left, right.left));
    const float overlap_height =
        (std::max)(0.0F, (std::min)(left.bottom, right.bottom) -
                              (std::max)(left.top, right.top));
    const float intersection = overlap_width * overlap_height;
    const float left_area = (left.right - left.left) * (left.bottom - left.top);
    const float right_area =
        (right.right - right.left) * (right.bottom - right.top);
    const float union_area = left_area + right_area - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

struct Candidate
{
    Detection detection;
    std::size_t source_index;
};

} // namespace

std::vector<DetectionFrame> decode_raw_yolo(
    const std::vector<ImageView>& images,
    const std::vector<LetterboxTransform>& transforms,
    const RawYoloOutputView& outputs)
{
    try
    {
        if (images.empty() || transforms.size() != images.size())
        {
            throw_inference("raw YOLO image and transform counts are inconsistent");
        }
        if (outputs.predictions == nullptr || outputs.class_count == 0U ||
            outputs.candidate_count == 0U || outputs.max_detections == 0U)
        {
            throw_inference("raw YOLO output dimensions and limits must be positive");
        }
        if (!std::isfinite(outputs.score_threshold) ||
            outputs.score_threshold < 0.0F || outputs.score_threshold > 1.0F ||
            !std::isfinite(outputs.iou_threshold) ||
            outputs.iou_threshold < 0.0F || outputs.iou_threshold > 1.0F)
        {
            throw_inference("raw YOLO thresholds must be finite within [0,1]");
        }
        std::size_t values_per_image = 0U;
        if (!checked_multiply_size(kBoxValueCount + outputs.class_count,
                                   outputs.candidate_count, &values_per_image))
        {
            throw_resource("raw YOLO output element count overflow");
        }
        std::size_t expected_count = 0U;
        if (!checked_multiply_size(images.size(), values_per_image,
                                   &expected_count) ||
            expected_count != outputs.prediction_count)
        {
            throw_inference("raw YOLO output element count is inconsistent");
        }

        std::vector<DetectionFrame> frames;
        frames.reserve(images.size());
        for (std::size_t image_index = 0U; image_index < images.size(); ++image_index)
        {
            const ImageView& image = images[image_index];
            const LetterboxTransform& transform = transforms[image_index];
            if (!std::isfinite(transform.scale) || transform.scale <= 0.0F ||
                !std::isfinite(transform.pad_x) || !std::isfinite(transform.pad_y) ||
                transform.source_width != image.width ||
                transform.source_height != image.height)
            {
                throw_inference("raw YOLO letterbox transform is invalid");
            }

            std::vector<Candidate> candidates;
            candidates.reserve((std::min)(outputs.candidate_count,
                                           outputs.max_detections));
            const std::size_t image_base = image_index * values_per_image;
            for (std::size_t candidate_index = 0U;
                 candidate_index < outputs.candidate_count; ++candidate_index)
            {
                float best_score = -1.0F;
                std::size_t best_class = 0U;
                for (std::size_t class_index = 0U;
                     class_index < outputs.class_count; ++class_index)
                {
                    const float score = value_at(
                        outputs, image_base +
                                     (kBoxValueCount + class_index) *
                                         outputs.candidate_count +
                                     candidate_index);
                    if (!std::isfinite(score) || score < 0.0F || score > 1.0F)
                    {
                        throw_inference("raw YOLO score is outside [0,1]");
                    }
                    if (score > best_score)
                    {
                        best_score = score;
                        best_class = class_index;
                    }
                }
                if (best_score < outputs.score_threshold)
                {
                    continue;
                }

                const float center_x = value_at(
                    outputs, image_base + candidate_index);
                const float center_y = value_at(
                    outputs, image_base + outputs.candidate_count + candidate_index);
                const float width = value_at(
                    outputs, image_base + 2U * outputs.candidate_count + candidate_index);
                const float height = value_at(
                    outputs, image_base + 3U * outputs.candidate_count + candidate_index);
                if (!std::isfinite(center_x) || !std::isfinite(center_y) ||
                    !std::isfinite(width) || !std::isfinite(height) ||
                    width <= 0.0F || height <= 0.0F)
                {
                    throw_inference("raw YOLO box is invalid");
                }

                const float inverse_scale = 1.0F / transform.scale;
                const float half_width = width * 0.5F;
                const float half_height = height * 0.5F;
                const float image_width = static_cast<float>(image.width);
                const float image_height = static_cast<float>(image.height);
                const BoxF box {
                    (std::clamp)((center_x - half_width - transform.pad_x) * inverse_scale,
                                 0.0F, image_width),
                    (std::clamp)((center_y - half_height - transform.pad_y) * inverse_scale,
                                 0.0F, image_height),
                    (std::clamp)((center_x + half_width - transform.pad_x) * inverse_scale,
                                 0.0F, image_width),
                    (std::clamp)((center_y + half_height - transform.pad_y) * inverse_scale,
                                 0.0F, image_height),
                };
                if (box.left >= box.right || box.top >= box.bottom ||
                    best_class > static_cast<std::size_t>(
                                     (std::numeric_limits<std::int32_t>::max)()))
                {
                    continue;
                }
                candidates.push_back({{box, best_score,
                                       static_cast<std::int32_t>(best_class)},
                                      candidate_index});
            }

            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate& left, const Candidate& right) {
                          if (left.detection.score != right.detection.score)
                          {
                              return left.detection.score > right.detection.score;
                          }
                          return left.source_index < right.source_index;
                      });

            DetectionFrame frame {image.width, image.height, {}};
            frame.detections.reserve(
                (std::min)(candidates.size(), outputs.max_detections));
            for (const Candidate& candidate : candidates)
            {
                bool suppressed = false;
                for (const Detection& selected : frame.detections)
                {
                    if (selected.class_id == candidate.detection.class_id &&
                        intersection_over_union(selected.box,
                                                candidate.detection.box) >
                            outputs.iou_threshold)
                    {
                        suppressed = true;
                        break;
                    }
                }
                if (!suppressed)
                {
                    frame.detections.push_back(candidate.detection);
                    if (frame.detections.size() == outputs.max_detections)
                    {
                        break;
                    }
                }
            }
            frames.push_back(std::move(frame));
        }
        return frames;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("raw YOLO output allocation failed");
    }
    catch (const std::length_error&)
    {
        throw_resource("raw YOLO output container capacity exceeded");
    }
}

} // namespace kfcore::yolo::detail
