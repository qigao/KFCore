#pragma once

#include "kfcore/vision_models/types.hpp"

namespace kfcore::vision_models::detail
{

struct PalmRow
{
    float score;
    float box_x;
    float box_y;
    float box_size;
    float keypoint0_x;
    float keypoint0_y;
    float keypoint2_x;
    float keypoint2_y;
};

struct FaceRoi
{
    Point2f                center;
    float                  side = 0.0F;
    image::AffineTransform destination_to_source;
};

PalmDetection decode_palm_row(const PalmRow& row,
                              const image::LetterboxTransform& letterbox,
                              std::int32_t model_extent);

image::AffineTransform hand_roi_transform(const RotatedRoi& roi,
                                          std::int32_t destination_extent);

FaceRoi make_face_roi(const RectF& box, std::int32_t destination_extent);

Point2f transform_point(const image::AffineTransform& transform, float x, float y);

} // namespace kfcore::vision_models::detail
