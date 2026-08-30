#pragma once

#include "kfcore/face_models/types.hpp"
#include "kfcore/image_processor/types.hpp"

namespace kfcore::face_models::detail
{

struct FaceRoi
{
    float                  center_x = 0.0F;
    float                  center_y = 0.0F;
    float                  side     = 0.0F;
    image::AffineTransform destination_to_source;
};

FaceRoi make_face_roi(const RectF& box, std::int32_t destination_extent);

} // namespace kfcore::face_models::detail
