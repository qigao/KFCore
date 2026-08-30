#include "facemesh_geometry.hpp"

#include "kfcore/face_models/error.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace kfcore::face_models::detail
{
namespace
{

constexpr float kFaceRoiScale      = 1.35F;
constexpr float kFaceVerticalShift = 0.05F;

[[noreturn]] void throw_geometry(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::ModelContractMismatch,
                         "face geometry stage: " + detail);
}

void require_finite(float value, const char* name)
{
    if (!std::isfinite(value))
    {
        throw_geometry(std::string(name) + " must be finite");
    }
}

} // namespace

FaceRoi make_face_roi(const RectF& box, std::int32_t destination_extent)
{
    require_finite(box.x, "face box x");
    require_finite(box.y, "face box y");
    require_finite(box.width, "face box width");
    require_finite(box.height, "face box height");
    if (box.width <= 0.0F || box.height <= 0.0F || destination_extent <= 0)
    {
        throw_geometry("face box and destination extent must be positive");
    }
    FaceRoi roi;
    roi.side     = (std::max)(box.width, box.height) * kFaceRoiScale;
    roi.center_x = box.x + box.width * 0.5F;
    roi.center_y = box.y + box.height * 0.5F - box.height * kFaceVerticalShift;
    const float scale  = roi.side / static_cast<float>(destination_extent);
    const float center = (static_cast<float>(destination_extent) - 1.0F) * 0.5F;
    roi.destination_to_source.destination_to_source = {
        scale, 0.0F, roi.center_x - scale * center,
        0.0F, scale, roi.center_y - scale * center,
    };
    return roi;
}

} // namespace kfcore::face_models::detail
