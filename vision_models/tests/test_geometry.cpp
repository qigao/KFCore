#include "geometry.hpp"
#include "kfcore/vision_models/types.hpp"
#include "tinytest.hpp"

#include <cmath>

using namespace kfcore::vision_models;

namespace
{

constexpr float kTolerance = 1.0e-4F;

void check_close(float actual, float expected)
{
    check_true(std::fabs(actual - expected) <= kTolerance);
}

} // namespace

spec("vision model geometry")
{
    it("decodes Palm letterbox coordinates into one source-image fact space")
    {
        const kfcore::image::LetterboxTransform letterbox { 0.3F, 0.0F, 24.0F, 640, 480 };
        const detail::PalmRow row { 0.9F, 0.5F, 0.5F, 0.25F,
                                    0.5F, 0.55F, 0.5F, 0.45F };

        const PalmDetection palm = detail::decode_palm_row(row, letterbox, 192);

        check_close(palm.box.x, 240.0F);
        check_close(palm.box.y, 160.0F);
        check_close(palm.box.width, 160.0F);
        check_close(palm.box.height, 160.0F);
        check_close(palm.roi.center.x, 320.0F);
        check_close(palm.roi.center.y, 160.0F);
        check_close(palm.roi.size, 464.0F);
        check_close(palm.roi.rotation_radians, 0.0F);
    }

    it("maps a hand model crop center back to its rotated source ROI center")
    {
        const RotatedRoi roi { { 100.0F, 80.0F }, 56.0F, 0.35F };
        const kfcore::image::AffineTransform transform = detail::hand_roi_transform(roi, 224);

        const Point2f mapped = detail::transform_point(transform, 111.5F, 111.5F);
        check_close(mapped.x, roi.center.x);
        check_close(mapped.y, roi.center.y);
    }

    it("expands and vertically shifts the MediaPipe face ROI")
    {
        const RectF box { 100.0F, 50.0F, 100.0F, 100.0F };
        const detail::FaceRoi roi = detail::make_face_roi(box, 192);

        check_close(roi.center.x, 150.0F);
        check_close(roi.center.y, 95.0F);
        check_close(roi.side, 135.0F);
        const Point2f mapped = detail::transform_point(roi.destination_to_source, 95.5F, 95.5F);
        check_close(mapped.x, roi.center.x);
        check_close(mapped.y, roi.center.y);
    }
}
