#include "geometry.hpp"
#include "tinytest.hpp"

#include <cmath>

using namespace kfcore::face_applications::detail;

namespace
{

bool approximately_equal(float actual, float expected, float tolerance = 1.0e-3F)
{
    return std::fabs(actual - expected) <= tolerance;
}

} // namespace

spec("OpenCV-free face geometry")
{
    it("builds the Face68 crop and inverse maps its center")
    {
        kfcore::image::BgrImage image;
        image.width = 400;
        image.height = 300;
        image.pixels.assign(400U * 300U * 3U, 0U);
        const FaceBox box { 100.0F, 80.0F, 220.0F, 180.0F };

        const AlignedFace crop = crop_face68(image, box, image.pixels.size());
        const Point2f center = transform_point(crop.source_to_aligned, { 160.0F, 130.0F });
        const Point2f restored = transform_point(crop.aligned_to_source, center);

        check(crop.image.width == 256);
        check(crop.image.height == 256);
        check_true(approximately_equal(center.x, 128.0F));
        check_true(approximately_equal(center.y, 128.0F));
        check_true(approximately_equal(restored.x, 160.0F));
        check_true(approximately_equal(restored.y, 130.0F));
    }

    it("reduces 68 landmarks to the reference eye nose and mouth points")
    {
        kfcore::face_models::Face68Result landmarks {};
        for (std::size_t index = 36; index <= 41; ++index)
        {
            landmarks[index] = { static_cast<float>(index), 10.0F, 1.0F };
        }
        for (std::size_t index = 42; index <= 47; ++index)
        {
            landmarks[index] = { static_cast<float>(index), 20.0F, 1.0F };
        }
        landmarks[30] = { 30.0F, 30.0F, 1.0F };
        landmarks[48] = { 48.0F, 40.0F, 1.0F };
        landmarks[54] = { 54.0F, 40.0F, 1.0F };

        const FiveLandmarks five = extract_five_landmarks(landmarks);

        check_true(approximately_equal(five[0].x, 38.5F));
        check_true(approximately_equal(five[1].x, 44.5F));
        check_true(approximately_equal(five[2].x, 30.0F));
        check_true(approximately_equal(five[3].x, 48.0F));
        check_true(approximately_equal(five[4].x, 54.0F));
    }

    it("solves and inverts a deterministic similarity transform")
    {
        const FiveLandmarks source = {
            Point2f { 0.0F, 0.0F }, Point2f { 2.0F, 0.0F }, Point2f { 1.0F, 1.0F },
            Point2f { 0.0F, 2.0F }, Point2f { 2.0F, 2.0F },
        };
        FiveLandmarks target {};
        for (std::size_t index = 0; index < source.size(); ++index)
        {
            target[index] = { 3.0F - 2.0F * source[index].y,
                              4.0F + 2.0F * source[index].x };
        }

        const FaceTransform transform = alignment_transform(source, target);
        for (std::size_t index = 0; index < source.size(); ++index)
        {
            const Point2f actual = transform_point(transform.source_to_aligned, source[index]);
            const Point2f restored = transform_point(transform.aligned_to_source, target[index]);
            check_true(approximately_equal(actual.x, target[index].x));
            check_true(approximately_equal(actual.y, target[index].y));
            check_true(approximately_equal(restored.x, source[index].x));
            check_true(approximately_equal(restored.y, source[index].y));
        }
    }
}

