#include "kfcore/face_applications/error.hpp"
#include "kfcore/face_applications/geometry.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <limits>

using namespace kfcore::face_applications;

namespace
{

bool approximately_equal(float actual, float expected, float tolerance = 1.0e-3F)
{
    return std::fabs(actual - expected) <= tolerance;
}

} // namespace

spec("face application geometry")
{
    it("builds the reference Face68 square crop and inverse maps points")
    {
        cv::Mat image(300, 400, CV_8UC3, cv::Scalar::all(0));
        const FaceBox box { 100.0F, 80.0F, 220.0F, 180.0F };
        const FaceTransform transform = face68_transform(box);
        const AlignedFace crop = crop_face68(image, box);

        check(crop.image.rows == 256);
        check(crop.image.cols == 256);
        check_true(approximately_equal(transform.source_to_aligned(0, 0),
                                       crop.source_to_aligned(0, 0)));
        check_true(approximately_equal(transform.aligned_to_source(0, 2),
                                       crop.aligned_to_source(0, 2)));
        const cv::Point2f center = transform_point(crop.source_to_aligned, { 160.0F, 130.0F });
        check_true(approximately_equal(center.x, 128.0F));
        check_true(approximately_equal(center.y, 128.0F));
        const cv::Point2f restored = transform_point(crop.aligned_to_source, center);
        check_true(approximately_equal(restored.x, 160.0F));
        check_true(approximately_equal(restored.y, 130.0F));
    }

    it("reduces 68 landmarks to the reference five points")
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

    it("solves a deterministic similarity transform")
    {
        const FiveLandmarks source = { cv::Point2f(0.0F, 0.0F), cv::Point2f(2.0F, 0.0F),
                                       cv::Point2f(1.0F, 1.0F), cv::Point2f(0.0F, 2.0F),
                                       cv::Point2f(2.0F, 2.0F) };
        FiveLandmarks target {};
        for (std::size_t index = 0; index < source.size(); ++index)
        {
            target[index] = { 3.0F - 2.0F * source[index].y,
                              4.0F + 2.0F * source[index].x };
        }
        const cv::Matx23f transform = similarity_transform(source, target);
        const FaceTransform pair = alignment_transform(source, target);
        for (std::size_t index = 0; index < source.size(); ++index)
        {
            const cv::Point2f actual = transform_point(transform, source[index]);
            check_true(approximately_equal(actual.x, target[index].x));
            check_true(approximately_equal(actual.y, target[index].y));
            const cv::Point2f restored = transform_point(pair.aligned_to_source, target[index]);
            check_true(approximately_equal(restored.x, source[index].x));
            check_true(approximately_equal(restored.y, source[index].y));
        }
    }

    it("exposes the exact three canonical landmark sets")
    {
        check_true(approximately_equal(arcface_template()[0].x, 38.29459984F));
        check_true(approximately_equal(inswapper_template()[0].x, 46.29459968F));
        check_true(approximately_equal(gfpgan_template()[0].x, 192.98138112F));
        check_true(approximately_equal(arcface_template()[4].y, 92.20409968F));
        check_true(approximately_equal(gfpgan_template()[4].y, 371.1511808F));
    }

    it("rejects invalid images boxes and landmarks")
    {
        cv::Mat empty;
        check_throws_as(crop_face68(empty, { 0.0F, 0.0F, 10.0F, 10.0F }),
                        FaceApplicationError);

        cv::Mat gray(20, 20, CV_8UC1, cv::Scalar::all(0));
        check_throws_as(crop_face68(gray, { 0.0F, 0.0F, 10.0F, 10.0F }),
                        FaceApplicationError);

        cv::Mat image(20, 20, CV_8UC3, cv::Scalar::all(0));
        check_throws_as(crop_face68(image, { 5.0F, 5.0F, 5.0F, 10.0F }),
                        FaceApplicationError);

        FiveLandmarks invalid = arcface_template();
        invalid[0].x = (std::numeric_limits<float>::quiet_NaN)();
        check_throws_as(similarity_transform(invalid, arcface_template()),
                        FaceApplicationError);
    }
}
