#include "kfcore/face_applications/error.hpp"
#include "kfcore/face_applications/preprocess.hpp"
#include "tinytest.hpp"

#include <cmath>

using namespace kfcore::face_applications;

namespace
{

bool approximately_equal(float actual, float expected, float tolerance = 1.0e-6F)
{
    return std::fabs(actual - expected) <= tolerance;
}

cv::Mat constant_bgr(int extent, std::uint8_t blue, std::uint8_t green, std::uint8_t red)
{
    return cv::Mat(extent, extent, CV_8UC3, cv::Scalar(blue, green, red));
}

} // namespace

spec("face model image preprocessing")
{
    it("keeps BGR planar order for Face68 and divides by 255")
    {
        const cv::Mat image = constant_bgr(256, 10, 20, 30);
        const std::vector<float> tensor = preprocess_face68(image);
        const std::size_t plane = 256U * 256U;
        check(tensor.size() == 3U * plane);
        check_true(approximately_equal(tensor[0], 10.0F / 255.0F));
        check_true(approximately_equal(tensor[plane], 20.0F / 255.0F));
        check_true(approximately_equal(tensor[2U * plane], 30.0F / 255.0F));
    }

    it("uses RGB planar normalized inputs for ArcFace InSwapper and GFPGAN")
    {
        const std::vector<float> arcface =
            preprocess_arcface(constant_bgr(112, 0, 127, 255));
        const std::size_t arc_plane = 112U * 112U;
        check_true(approximately_equal(arcface[0], 1.0F));
        check_true(approximately_equal(arcface[arc_plane], 127.0F / 127.5F - 1.0F));
        check_true(approximately_equal(arcface[2U * arc_plane], -1.0F));

        const std::vector<float> swap =
            preprocess_inswapper(constant_bgr(128, 0, 127, 255));
        const std::size_t swap_plane = 128U * 128U;
        check_true(approximately_equal(swap[0], 1.0F));
        check_true(approximately_equal(swap[swap_plane], 127.0F / 255.0F));
        check_true(approximately_equal(swap[2U * swap_plane], 0.0F));

        const std::vector<float> gfpgan =
            preprocess_gfpgan(constant_bgr(512, 0, 127, 255));
        const std::size_t gfp_plane = 512U * 512U;
        check_true(approximately_equal(gfpgan[0], 1.0F));
        check_true(approximately_equal(gfpgan[gfp_plane], 127.0F / 127.5F - 1.0F));
        check_true(approximately_equal(gfpgan[2U * gfp_plane], -1.0F));
    }

    it("normalizes resized age gender RGB with ImageNet constants")
    {
        const cv::Mat image = constant_bgr(10, 0, 0, 255);
        const std::vector<float> tensor = preprocess_age_gender(image);
        const std::size_t plane = 224U * 224U;
        check(tensor.size() == 3U * plane);
        check_true(approximately_equal(tensor[0], (1.0F - 0.485F) / 0.229F, 1.0e-5F));
        check_true(approximately_equal(tensor[plane], (0.0F - 0.456F) / 0.224F, 1.0e-5F));
        check_true(approximately_equal(tensor[2U * plane], (0.0F - 0.406F) / 0.225F, 1.0e-5F));
    }

    it("does not mutate input storage and rejects wrong extents or types")
    {
        cv::Mat image = constant_bgr(112, 5, 6, 7);
        const cv::Mat before = image.clone();
        (void)preprocess_arcface(image);
        check(cv::countNonZero(image.reshape(1) != before.reshape(1)) == 0);

        check_throws_as(preprocess_arcface(constant_bgr(111, 0, 0, 0)),
                        FaceApplicationError);
        cv::Mat gray(112, 112, CV_8UC1, cv::Scalar::all(0));
        check_throws_as(preprocess_arcface(gray), FaceApplicationError);
    }
}
