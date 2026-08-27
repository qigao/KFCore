#include "kfcore/face_applications/composer.hpp"
#include "kfcore/face_applications/error.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <limits>

using namespace kfcore::face_applications;

namespace
{

bool image_equal(const cv::Mat& left, const cv::Mat& right)
{
    return left.size() == right.size() && left.type() == right.type() &&
           cv::countNonZero(left.reshape(1) != right.reshape(1)) == 0;
}

} // namespace

spec("face result composition")
{
    it("creates the reference blurred static box mask")
    {
        const cv::Mat mask = create_static_box_mask(cv::Size(128, 128));
        check(mask.type() == CV_32FC1);
        check(mask.rows == 128);
        check(mask.cols == 128);
        check(mask.at<float>(0, 0) <= 0.01F);
        check(mask.at<float>(64, 64) >= 0.99F);
        check(mask.at<float>(9, 64) > 0.0F);
        check(mask.at<float>(9, 64) < 1.0F);
    }

    it("decodes RGB CHW swap and enhancer outputs into BGR pixels")
    {
        kfcore::face_models::InSwapperResult swap;
        swap.values.assign(kfcore::face_models::kInSwapperOutputElementCount, 0.0F);
        const std::size_t swap_plane = 128U * 128U;
        swap.values[0] = 1.2F;
        swap.values[swap_plane] = 0.5F;
        swap.values[2U * swap_plane] = -0.2F;
        const cv::Vec3b swap_pixel = decode_inswapper(swap).at<cv::Vec3b>(0, 0);
        check(swap_pixel[0] == std::uint8_t { 0 });
        check(swap_pixel[1] == std::uint8_t { 127 });
        check(swap_pixel[2] == std::uint8_t { 255 });

        kfcore::face_models::GfpGanResult enhanced;
        enhanced.values.assign(kfcore::face_models::kGfpGanOutputElementCount, 0.0F);
        const std::size_t gfp_plane = 512U * 512U;
        enhanced.values[0] = 1.0F;
        enhanced.values[gfp_plane] = 0.0F;
        enhanced.values[2U * gfp_plane] = -1.0F;
        const cv::Vec3b enhanced_pixel = decode_gfpgan(enhanced).at<cv::Vec3b>(0, 0);
        check(enhanced_pixel[0] == std::uint8_t { 0 });
        check(enhanced_pixel[1] == std::uint8_t { 127 });
        check(enhanced_pixel[2] == std::uint8_t { 255 });
    }

    it("pastes with identity transform and leaves caller images unchanged")
    {
        cv::Mat target(8, 8, CV_8UC3, cv::Scalar(10, 20, 30));
        cv::Mat aligned(8, 8, CV_8UC3, cv::Scalar(110, 120, 130));
        const cv::Mat before = target.clone();
        cv::Mat one_mask(8, 8, CV_32FC1, cv::Scalar::all(1.0));
        const cv::Matx23f identity(1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F);

        const cv::Mat pasted = paste_back(target, aligned, one_mask, identity);
        check_true(image_equal(pasted, aligned));
        check_true(image_equal(target, before));

        cv::Mat zero_mask(8, 8, CV_32FC1, cv::Scalar::all(0.0));
        check_true(image_equal(paste_back(target, aligned, zero_mask, identity), target));
    }

    it("blends enhancer output with explicit bounded strength")
    {
        const cv::Mat base(2, 2, CV_8UC3, cv::Scalar::all(20));
        const cv::Mat enhanced(2, 2, CV_8UC3, cv::Scalar::all(220));
        check_true(image_equal(blend_images(base, enhanced, 0.0F), base));
        check_true(image_equal(blend_images(base, enhanced, 1.0F), enhanced));
        const cv::Mat quarter = blend_images(base, enhanced, 0.25F);
        check(quarter.at<cv::Vec3b>(0, 0)[0] == std::uint8_t { 70 });
    }

    it("rejects malformed outputs masks images and blend values")
    {
        kfcore::face_models::InSwapperResult short_output;
        short_output.values.assign(1, 0.0F);
        check_throws_as(decode_inswapper(short_output), FaceApplicationError);

        kfcore::face_models::GfpGanResult non_finite;
        non_finite.values.assign(kfcore::face_models::kGfpGanOutputElementCount, 0.0F);
        non_finite.values[0] = (std::numeric_limits<float>::quiet_NaN)();
        check_throws_as(decode_gfpgan(non_finite), FaceApplicationError);

        const cv::Mat target(8, 8, CV_8UC3, cv::Scalar::all(0));
        const cv::Mat aligned(8, 8, CV_8UC3, cv::Scalar::all(0));
        const cv::Mat wrong_mask(7, 8, CV_32FC1, cv::Scalar::all(1.0));
        const cv::Matx23f identity(1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F);
        check_throws_as(paste_back(target, aligned, wrong_mask, identity),
                        FaceApplicationError);
        check_throws_as(blend_images(target, aligned, 1.1F), FaceApplicationError);
    }
}
