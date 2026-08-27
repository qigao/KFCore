#include "kfcore/face_applications/composer.hpp"

#include "kfcore/face_applications/error.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace kfcore::face_applications
{
namespace
{

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw FaceApplicationError(FaceApplicationErrorCode::InvalidArgument,
                               "face composition validation stage: " + detail);
}

[[noreturn]] void throw_output(const std::string& detail)
{
    throw FaceApplicationError(FaceApplicationErrorCode::RuntimeFailure,
                               "face model output decoding stage: " + detail);
}

void validate_bgr(const cv::Mat& image, const char* subject)
{
    if (image.empty() || image.type() != CV_8UC3)
    {
        throw_invalid(std::string(subject) + " must be a non-empty CV_8UC3 image");
    }
}

std::uint8_t byte_from_unit(float value)
{
    if (!std::isfinite(value))
    {
        throw_output("all tensor values must be finite");
    }
    return static_cast<std::uint8_t>((std::clamp)(value, 0.0F, 1.0F) * 255.0F);
}

cv::Mat decode_rgb_chw(const std::vector<float>& values, int extent, bool signed_unit,
                       std::size_t expected_count)
{
    if (values.size() != expected_count)
    {
        throw_output("tensor element count does not match the model contract");
    }
    const std::size_t plane = static_cast<std::size_t>(extent) * extent;
    cv::Mat result(extent, extent, CV_8UC3);
    for (int row = 0; row < extent; ++row)
    {
        cv::Vec3b* pixels = result.ptr<cv::Vec3b>(row);
        for (int column = 0; column < extent; ++column)
        {
            const std::size_t pixel = static_cast<std::size_t>(row) * extent + column;
            for (std::size_t rgb_channel = 0; rgb_channel < 3U; ++rgb_channel)
            {
                float value = values[rgb_channel * plane + pixel];
                if (signed_unit)
                {
                    value = (value + 1.0F) * 0.5F;
                }
                pixels[column][2U - rgb_channel] = byte_from_unit(value);
            }
        }
    }
    return result;
}

void validate_transform(const cv::Matx23f& transform)
{
    for (float value : transform.val)
    {
        if (!std::isfinite(value))
        {
            throw_invalid("affine transform values must be finite");
        }
    }
}

} // namespace

cv::Mat create_static_box_mask(cv::Size crop_size, const FaceMaskOptions& options)
{
    if (crop_size.width <= 0 || crop_size.height <= 0 || crop_size.width > 4096 ||
        crop_size.height > 4096)
    {
        throw_invalid("mask extent must be in [1,4096]");
    }
    if (!std::isfinite(options.blur_fraction) || options.blur_fraction < 0.0F ||
        options.blur_fraction > 1.0F)
    {
        throw_invalid("mask blur_fraction must be in [0,1]");
    }
    for (int padding : options.padding_percent)
    {
        if (padding < 0 || padding > 100)
        {
            throw_invalid("mask padding percentages must be in [0,100]");
        }
    }

    const float blur_amount = static_cast<float>(crop_size.width) * 0.5F *
                              options.blur_fraction;
    const int blur_area = (std::max)(static_cast<int>(blur_amount / 2.0F), 1);
    const int top = (std::max)(blur_area,
                               crop_size.height * options.padding_percent[0] / 100);
    const int right = (std::max)(blur_area,
                                 crop_size.width * options.padding_percent[1] / 100);
    const int bottom = (std::max)(blur_area,
                                  crop_size.height * options.padding_percent[2] / 100);
    const int left = (std::max)(blur_area,
                                crop_size.width * options.padding_percent[3] / 100);

    cv::Mat mask(crop_size, CV_32FC1, cv::Scalar::all(1.0));
    mask(cv::Rect(0, 0, crop_size.width, (std::min)(top, crop_size.height))).setTo(0.0F);
    mask(cv::Rect(0, (std::max)(0, crop_size.height - bottom), crop_size.width,
                  (std::min)(bottom, crop_size.height))).setTo(0.0F);
    mask(cv::Rect(0, 0, (std::min)(left, crop_size.width), crop_size.height)).setTo(0.0F);
    mask(cv::Rect((std::max)(0, crop_size.width - right), 0,
                  (std::min)(right, crop_size.width), crop_size.height)).setTo(0.0F);
    if (blur_amount > 0.0F)
    {
        cv::GaussianBlur(mask, mask, cv::Size(), blur_amount * 0.25F,
                         blur_amount * 0.25F, cv::BORDER_REPLICATE);
    }
    cv::min(mask, 1.0F, mask);
    cv::max(mask, 0.0F, mask);
    return mask;
}

cv::Mat decode_inswapper(const kfcore::face_models::InSwapperResult& output)
{
    return decode_rgb_chw(output.values, 128, false,
                          kfcore::face_models::kInSwapperOutputElementCount);
}

cv::Mat decode_gfpgan(const kfcore::face_models::GfpGanResult& output)
{
    return decode_rgb_chw(output.values, 512, true,
                          kfcore::face_models::kGfpGanOutputElementCount);
}

cv::Mat paste_back(const cv::Mat& target_bgr, const cv::Mat& aligned_bgr,
                   const cv::Mat& aligned_mask, const cv::Matx23f& aligned_to_target)
{
    validate_bgr(target_bgr, "target");
    validate_bgr(aligned_bgr, "aligned face");
    validate_transform(aligned_to_target);
    if (aligned_mask.empty() || aligned_mask.type() != CV_32FC1 ||
        aligned_mask.size() != aligned_bgr.size())
    {
        throw_invalid("aligned mask must be CV_32FC1 with the aligned face extent");
    }

    cv::Mat warped_face;
    cv::Mat warped_mask;
    cv::warpAffine(aligned_bgr, warped_face, cv::Mat(aligned_to_target), target_bgr.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::warpAffine(aligned_mask, warped_mask, cv::Mat(aligned_to_target), target_bgr.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::min(warped_mask, 1.0F, warped_mask);
    cv::max(warped_mask, 0.0F, warped_mask);

    cv::Mat result = target_bgr.clone();
    for (int row = 0; row < result.rows; ++row)
    {
        cv::Vec3b* destination = result.ptr<cv::Vec3b>(row);
        const cv::Vec3b* source = warped_face.ptr<cv::Vec3b>(row);
        const cv::Vec3b* target = target_bgr.ptr<cv::Vec3b>(row);
        const float* alpha_values = warped_mask.ptr<float>(row);
        for (int column = 0; column < result.cols; ++column)
        {
            const float alpha = alpha_values[column];
            if (!std::isfinite(alpha))
            {
                throw_invalid("warped mask values must be finite");
            }
            for (std::size_t channel = 0; channel < 3U; ++channel)
            {
                const float value = alpha * source[column][channel] +
                                    (1.0F - alpha) * target[column][channel];
                destination[column][channel] = static_cast<std::uint8_t>(
                    (std::clamp)(value, 0.0F, 255.0F));
            }
        }
    }
    return result;
}

cv::Mat blend_images(const cv::Mat& base_bgr, const cv::Mat& enhanced_bgr,
                     float enhanced_strength)
{
    validate_bgr(base_bgr, "base");
    validate_bgr(enhanced_bgr, "enhanced");
    if (base_bgr.size() != enhanced_bgr.size())
    {
        throw_invalid("blend image extents must match");
    }
    if (!std::isfinite(enhanced_strength) || enhanced_strength < 0.0F ||
        enhanced_strength > 1.0F)
    {
        throw_invalid("enhanced strength must be in [0,1]");
    }
    cv::Mat result;
    cv::addWeighted(base_bgr, 1.0F - enhanced_strength, enhanced_bgr,
                    enhanced_strength, 0.0, result);
    return result;
}

} // namespace kfcore::face_applications
