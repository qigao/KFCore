#pragma once

#include "kfcore/face_models/types.hpp"

#include <opencv2/core.hpp>

#include <array>

namespace kfcore::face_applications
{

struct FaceMaskOptions
{
    float              blur_fraction = 0.3F;
    std::array<int, 4> padding_percent { 0, 0, 0, 0 }; // top, right, bottom, left
};

[[nodiscard]] cv::Mat create_static_box_mask(
    cv::Size crop_size, const FaceMaskOptions& options = {});
[[nodiscard]] cv::Mat decode_inswapper(const kfcore::face_models::InSwapperResult& output);
[[nodiscard]] cv::Mat decode_gfpgan(const kfcore::face_models::GfpGanResult& output);
[[nodiscard]] cv::Mat paste_back(const cv::Mat& target_bgr, const cv::Mat& aligned_bgr,
                                 const cv::Mat& aligned_mask,
                                 const cv::Matx23f& aligned_to_target);
[[nodiscard]] cv::Mat blend_images(const cv::Mat& base_bgr, const cv::Mat& enhanced_bgr,
                                   float enhanced_strength);

} // namespace kfcore::face_applications
