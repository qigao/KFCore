#pragma once

#include "kfcore/face_models/types.hpp"

#include <opencv2/core.hpp>

#include <array>

namespace kfcore::face_applications
{

struct FaceBox
{
    float left   = 0.0F;
    float top    = 0.0F;
    float right  = 0.0F;
    float bottom = 0.0F;
};

using FiveLandmarks = std::array<cv::Point2f, 5>;

struct FaceTransform
{
    cv::Matx23f source_to_aligned;
    cv::Matx23f aligned_to_source;
};

struct AlignedFace
{
    cv::Mat     image;
    cv::Matx23f source_to_aligned;
    cv::Matx23f aligned_to_source;
};

[[nodiscard]] cv::Point2f transform_point(const cv::Matx23f& transform,
                                          const cv::Point2f& point);
[[nodiscard]] FaceTransform face68_transform(const FaceBox& box);
[[nodiscard]] AlignedFace crop_face68(const cv::Mat& bgr_image, const FaceBox& box);
[[nodiscard]] kfcore::face_models::Face68Result map_face68_to_source(
    const kfcore::face_models::Face68Result& landmarks,
    const cv::Matx23f& aligned_to_source);
[[nodiscard]] FiveLandmarks extract_five_landmarks(
    const kfcore::face_models::Face68Result& landmarks);
[[nodiscard]] cv::Matx23f similarity_transform(const FiveLandmarks& source,
                                               const FiveLandmarks& target);
[[nodiscard]] FaceTransform alignment_transform(const FiveLandmarks& source,
                                                const FiveLandmarks& target);
[[nodiscard]] AlignedFace align_face(const cv::Mat& bgr_image,
                                     const FiveLandmarks& source,
                                     const FiveLandmarks& target, int extent);

[[nodiscard]] const FiveLandmarks& arcface_template() noexcept;
[[nodiscard]] const FiveLandmarks& inswapper_template() noexcept;
[[nodiscard]] const FiveLandmarks& gfpgan_template() noexcept;

} // namespace kfcore::face_applications
