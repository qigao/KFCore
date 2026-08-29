#pragma once

#include "kfcore/face_models/types.hpp"

#include <array>

namespace kfcore::face_applications
{

struct Point2f
{
    float x = 0.0F;
    float y = 0.0F;
};

struct FaceBox
{
    float left   = 0.0F;
    float top    = 0.0F;
    float right  = 0.0F;
    float bottom = 0.0F;
};

using FiveLandmarks = std::array<Point2f, 5>;

struct AffineMatrix
{
    std::array<float, 6> values { 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F };
};

struct FaceTransform
{
    AffineMatrix source_to_aligned;
    AffineMatrix aligned_to_source;
};

[[nodiscard]] Point2f transform_point(const AffineMatrix& transform, Point2f point);
[[nodiscard]] AffineMatrix inverse_affine(const AffineMatrix& transform);
[[nodiscard]] FaceTransform face68_transform(const FaceBox& box);
[[nodiscard]] kfcore::face_models::Face68Result map_face68_to_source(
    const kfcore::face_models::Face68Result& landmarks,
    const AffineMatrix& aligned_to_source);
[[nodiscard]] FiveLandmarks extract_five_landmarks(
    const kfcore::face_models::Face68Result& landmarks);
[[nodiscard]] AffineMatrix similarity_transform(const FiveLandmarks& source,
                                                const FiveLandmarks& target);
[[nodiscard]] FaceTransform alignment_transform(const FiveLandmarks& source,
                                                const FiveLandmarks& target);

[[nodiscard]] const FiveLandmarks& arcface_template() noexcept;
[[nodiscard]] const FiveLandmarks& inswapper_template() noexcept;
[[nodiscard]] const FiveLandmarks& gfpgan_template() noexcept;

} // namespace kfcore::face_applications
