#pragma once

#include "kfcore/face_applications/cpu.hpp"
#include "kfcore/face_models/types.hpp"
#include "kfcore/image_processor/types.hpp"

#include <array>
#include <cstddef>

namespace kfcore::face_applications::detail
{

using Point2f = CpuPoint2f;
using FaceBox = CpuFaceBox;
using FiveLandmarks = CpuFiveLandmarks;

struct AffineMatrix
{
    std::array<float, 6> values { 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F };
};

struct FaceTransform
{
    AffineMatrix source_to_aligned;
    AffineMatrix aligned_to_source;
};

struct AlignedFace
{
    kfcore::image::BgrImage image;
    AffineMatrix source_to_aligned;
    AffineMatrix aligned_to_source;
};

[[nodiscard]] Point2f transform_point(const AffineMatrix& transform, Point2f point);
[[nodiscard]] AffineMatrix inverse_affine(const AffineMatrix& transform);
[[nodiscard]] FaceTransform face68_transform(const FaceBox& box);
[[nodiscard]] AlignedFace crop_face68(const kfcore::image::BgrImage& image,
                                      const FaceBox& box, std::size_t max_image_bytes);
[[nodiscard]] kfcore::face_models::Face68Result map_face68_to_source(
    const kfcore::face_models::Face68Result& landmarks,
    const AffineMatrix& aligned_to_source);
[[nodiscard]] FiveLandmarks extract_five_landmarks(
    const kfcore::face_models::Face68Result& landmarks);
[[nodiscard]] AffineMatrix similarity_transform(const FiveLandmarks& source,
                                                const FiveLandmarks& target);
[[nodiscard]] FaceTransform alignment_transform(const FiveLandmarks& source,
                                                const FiveLandmarks& target);
[[nodiscard]] AlignedFace align_face(const kfcore::image::BgrImage& image,
                                     const FiveLandmarks& source,
                                     const FiveLandmarks& target, int extent,
                                     std::size_t max_image_bytes);

[[nodiscard]] const FiveLandmarks& arcface_template() noexcept;
[[nodiscard]] const FiveLandmarks& inswapper_template() noexcept;
[[nodiscard]] const FiveLandmarks& gfpgan_template() noexcept;

} // namespace kfcore::face_applications::detail
