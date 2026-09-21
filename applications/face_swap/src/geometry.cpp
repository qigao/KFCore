#include "geometry.hpp"

#include "kfcore/image_processor/cpu.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace kfcore::face_applications::detail
{
namespace
{

constexpr int   kFace68Extent          = 256;
constexpr float kFace68ReferenceExtent = 195.0F;

const FiveLandmarks kArcFaceTemplate = {
    Point2f { 38.29459984F, 51.69630032F }, Point2f { 73.53180016F, 51.50140016F },
    Point2f { 56.0252F, 71.73660032F }, Point2f { 41.54929968F, 92.36549952F },
    Point2f { 70.72989952F, 92.20409968F },
};

const FiveLandmarks kInSwapperTemplate = {
    Point2f { 46.29459968F, 51.69629952F }, Point2f { 81.53180032F, 51.50140032F },
    Point2f { 64.02519936F, 71.73660032F }, Point2f { 49.54930048F, 92.36550016F },
    Point2f { 78.72989952F, 92.20409984F },
};

const FiveLandmarks kGfpGanTemplate = {
    Point2f { 192.98138112F, 239.94707968F },
    Point2f { 318.90276864F, 240.19360256F },
    Point2f { 256.63415808F, 314.01934848F },
    Point2f { 201.26116864F, 371.410432F },
    Point2f { 313.0890496F, 371.1511808F },
};

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw FaceApplicationError(FaceApplicationErrorCode::InvalidArgument,
                                  "face geometry validation stage: " + detail);
}

[[noreturn]] void throw_landmarks(const std::string& detail)
{
    throw FaceApplicationError(FaceApplicationErrorCode::InvalidLandmarks,
                                  "face landmark validation stage: " + detail);
}

bool finite(Point2f point) noexcept
{
    return std::isfinite(point.x) && std::isfinite(point.y);
}

void validate_landmarks(const FiveLandmarks& landmarks)
{
    for (Point2f point : landmarks)
    {
        if (!finite(point))
        {
            throw_landmarks("all points must be finite");
        }
    }
}

kfcore::image::AffineTransform image_transform(const AffineMatrix& matrix)
{
    return { matrix.values };
}

} // namespace

Point2f transform_point(const AffineMatrix& transform, Point2f point)
{
    if (!finite(point))
    {
        throw_landmarks("point must be finite");
    }
    const auto& m = transform.values;
    const Point2f result { m[0] * point.x + m[1] * point.y + m[2],
                           m[3] * point.x + m[4] * point.y + m[5] };
    if (!finite(result))
    {
        throw_landmarks("transformed point must remain finite");
    }
    return result;
}

AffineMatrix inverse_affine(const AffineMatrix& transform)
{
    const auto& m = transform.values;
    for (float value : m)
    {
        if (!std::isfinite(value))
        {
            throw_landmarks("affine transform values must be finite");
        }
    }
    const double determinant = static_cast<double>(m[0]) * m[4] -
                               static_cast<double>(m[1]) * m[3];
    if (!std::isfinite(determinant) || std::fabs(determinant) <= 1.0e-12)
    {
        throw_landmarks("affine transform must be invertible");
    }
    const double reciprocal = 1.0 / determinant;
    AffineMatrix result;
    auto& inverse = result.values;
    inverse[0] = static_cast<float>(m[4] * reciprocal);
    inverse[1] = static_cast<float>(-m[1] * reciprocal);
    inverse[3] = static_cast<float>(-m[3] * reciprocal);
    inverse[4] = static_cast<float>(m[0] * reciprocal);
    inverse[2] = -(inverse[0] * m[2] + inverse[1] * m[5]);
    inverse[5] = -(inverse[3] * m[2] + inverse[4] * m[5]);
    return result;
}

FaceTransform face68_transform(const FaceBox& box)
{
    if (!std::isfinite(box.left) || !std::isfinite(box.top) ||
        !std::isfinite(box.right) || !std::isfinite(box.bottom) ||
        box.right <= box.left || box.bottom <= box.top)
    {
        throw_invalid("face box must be finite with positive width and height");
    }
    const float extent = (std::max)(box.right - box.left, box.bottom - box.top);
    const float scale = kFace68ReferenceExtent / extent;
    const float center_x = (box.left + box.right) * 0.5F;
    const float center_y = (box.top + box.bottom) * 0.5F;
    const AffineMatrix source_to_aligned {
        { scale, 0.0F, 128.0F - scale * center_x,
          0.0F, scale, 128.0F - scale * center_y },
    };
    return { source_to_aligned, inverse_affine(source_to_aligned) };
}

AlignedFace crop_face68(const kfcore::image::BgrImage& image, const FaceBox& box,
                        std::size_t max_image_bytes)
{
    const FaceTransform transform = face68_transform(box);
    AlignedFace result;
    result.source_to_aligned = transform.source_to_aligned;
    result.aligned_to_source = transform.aligned_to_source;
    result.image = kfcore::image::CpuImageProcessor::warp_affine_bgr(
        image, kFace68Extent, kFace68Extent,
        image_transform(transform.aligned_to_source), 0.0F, max_image_bytes);
    return result;
}

kfcore::face_models::Face68Result map_face68_to_source(
    const kfcore::face_models::Face68Result& landmarks,
    const AffineMatrix& aligned_to_source)
{
    kfcore::face_models::Face68Result result {};
    for (std::size_t index = 0; index < landmarks.size(); ++index)
    {
        if (!std::isfinite(landmarks[index].score))
        {
            throw_landmarks("scores must be finite");
        }
        const Point2f mapped = transform_point(
            aligned_to_source, { landmarks[index].x, landmarks[index].y });
        result[index] = { mapped.x, mapped.y, landmarks[index].score };
    }
    return result;
}

FiveLandmarks extract_five_landmarks(
    const kfcore::face_models::Face68Result& landmarks)
{
    FiveLandmarks result {};
    for (std::size_t index = 36; index <= 41; ++index)
    {
        result[0].x += landmarks[index].x;
        result[0].y += landmarks[index].y;
    }
    for (std::size_t index = 42; index <= 47; ++index)
    {
        result[1].x += landmarks[index].x;
        result[1].y += landmarks[index].y;
    }
    result[0].x /= 6.0F;
    result[0].y /= 6.0F;
    result[1].x /= 6.0F;
    result[1].y /= 6.0F;
    result[2] = { landmarks[30].x, landmarks[30].y };
    result[3] = { landmarks[48].x, landmarks[48].y };
    result[4] = { landmarks[54].x, landmarks[54].y };
    validate_landmarks(result);
    return result;
}

AffineMatrix similarity_transform(const FiveLandmarks& source,
                                  const FiveLandmarks& target)
{
    validate_landmarks(source);
    validate_landmarks(target);
    double source_x = 0.0;
    double source_y = 0.0;
    double target_x = 0.0;
    double target_y = 0.0;
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        source_x += source[index].x;
        source_y += source[index].y;
        target_x += target[index].x;
        target_y += target[index].y;
    }
    const double count = static_cast<double>(source.size());
    source_x /= count;
    source_y /= count;
    target_x /= count;
    target_y /= count;

    double denominator = 0.0;
    double real = 0.0;
    double imaginary = 0.0;
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        const double sx = static_cast<double>(source[index].x) - source_x;
        const double sy = static_cast<double>(source[index].y) - source_y;
        const double tx = static_cast<double>(target[index].x) - target_x;
        const double ty = static_cast<double>(target[index].y) - target_y;
        denominator += sx * sx + sy * sy;
        real += sx * tx + sy * ty;
        imaginary += sx * ty - sy * tx;
    }
    if (!std::isfinite(denominator) || denominator <= 1.0e-12)
    {
        throw_landmarks("similarity source points are degenerate");
    }
    const double a = real / denominator;
    const double b = imaginary / denominator;
    const double tx = target_x - a * source_x + b * source_y;
    const double ty = target_y - b * source_x - a * source_y;
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(tx) ||
        !std::isfinite(ty) || a * a + b * b <= 1.0e-12)
    {
        throw_landmarks("similarity transform must be finite and non-degenerate");
    }
    return { { static_cast<float>(a), static_cast<float>(-b), static_cast<float>(tx),
               static_cast<float>(b), static_cast<float>(a), static_cast<float>(ty) } };
}

FaceTransform alignment_transform(const FiveLandmarks& source,
                                  const FiveLandmarks& target)
{
    const AffineMatrix source_to_aligned = similarity_transform(source, target);
    return { source_to_aligned, inverse_affine(source_to_aligned) };
}

AlignedFace align_face(const kfcore::image::BgrImage& image,
                       const FiveLandmarks& source, const FiveLandmarks& target,
                       int extent, std::size_t max_image_bytes)
{
    if (extent <= 0 || extent > 4096)
    {
        throw_invalid("aligned extent must be in [1,4096]");
    }
    const FaceTransform transform = alignment_transform(source, target);
    AlignedFace result;
    result.source_to_aligned = transform.source_to_aligned;
    result.aligned_to_source = transform.aligned_to_source;
    result.image = kfcore::image::CpuImageProcessor::warp_affine_bgr(
        image, extent, extent, image_transform(transform.aligned_to_source), 0.0F,
        max_image_bytes);
    return result;
}

const FiveLandmarks& arcface_template() noexcept { return kArcFaceTemplate; }
const FiveLandmarks& inswapper_template() noexcept { return kInSwapperTemplate; }
const FiveLandmarks& gfpgan_template() noexcept { return kGfpGanTemplate; }

} // namespace kfcore::face_applications::detail

