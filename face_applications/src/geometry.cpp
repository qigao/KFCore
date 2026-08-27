#include "kfcore/face_applications/geometry.hpp"

#include "kfcore/face_applications/error.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace kfcore::face_applications
{
namespace
{

constexpr int kFace68Extent = 256;
constexpr float kFace68ReferenceExtent = 195.0F;

const FiveLandmarks kArcFaceTemplate = {
    cv::Point2f(38.29459984F, 51.69630032F), cv::Point2f(73.53180016F, 51.50140016F),
    cv::Point2f(56.0252F, 71.73660032F), cv::Point2f(41.54929968F, 92.36549952F),
    cv::Point2f(70.72989952F, 92.20409968F)
};

const FiveLandmarks kInSwapperTemplate = {
    cv::Point2f(46.29459968F, 51.69629952F), cv::Point2f(81.53180032F, 51.50140032F),
    cv::Point2f(64.02519936F, 71.73660032F), cv::Point2f(49.54930048F, 92.36550016F),
    cv::Point2f(78.72989952F, 92.20409984F)
};

const FiveLandmarks kGfpGanTemplate = {
    cv::Point2f(192.98138112F, 239.94707968F),
    cv::Point2f(318.90276864F, 240.19360256F),
    cv::Point2f(256.63415808F, 314.01934848F),
    cv::Point2f(201.26116864F, 371.410432F),
    cv::Point2f(313.0890496F, 371.1511808F)
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

void validate_image(const cv::Mat& image)
{
    if (image.empty())
    {
        throw_invalid("BGR image must not be empty");
    }
    if (image.type() != CV_8UC3)
    {
        throw_invalid("BGR image must use CV_8UC3 storage");
    }
}

bool finite_point(const cv::Point2f& point) noexcept
{
    return std::isfinite(point.x) && std::isfinite(point.y);
}

void validate_landmarks(const FiveLandmarks& landmarks)
{
    for (const cv::Point2f& point : landmarks)
    {
        if (!finite_point(point))
        {
            throw_landmarks("all points must be finite");
        }
    }
}

cv::Matx23f inverse_affine(const cv::Matx23f& transform)
{
    const double determinant = static_cast<double>(transform(0, 0)) * transform(1, 1) -
                               static_cast<double>(transform(0, 1)) * transform(1, 0);
    if (!std::isfinite(determinant) || std::fabs(determinant) <= 1.0e-12)
    {
        throw_landmarks("affine transform must be invertible");
    }
    const double inverse_determinant = 1.0 / determinant;
    cv::Matx23f inverse;
    inverse(0, 0) = static_cast<float>(transform(1, 1) * inverse_determinant);
    inverse(0, 1) = static_cast<float>(-transform(0, 1) * inverse_determinant);
    inverse(1, 0) = static_cast<float>(-transform(1, 0) * inverse_determinant);
    inverse(1, 1) = static_cast<float>(transform(0, 0) * inverse_determinant);
    inverse(0, 2) = -(inverse(0, 0) * transform(0, 2) +
                      inverse(0, 1) * transform(1, 2));
    inverse(1, 2) = -(inverse(1, 0) * transform(0, 2) +
                      inverse(1, 1) * transform(1, 2));
    return inverse;
}

} // namespace

cv::Point2f transform_point(const cv::Matx23f& transform, const cv::Point2f& point)
{
    if (!finite_point(point))
    {
        throw_landmarks("point must be finite");
    }
    const cv::Point2f result(transform(0, 0) * point.x + transform(0, 1) * point.y +
                                 transform(0, 2),
                             transform(1, 0) * point.x + transform(1, 1) * point.y +
                                 transform(1, 2));
    if (!finite_point(result))
    {
        throw_landmarks("transformed point must remain finite");
    }
    return result;
}

FaceTransform face68_transform(const FaceBox& box)
{
    if (!std::isfinite(box.left) || !std::isfinite(box.top) || !std::isfinite(box.right) ||
        !std::isfinite(box.bottom) || box.right <= box.left || box.bottom <= box.top)
    {
        throw_invalid("face box must be finite with positive width and height");
    }
    const float extent = (std::max)(box.right - box.left, box.bottom - box.top);
    const float scale = kFace68ReferenceExtent / extent;
    const float center_x = (box.left + box.right) * 0.5F;
    const float center_y = (box.top + box.bottom) * 0.5F;
    const cv::Matx23f source_to_aligned(scale, 0.0F, 128.0F - scale * center_x,
                                        0.0F, scale, 128.0F - scale * center_y);
    return { source_to_aligned, inverse_affine(source_to_aligned) };
}

AlignedFace crop_face68(const cv::Mat& bgr_image, const FaceBox& box)
{
    validate_image(bgr_image);
    const FaceTransform transform = face68_transform(box);
    AlignedFace result;
    result.source_to_aligned = transform.source_to_aligned;
    result.aligned_to_source = transform.aligned_to_source;
    cv::warpAffine(bgr_image, result.image, cv::Mat(result.source_to_aligned),
                   cv::Size(kFace68Extent, kFace68Extent), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar::all(0));
    return result;
}

kfcore::face_models::Face68Result map_face68_to_source(
    const kfcore::face_models::Face68Result& landmarks,
    const cv::Matx23f& aligned_to_source)
{
    kfcore::face_models::Face68Result result {};
    for (std::size_t index = 0; index < landmarks.size(); ++index)
    {
        const auto& landmark = landmarks[index];
        if (!std::isfinite(landmark.score))
        {
            throw_landmarks("scores must be finite");
        }
        const cv::Point2f mapped =
            transform_point(aligned_to_source, { landmark.x, landmark.y });
        result[index] = { mapped.x, mapped.y, landmark.score };
    }
    return result;
}

FiveLandmarks extract_five_landmarks(const kfcore::face_models::Face68Result& landmarks)
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

cv::Matx23f similarity_transform(const FiveLandmarks& source,
                                 const FiveLandmarks& target)
{
    validate_landmarks(source);
    validate_landmarks(target);
    cv::Mat coefficients(static_cast<int>(source.size() * 2U), 4, CV_64F);
    cv::Mat observations(static_cast<int>(source.size() * 2U), 1, CV_64F);
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        const int x_row = static_cast<int>(index * 2U);
        const int y_row = x_row + 1;
        coefficients.at<double>(x_row, 0) = source[index].x;
        coefficients.at<double>(x_row, 1) = -source[index].y;
        coefficients.at<double>(x_row, 2) = 1.0;
        coefficients.at<double>(x_row, 3) = 0.0;
        coefficients.at<double>(y_row, 0) = source[index].y;
        coefficients.at<double>(y_row, 1) = source[index].x;
        coefficients.at<double>(y_row, 2) = 0.0;
        coefficients.at<double>(y_row, 3) = 1.0;
        observations.at<double>(x_row, 0) = target[index].x;
        observations.at<double>(y_row, 0) = target[index].y;
    }
    cv::Mat solution;
    if (!cv::solve(coefficients, observations, solution, cv::DECOMP_SVD))
    {
        throw_landmarks("similarity transform cannot be solved");
    }
    const double a = solution.at<double>(0, 0);
    const double b = solution.at<double>(1, 0);
    const double tx = solution.at<double>(2, 0);
    const double ty = solution.at<double>(3, 0);
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(tx) || !std::isfinite(ty) ||
        a * a + b * b <= 1.0e-12)
    {
        throw_landmarks("similarity transform must be finite and non-degenerate");
    }
    return cv::Matx23f(static_cast<float>(a), static_cast<float>(-b),
                       static_cast<float>(tx), static_cast<float>(b), static_cast<float>(a),
                       static_cast<float>(ty));
}

FaceTransform alignment_transform(const FiveLandmarks& source,
                                  const FiveLandmarks& target)
{
    const cv::Matx23f source_to_aligned = similarity_transform(source, target);
    return { source_to_aligned, inverse_affine(source_to_aligned) };
}

AlignedFace align_face(const cv::Mat& bgr_image, const FiveLandmarks& source,
                       const FiveLandmarks& target, int extent)
{
    validate_image(bgr_image);
    if (extent <= 0 || extent > 4096)
    {
        throw_invalid("aligned extent must be in [1,4096]");
    }
    const FaceTransform transform = alignment_transform(source, target);
    AlignedFace result;
    result.source_to_aligned = transform.source_to_aligned;
    result.aligned_to_source = transform.aligned_to_source;
    cv::warpAffine(bgr_image, result.image, cv::Mat(result.source_to_aligned),
                   cv::Size(extent, extent), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar::all(0));
    return result;
}

const FiveLandmarks& arcface_template() noexcept { return kArcFaceTemplate; }
const FiveLandmarks& inswapper_template() noexcept { return kInSwapperTemplate; }
const FiveLandmarks& gfpgan_template() noexcept { return kGfpGanTemplate; }

} // namespace kfcore::face_applications
