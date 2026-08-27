#include "kfcore/face_applications/preprocess.hpp"

#include "kfcore/face_applications/error.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kfcore::face_applications
{
namespace
{

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw FaceApplicationError(FaceApplicationErrorCode::InvalidArgument,
                               "face preprocessing validation stage: " + detail);
}

void validate_bgr(const cv::Mat& image, int expected_extent, const char* model)
{
    if (image.empty())
    {
        throw_invalid(std::string(model) + " image must not be empty");
    }
    if (image.type() != CV_8UC3)
    {
        throw_invalid(std::string(model) + " image must use CV_8UC3 storage");
    }
    if (expected_extent > 0 &&
        (image.rows != expected_extent || image.cols != expected_extent))
    {
        throw_invalid(std::string(model) + " image has the wrong extent");
    }
}

std::vector<float> planar(const cv::Mat& image, bool rgb, float scale, float offset,
                          const std::array<float, 3>* means = nullptr,
                          const std::array<float, 3>* standard_deviations = nullptr)
{
    const std::size_t plane = static_cast<std::size_t>(image.rows) * image.cols;
    std::vector<float> result(plane * 3U);
    for (int row = 0; row < image.rows; ++row)
    {
        const cv::Vec3b* pixels = image.ptr<cv::Vec3b>(row);
        for (int column = 0; column < image.cols; ++column)
        {
            const std::size_t pixel = static_cast<std::size_t>(row) * image.cols + column;
            for (std::size_t channel = 0; channel < 3U; ++channel)
            {
                const std::size_t source_channel = rgb ? 2U - channel : channel;
                float value = static_cast<float>(pixels[column][source_channel]) * scale + offset;
                if (means != nullptr && standard_deviations != nullptr)
                {
                    value = (value - (*means)[channel]) / (*standard_deviations)[channel];
                }
                result[channel * plane + pixel] = value;
            }
        }
    }
    return result;
}

} // namespace

std::vector<float> preprocess_face68(const cv::Mat& aligned_bgr)
{
    validate_bgr(aligned_bgr, 256, "Face68");
    return planar(aligned_bgr, false, 1.0F / 255.0F, 0.0F);
}

std::vector<float> preprocess_arcface(const cv::Mat& aligned_bgr)
{
    validate_bgr(aligned_bgr, 112, "ArcFace");
    return planar(aligned_bgr, true, 1.0F / 127.5F, -1.0F);
}

std::vector<float> preprocess_inswapper(const cv::Mat& aligned_bgr)
{
    validate_bgr(aligned_bgr, 128, "InSwapper");
    return planar(aligned_bgr, true, 1.0F / 255.0F, 0.0F);
}

std::vector<float> preprocess_gfpgan(const cv::Mat& aligned_bgr)
{
    validate_bgr(aligned_bgr, 512, "GFPGAN");
    return planar(aligned_bgr, true, 1.0F / 127.5F, -1.0F);
}

std::vector<float> preprocess_age_gender(const cv::Mat& face_bgr)
{
    validate_bgr(face_bgr, 0, "AgeGender");
    cv::Mat resized;
    cv::resize(face_bgr, resized, cv::Size(224, 224), 0.0, 0.0, cv::INTER_LINEAR);
    static constexpr std::array<float, 3> kMeans = { 0.485F, 0.456F, 0.406F };
    static constexpr std::array<float, 3> kStandardDeviations = { 0.229F, 0.224F, 0.225F };
    return planar(resized, true, 1.0F / 255.0F, 0.0F, &kMeans, &kStandardDeviations);
}

} // namespace kfcore::face_applications
