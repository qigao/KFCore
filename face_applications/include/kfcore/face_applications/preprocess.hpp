#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace kfcore::face_applications
{

[[nodiscard]] std::vector<float> preprocess_face68(const cv::Mat& aligned_bgr);
[[nodiscard]] std::vector<float> preprocess_arcface(const cv::Mat& aligned_bgr);
[[nodiscard]] std::vector<float> preprocess_inswapper(const cv::Mat& aligned_bgr);
[[nodiscard]] std::vector<float> preprocess_gfpgan(const cv::Mat& aligned_bgr);
[[nodiscard]] std::vector<float> preprocess_age_gender(const cv::Mat& face_bgr);

} // namespace kfcore::face_applications
