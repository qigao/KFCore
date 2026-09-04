#pragma once

#include "kfcore/image_processor/types.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace kfcore::face_applications::demo
{

inline kfcore::image::ImageView borrowed_bgr(const cv::Mat& image, const char* role)
{
    if (image.empty() || image.type() != CV_8UC3)
    {
        throw std::runtime_error(std::string(role) + " must be a non-empty CV_8UC3 image");
    }
    const std::size_t row_bytes = static_cast<std::size_t>(image.cols) * 3U;
    const std::size_t span = static_cast<std::size_t>(image.rows - 1) * image.step + row_bytes;
    return { image.data, span, image.cols, image.rows, image.step,
             kfcore::image::PixelFormat::Bgr8, kfcore::image::MemoryKind::Host };
}

inline cv::Mat borrowed_bgr(kfcore::image::BgrImage& image, const char* role)
{
    const std::size_t expected = image.width > 0 && image.height > 0
        ? static_cast<std::size_t>(image.width) * image.height * 3U
        : 0U;
    if (image.width <= 0 || image.height <= 0 || image.pixels.size() != expected)
    {
        throw std::runtime_error(std::string(role) + " BGR storage is malformed");
    }
    return cv::Mat(image.height, image.width, CV_8UC3, image.pixels.data(),
                   static_cast<std::size_t>(image.width) * 3U);
}

} // namespace kfcore::face_applications::demo
