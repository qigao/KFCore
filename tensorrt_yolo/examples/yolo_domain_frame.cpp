#include "yolo_domain_frame.hpp"

#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace kfcore::yolo::demo
{

cv::Mat to_bgr(const CapturedFrame& frame)
{
    const auto expected = packed_frame_bytes(frame.width, frame.height, frame.format);
    if (!expected.has_value() || frame.pixels.empty() ||
        frame.pixels.size() != *expected)
    {
        throw std::invalid_argument(
            "captured frame is empty, malformed, or uses an unsupported format");
    }

    cv::Mat result;
    if (frame.format == TURBO_VIDEO_CAPTURE_FORMAT_RGB24)
    {
        const cv::Mat rgb(frame.height, frame.width, CV_8UC3,
                          const_cast<std::uint8_t*>(frame.pixels.data()));
        cv::cvtColor(rgb, result, cv::COLOR_RGB2BGR);
    }
    else if (frame.format == TURBO_VIDEO_CAPTURE_FORMAT_BGRA)
    {
        const cv::Mat bgra(frame.height, frame.width, CV_8UC4,
                           const_cast<std::uint8_t*>(frame.pixels.data()));
        cv::cvtColor(bgra, result, cv::COLOR_BGRA2BGR);
    }
    else
    {
        const cv::Mat yuv(frame.height + frame.height / 2, frame.width, CV_8UC1,
                          const_cast<std::uint8_t*>(frame.pixels.data()));
        const int conversion = frame.format == TURBO_VIDEO_CAPTURE_FORMAT_NV12
                                   ? cv::COLOR_YUV2BGR_NV12
                                   : cv::COLOR_YUV2BGR_I420;
        cv::cvtColor(yuv, result, conversion);
    }
    if (result.empty() || result.type() != CV_8UC3 || result.rows != frame.height ||
        result.cols != frame.width)
    {
        throw std::runtime_error("OpenCV Lite produced an invalid BGR capture frame");
    }
    return result;
}

} // namespace kfcore::yolo::demo
