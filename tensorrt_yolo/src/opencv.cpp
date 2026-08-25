#include "kfcore/yolo/opencv.hpp"

#include "kfcore/yolo/error.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace kfcore::yolo {
namespace {

[[noreturn]] void throw_invalid_argument(const char* message) {
    throw YoloError(YoloErrorCode::InvalidArgument, message);
}

void validate_image(const cv::Mat& image) {
    if (image.empty() || image.data == nullptr) {
        throw_invalid_argument("OpenCV image must not be empty");
    }
    if (image.type() != CV_8UC3) {
        throw_invalid_argument("OpenCV image must have type CV_8UC3");
    }
    if (image.cols <= 0 || image.rows <= 0) {
        throw_invalid_argument("OpenCV image dimensions must be positive");
    }
    if (image.cols > (std::numeric_limits<std::int32_t>::max)() ||
        image.rows > (std::numeric_limits<std::int32_t>::max)()) {
        throw_invalid_argument("OpenCV image dimensions exceed ImageView range");
    }
    const std::size_t width = static_cast<std::size_t>(image.cols);
    if (width > (std::numeric_limits<std::size_t>::max)() / 3U) {
        throw_invalid_argument("OpenCV image row byte count overflowed");
    }
    const std::size_t row_bytes = width * 3U;
    const std::size_t stride = image.step[0];
    if (stride < row_bytes) {
        throw_invalid_argument("OpenCV image row stride is smaller than its pixel row");
    }
    const std::size_t preceding_rows = static_cast<std::size_t>(image.rows - 1);
    if (preceding_rows > ((std::numeric_limits<std::size_t>::max)() - row_bytes) / stride) {
        throw_invalid_argument("OpenCV image byte span overflowed");
    }
}

void validate_pixel_format(PixelFormat pixel_format) {
    if (pixel_format != PixelFormat::Bgr8 && pixel_format != PixelFormat::Rgb8) {
        throw_invalid_argument("pixel format must be Bgr8 or Rgb8");
    }
}

void validate_options(const DrawOptions& options) {
    if (options.line_thickness <= 0) {
        throw_invalid_argument("draw line_thickness must be positive");
    }
    if (!std::isfinite(options.font_scale) || options.font_scale <= 0.0) {
        throw_invalid_argument("draw font_scale must be finite and positive");
    }
}

void validate_tracks(const cv::Mat& image, const TrackFrame& tracks) {
    if (tracks.image_width != image.cols || tracks.image_height != image.rows) {
        throw_invalid_argument("TrackFrame dimensions must match the OpenCV image");
    }
    for (const TrackedDetection& tracked : tracks.detections) {
        const BoxF& box = tracked.detection.box;
        if (!std::isfinite(box.left) || !std::isfinite(box.top) ||
            !std::isfinite(box.right) || !std::isfinite(box.bottom)) {
            throw_invalid_argument("track coordinates must be finite");
        }
        if (box.left >= box.right || box.top >= box.bottom) {
            throw_invalid_argument("track boxes must have positive area");
        }
    }
}

cv::Scalar color_for(std::uint64_t value) noexcept {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31U;
    return cv::Scalar(64U + (value & 0xBFU),
                      64U + ((value >> 8U) & 0xBFU),
                      64U + ((value >> 16U) & 0xBFU));
}

int clipped_floor(float coordinate, int maximum) noexcept {
    const double clipped = std::max(0.0, std::min(static_cast<double>(coordinate),
                                                   static_cast<double>(maximum)));
    return static_cast<int>(std::floor(clipped));
}

int clipped_ceil(float coordinate, int maximum) noexcept {
    const double clipped = std::max(0.0, std::min(static_cast<double>(coordinate),
                                                   static_cast<double>(maximum)));
    return static_cast<int>(std::ceil(clipped));
}

std::uint64_t unconfirmed_color_key(const TrackedDetection& tracked, std::size_t index) noexcept {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(tracked.detection.class_id)) << 32U) ^
           static_cast<std::uint64_t>(index);
}

}  // namespace

ImageView image_view(const cv::Mat& image, PixelFormat pixel_format) {
    validate_image(image);
    validate_pixel_format(pixel_format);
    return {image.data,
            static_cast<std::int32_t>(image.cols),
            static_cast<std::int32_t>(image.rows),
            image.step[0],
            pixel_format,
            MemoryKind::Host};
}

void draw_tracks(cv::Mat& image, const TrackFrame& tracks, const DrawOptions& options) {
    validate_image(image);
    validate_options(options);
    validate_tracks(image, tracks);

    const int maximum_x = image.cols - 1;
    const int maximum_y = image.rows - 1;
    for (std::size_t index = 0; index < tracks.detections.size(); ++index) {
        const TrackedDetection& tracked = tracks.detections[index];
        if (!tracked.track_id.has_value() && !options.draw_unconfirmed) {
            continue;
        }

        const BoxF& box = tracked.detection.box;
        const int left = clipped_floor(box.left, maximum_x);
        const int top = clipped_floor(box.top, maximum_y);
        const int right = clipped_ceil(box.right, maximum_x);
        const int bottom = clipped_ceil(box.bottom, maximum_y);
        const std::uint64_t color_key = tracked.track_id.has_value()
                                            ? *tracked.track_id
                                            : unconfirmed_color_key(tracked, index);
        const cv::Scalar color = color_for(color_key);
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color,
                      options.line_thickness, cv::LINE_8);

        if (tracked.track_id.has_value()) {
            const std::string label = std::to_string(*tracked.track_id);
            const int label_y = top > 1 ? top - 1 : top;
            cv::putText(image, label, cv::Point(left, label_y), cv::FONT_HERSHEY_SIMPLEX,
                        options.font_scale, color, options.line_thickness, cv::LINE_8);
        }
    }
}

}  // namespace kfcore::yolo
