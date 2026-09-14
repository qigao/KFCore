#include "kfcore/yolo/detector.hpp"

#include "kfcore/yolo/error.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace kfcore::yolo
{
namespace
{

[[noreturn]] void throw_image(const char* message)
{
    throw YoloError(YoloErrorCode::InvalidArgument, message);
}

std::size_t checked_multiply(std::size_t left, std::size_t right)
{
    if (left != 0U && right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_image("YOLO image byte span overflows size_t");
    }
    return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right)
{
    if (right > (std::numeric_limits<std::size_t>::max)() - left)
    {
        throw_image("YOLO image byte span overflows size_t");
    }
    return left + right;
}

std::size_t required_span(const image::ImageView& value)
{
    if (value.data == nullptr || value.width <= 0 || value.height <= 0)
    {
        throw_image("YOLO image data and dimensions must be valid");
    }
    const std::size_t width = static_cast<std::size_t>(value.width);
    const std::size_t height = static_cast<std::size_t>(value.height);

    if (value.pixel_format == image::PixelFormat::Nv12 ||
        value.pixel_format == image::PixelFormat::Nv21 ||
        value.pixel_format == image::PixelFormat::I420)
    {
        if ((value.width & 1) != 0 || (value.height & 1) != 0 ||
            value.row_stride < width || (value.row_stride & 1U) != 0U)
        {
            throw_image("YOLO planar YUV dimensions or stride are invalid");
        }
        const std::size_t y_storage = checked_multiply(height, value.row_stride);
        const std::size_t chroma_rows = height / 2U;
        if (value.pixel_format == image::PixelFormat::Nv12 ||
            value.pixel_format == image::PixelFormat::Nv21)
        {
            const std::size_t chroma_span = checked_add(
                checked_multiply(chroma_rows - 1U, value.row_stride), width);
            return checked_add(y_storage, chroma_span);
        }
        const std::size_t chroma_stride = value.row_stride / 2U;
        const std::size_t u_storage = checked_multiply(chroma_rows, chroma_stride);
        const std::size_t v_span = checked_add(
            checked_multiply(chroma_rows - 1U, chroma_stride), width / 2U);
        return checked_add(checked_add(y_storage, u_storage), v_span);
    }

    const bool packed_422 = value.pixel_format == image::PixelFormat::Yuy2 ||
                            value.pixel_format == image::PixelFormat::Uyvy;
    if (value.pixel_format != image::PixelFormat::Bgr8 &&
        value.pixel_format != image::PixelFormat::Rgb8 && !packed_422)
    {
        throw_image("YOLO detector does not accept this image pixel format");
    }
    if (packed_422 && (value.width & 1) != 0)
    {
        throw_image("YOLO packed YUV422 image width must be even");
    }
    const std::size_t row_bytes =
        checked_multiply(width, packed_422 ? 2U : 3U);
    if (value.row_stride < row_bytes)
    {
        throw_image("YOLO image row stride is smaller than one packed row");
    }
    return checked_add(
        checked_multiply(static_cast<std::size_t>(value.height - 1), value.row_stride),
        row_bytes);
}

PixelFormat yolo_pixel_format(image::PixelFormat format)
{
    switch (format)
    {
    case image::PixelFormat::Bgr8: return PixelFormat::Bgr8;
    case image::PixelFormat::Rgb8: return PixelFormat::Rgb8;
    case image::PixelFormat::Nv12: return PixelFormat::Nv12;
    case image::PixelFormat::I420: return PixelFormat::I420;
    case image::PixelFormat::Nv21: return PixelFormat::Nv21;
    case image::PixelFormat::Yuy2: return PixelFormat::Yuy2;
    case image::PixelFormat::Uyvy: return PixelFormat::Uyvy;
    case image::PixelFormat::Gray8:
        throw_image("YOLO detector does not accept Gray8 input");
    }
    throw_image("YOLO detector received an unknown pixel format");
}

MemoryKind yolo_memory_kind(image::MemoryKind kind)
{
    switch (kind)
    {
    case image::MemoryKind::Host: return MemoryKind::Host;
    case image::MemoryKind::CudaDevice: return MemoryKind::CudaDevice;
    }
    throw_image("YOLO detector received an unknown memory kind");
}

} // namespace

DetectionFrame YoloDetector::detect(const image::ImageView& image)
{
    const std::size_t required = required_span(image);
    if (image.byte_size < required)
    {
        throw_image("YOLO image storage is smaller than its declared layout");
    }
    const ImageView adapted {
        image.data,
        image.width,
        image.height,
        image.row_stride,
        yolo_pixel_format(image.pixel_format),
        yolo_memory_kind(image.memory_kind),
    };
    return detect(adapted);
}

} // namespace kfcore::yolo
