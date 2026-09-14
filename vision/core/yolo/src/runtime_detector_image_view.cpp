#include "kfcore/yolo/detector.hpp"

#include "kfcore/yolo/error.hpp"

namespace kfcore::yolo
{
namespace
{

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
        throw YoloError(YoloErrorCode::InvalidArgument,
                        "YOLO detector does not accept Gray8 input");
    }
    throw YoloError(YoloErrorCode::InvalidArgument,
                    "YOLO detector received an unknown pixel format");
}

MemoryKind yolo_memory_kind(image::MemoryKind kind)
{
    switch (kind)
    {
    case image::MemoryKind::Host: return MemoryKind::Host;
    case image::MemoryKind::CudaDevice: return MemoryKind::CudaDevice;
    }
    throw YoloError(YoloErrorCode::InvalidArgument,
                    "YOLO detector received an unknown memory kind");
}

} // namespace

DetectionFrame YoloDetector::detect(const image::ImageView& image)
{
    return detect({image.data,
                   image.width,
                   image.height,
                   image.row_stride,
                   yolo_pixel_format(image.pixel_format),
                   yolo_memory_kind(image.memory_kind)});
}

} // namespace kfcore::yolo
