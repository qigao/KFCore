#pragma once

#include "kfcore/image_processor/error.hpp"
#include "kfcore/image_processor/types.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <utility>

namespace kfcore::image::detail
{

struct PackedImageLayout
{
    std::size_t row_bytes        = 0U;
    std::size_t chroma_row_bytes = 0U;
    std::size_t chroma_rows      = 0U;
    std::size_t source_span      = 0U;
    std::size_t packed_bytes     = 0U;
};

[[noreturn]] inline void throw_layout_invalid(std::string message)
{
    throw ImageProcessorError(ImageProcessorErrorCode::InvalidArgument, std::move(message));
}

[[noreturn]] inline void throw_layout_resource(std::string message)
{
    throw ImageProcessorError(ImageProcessorErrorCode::ResourceLimitExceeded,
                              std::move(message));
}

inline std::size_t layout_checked_add(std::size_t left, std::size_t right,
                                      const char* stage)
{
    if (right > (std::numeric_limits<std::size_t>::max)() - left)
    {
        throw_layout_resource(std::string(stage) + " stage: byte count overflow");
    }
    return left + right;
}

inline std::size_t layout_checked_multiply(std::size_t left, std::size_t right,
                                           const char* stage)
{
    if (left != 0U && right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_layout_resource(std::string(stage) + " stage: byte count overflow");
    }
    return left * right;
}

// The caller validates non-null data, positive dimensions, memory kind and supported format.
inline PackedImageLayout packed_image_layout(const ImageView& image, const char* stage)
{
    const std::size_t width  = static_cast<std::size_t>(image.width);
    const std::size_t height = static_cast<std::size_t>(image.height);
    PackedImageLayout result;
    if (image.pixel_format == PixelFormat::Bgr8 ||
        image.pixel_format == PixelFormat::Rgb8)
    {
        result.row_bytes = layout_checked_multiply(width, 3U, stage);
        if (image.row_stride < result.row_bytes)
        {
            throw_layout_invalid(std::string(stage) +
                                 " stage: row stride is smaller than packed RGB8");
        }
        result.source_span = layout_checked_add(
            layout_checked_multiply(height - 1U, image.row_stride, stage),
            result.row_bytes, stage);
        result.packed_bytes = layout_checked_multiply(result.row_bytes, height, stage);
        return result;
    }

    if (image.pixel_format == PixelFormat::Yuy2 ||
        image.pixel_format == PixelFormat::Uyvy)
    {
        if ((image.width & 1) != 0)
        {
            throw_layout_invalid(std::string(stage) +
                                 " stage: YUY2 and UYVY width must be even");
        }
        result.row_bytes = layout_checked_multiply(width, 2U, stage);
        if (image.row_stride < result.row_bytes)
        {
            throw_layout_invalid(std::string(stage) +
                                 " stage: row stride is smaller than packed YUV422");
        }
        result.source_span = layout_checked_add(
            layout_checked_multiply(height - 1U, image.row_stride, stage),
            result.row_bytes, stage);
        result.packed_bytes = layout_checked_multiply(result.row_bytes, height, stage);
        return result;
    }

    if ((image.width & 1) != 0 || (image.height & 1) != 0)
    {
        throw_layout_invalid(std::string(stage) +
                             " stage: NV12, NV21, and I420 dimensions must be even");
    }
    if (image.row_stride < width || (image.row_stride & 1U) != 0U)
    {
        throw_layout_invalid(std::string(stage) +
                             " stage: YUV row stride must be even and at least the width");
    }

    result.row_bytes   = width;
    result.chroma_rows = height / 2U;
    const std::size_t y_storage = layout_checked_multiply(image.row_stride, height, stage);
    const std::size_t y_packed  = layout_checked_multiply(width, height, stage);
    if (image.pixel_format == PixelFormat::Nv12 ||
        image.pixel_format == PixelFormat::Nv21)
    {
        result.chroma_row_bytes = width;
        result.source_span = layout_checked_add(
            y_storage,
            layout_checked_add(
                layout_checked_multiply(result.chroma_rows - 1U, image.row_stride, stage),
                width, stage),
            stage);
    }
    else
    {
        const std::size_t chroma_stride = image.row_stride / 2U;
        result.chroma_row_bytes = width / 2U;
        const std::size_t u_storage =
            layout_checked_multiply(chroma_stride, result.chroma_rows, stage);
        result.source_span = layout_checked_add(
            layout_checked_add(y_storage, u_storage, stage),
            layout_checked_add(
                layout_checked_multiply(result.chroma_rows - 1U, chroma_stride, stage),
                result.chroma_row_bytes, stage),
            stage);
    }
    result.packed_bytes = layout_checked_add(
        y_packed, layout_checked_multiply(width, result.chroma_rows, stage), stage);
    return result;
}

} // namespace kfcore::image::detail
