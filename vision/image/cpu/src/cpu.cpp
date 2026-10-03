#include "kfcore/image_processor/cpu.hpp"

#include "image_layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <string>

namespace kfcore::image
{
namespace
{

constexpr std::size_t kChannels = 3U;

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw ImageProcessorError(ImageProcessorErrorCode::InvalidArgument,
                              "CPU image processing stage: " + detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw ImageProcessorError(ImageProcessorErrorCode::ResourceLimitExceeded,
                              "CPU image processing stage: " + detail);
}

std::size_t checked_multiply(std::size_t left, std::size_t right, const char* subject)
{
    if (left != 0U && right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_resource(std::string(subject) + " byte count overflow");
    }
    return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* subject)
{
    if (right > (std::numeric_limits<std::size_t>::max)() - left)
    {
        throw_resource(std::string(subject) + " byte count overflow");
    }
    return left + right;
}

std::size_t packed_bytes(std::int32_t width, std::int32_t height, const char* subject)
{
    if (width <= 0 || height <= 0)
    {
        throw_invalid(std::string(subject) + " dimensions must be positive");
    }
    std::size_t bytes = checked_multiply(static_cast<std::size_t>(width),
                                         static_cast<std::size_t>(height), subject);
    return checked_multiply(bytes, kChannels, subject);
}

void validate_limit(std::size_t bytes, std::size_t limit, const char* subject)
{
    if (limit == 0U || bytes > limit)
    {
        throw_resource(std::string(subject) + " bytes exceed configured limit");
    }
}

void validate_options(const PreprocessOptions& options)
{
    if (options.output_format != PixelFormat::Bgr8 &&
        options.output_format != PixelFormat::Rgb8)
    {
        throw_invalid("output pixel format must be Bgr8 or Rgb8");
    }
    if (!std::isfinite(options.border_value) || options.border_value < 0.0F ||
        options.border_value > 255.0F)
    {
        throw_invalid("border value must be finite within [0,255]");
    }
    for (std::size_t channel = 0; channel < kChannels; ++channel)
    {
        if (!std::isfinite(options.mean[channel]) ||
            !std::isfinite(options.stddev[channel]) || options.stddev[channel] <= 0.0F)
        {
            throw_invalid("mean must be finite and stddev must be finite and positive");
        }
    }
}

std::size_t validate_owned(const BgrImage& image, std::size_t limit)
{
    const std::size_t required = packed_bytes(image.width, image.height, "owned image");
    validate_limit(required, limit, "owned image");
    if (image.pixels.size() != required)
    {
        throw_invalid("owned BGR storage must be exactly width * height * 3 bytes");
    }
    return required;
}

struct BorrowedPlan
{
    std::size_t row_bytes   = 0U;
    std::size_t source_span = 0U;
};

BorrowedPlan validate_borrowed(const ImageView& source, std::size_t limit)
{
    if (source.data == nullptr)
    {
        throw_invalid("source data must not be null");
    }
    if (source.memory_kind != MemoryKind::Host)
    {
        throw_invalid("source image must use Host memory");
    }
    (void)packed_bytes(source.width, source.height, "source image");
    if (source.pixel_format != PixelFormat::Bgr8 &&
        source.pixel_format != PixelFormat::Rgb8 &&
        source.pixel_format != PixelFormat::Nv12 &&
        source.pixel_format != PixelFormat::I420 &&
        source.pixel_format != PixelFormat::Nv21 &&
        source.pixel_format != PixelFormat::Yuy2 &&
        source.pixel_format != PixelFormat::Uyvy)
    {
        throw_invalid(
            "source pixel format must be Bgr8, Rgb8, Nv12, I420, Nv21, Yuy2, or Uyvy");
    }
    const detail::PackedImageLayout layout =
        detail::packed_image_layout(source, "source image");
    validate_limit(layout.source_span, limit, "source image");
    if (source.byte_size < layout.source_span)
    {
        throw_invalid("source capacity is smaller than its dimensions and stride");
    }
    return { layout.row_bytes, layout.source_span };
}

std::array<std::uint8_t, 3> yuv_rgb(const ImageView& source, int x, int y)
{
    const auto* bytes = static_cast<const std::uint8_t*>(source.data);
    const std::size_t height = static_cast<std::size_t>(source.height);
    int y_value = 0;
    int u_value = 0;
    int v_value = 0;
    if (source.pixel_format == PixelFormat::Yuy2 ||
        source.pixel_format == PixelFormat::Uyvy)
    {
        const auto* pair = bytes + static_cast<std::size_t>(y) * source.row_stride +
                           static_cast<std::size_t>(x / 2) * 4U;
        if (source.pixel_format == PixelFormat::Yuy2)
        {
            y_value = pair[(x & 1) == 0 ? 0U : 2U];
            u_value = pair[1];
            v_value = pair[3];
        }
        else
        {
            y_value = pair[(x & 1) == 0 ? 1U : 3U];
            u_value = pair[0];
            v_value = pair[2];
        }
    }
    else
    {
        y_value = bytes[static_cast<std::size_t>(y) * source.row_stride +
                        static_cast<std::size_t>(x)];
        const std::size_t y_storage = source.row_stride * height;
        if (source.pixel_format == PixelFormat::Nv12 ||
            source.pixel_format == PixelFormat::Nv21)
        {
            const std::size_t uv_offset = y_storage +
                static_cast<std::size_t>(y / 2) * source.row_stride +
                static_cast<std::size_t>(x / 2) * 2U;
            const bool uv_order = source.pixel_format == PixelFormat::Nv12;
            u_value = bytes[uv_offset + (uv_order ? 0U : 1U)];
            v_value = bytes[uv_offset + (uv_order ? 1U : 0U)];
        }
        else
        {
            const std::size_t chroma_stride = source.row_stride / 2U;
            const std::size_t chroma_rows = height / 2U;
            const std::size_t chroma_offset =
                static_cast<std::size_t>(y / 2) * chroma_stride +
                static_cast<std::size_t>(x / 2);
            u_value = bytes[y_storage + chroma_offset];
            v_value = bytes[y_storage + chroma_stride * chroma_rows + chroma_offset];
        }
    }

    const int c = (std::max)(0, y_value - 16);
    const int d = u_value - 128;
    const int e = v_value - 128;
    const auto channel = [](int value)
    {
        return static_cast<std::uint8_t>((std::clamp)(value, 0, 255));
    };
    return {
        channel((298 * c + 409 * e + 128) / 256),
        channel((298 * c - 100 * d - 208 * e + 128) / 256),
        channel((298 * c + 516 * d + 128) / 256),
    };
}

float borrowed_channel(const ImageView& source, int x, int y, int output_channel,
                       PixelFormat output_format, float border_value)
{
    if (x < 0 || y < 0 || x >= source.width || y >= source.height)
    {
        return border_value;
    }
    const bool output_rgb = output_format == PixelFormat::Rgb8;
    const int semantic_channel = output_rgb ? output_channel : 2 - output_channel;
    if (source.pixel_format == PixelFormat::Nv12 ||
        source.pixel_format == PixelFormat::I420 ||
        source.pixel_format == PixelFormat::Nv21 ||
        source.pixel_format == PixelFormat::Yuy2 ||
        source.pixel_format == PixelFormat::Uyvy)
    {
        const auto rgb = yuv_rgb(source, x, y);
        return static_cast<float>(rgb[static_cast<std::size_t>(semantic_channel)]);
    }
    const auto* bytes = static_cast<const std::uint8_t*>(source.data);
    const auto* pixel = bytes + static_cast<std::size_t>(y) * source.row_stride +
                        static_cast<std::size_t>(x) * kChannels;
    const bool source_rgb = source.pixel_format == PixelFormat::Rgb8;
    const int storage_channel = source_rgb ? semantic_channel : 2 - semantic_channel;
    return static_cast<float>(pixel[storage_channel]);
}

float owned_channel(const BgrImage& source, int x, int y, int bgr_channel,
                    float border_value)
{
    if (x < 0 || y < 0 || x >= source.width || y >= source.height)
    {
        return border_value;
    }
    const std::size_t offset =
        (static_cast<std::size_t>(y) * source.width + static_cast<std::size_t>(x)) * kChannels;
    return static_cast<float>(source.pixels[offset + static_cast<std::size_t>(bgr_channel)]);
}

float bilinear_owned(const BgrImage& source, float source_x, float source_y,
                     int channel, float border_value)
{
    const int x0 = static_cast<int>(std::floor(source_x));
    const int y0 = static_cast<int>(std::floor(source_y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float fx = source_x - static_cast<float>(x0);
    const float fy = source_y - static_cast<float>(y0);
    const float top = owned_channel(source, x0, y0, channel, border_value) * (1.0F - fx) +
                      owned_channel(source, x1, y0, channel, border_value) * fx;
    const float bottom = owned_channel(source, x0, y1, channel, border_value) * (1.0F - fx) +
                         owned_channel(source, x1, y1, channel, border_value) * fx;
    return top * (1.0F - fy) + bottom * fy;
}

float normalize(float value, std::size_t channel, const PreprocessOptions& options)
{
    return (value / 255.0F - options.mean[channel]) / options.stddev[channel];
}

} // namespace

BgrImage CpuImageProcessor::copy_bgr(const ImageView& source, std::size_t max_image_bytes)
{
    try
    {
        const BorrowedPlan plan = validate_borrowed(source, max_image_bytes);
        const std::size_t destination_bytes =
            packed_bytes(source.width, source.height, "owned image");
        validate_limit(destination_bytes, max_image_bytes, "owned image");
        BgrImage result;
        result.width  = source.width;
        result.height = source.height;
        result.pixels.resize(destination_bytes);
        const auto* input = static_cast<const std::uint8_t*>(source.data);
        for (std::int32_t row = 0; row < source.height; ++row)
        {
            const auto* source_row = input + static_cast<std::size_t>(row) * source.row_stride;
            auto* destination_row = result.pixels.data() +
                                    static_cast<std::size_t>(row) *
                                        static_cast<std::size_t>(source.width) * kChannels;
            if (source.pixel_format == PixelFormat::Bgr8)
            {
                std::memcpy(destination_row, source_row, plan.row_bytes);
                continue;
            }
            if (source.pixel_format == PixelFormat::Nv12 ||
                source.pixel_format == PixelFormat::I420 ||
                source.pixel_format == PixelFormat::Nv21 ||
                source.pixel_format == PixelFormat::Yuy2 ||
                source.pixel_format == PixelFormat::Uyvy)
            {
                for (std::int32_t column = 0; column < source.width; ++column)
                {
                    const auto rgb = yuv_rgb(source, column, row);
                    auto* output = destination_row +
                        static_cast<std::size_t>(column) * kChannels;
                    output[0] = rgb[2];
                    output[1] = rgb[1];
                    output[2] = rgb[0];
                }
                continue;
            }
            for (std::int32_t column = 0; column < source.width; ++column)
            {
                const auto* pixel = source_row + static_cast<std::size_t>(column) * kChannels;
                auto* output = destination_row + static_cast<std::size_t>(column) * kChannels;
                output[0] = pixel[2];
                output[1] = pixel[1];
                output[2] = pixel[0];
            }
        }
        return result;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("owned image allocation failed");
    }
}

BgrImage CpuImageProcessor::resize_bgr(const BgrImage& source,
                                       std::int32_t destination_width,
                                       std::int32_t destination_height,
                                       std::size_t max_image_bytes)
{
    try
    {
        validate_owned(source, max_image_bytes);
        const std::size_t bytes = packed_bytes(destination_width, destination_height,
                                               "resize destination");
        validate_limit(bytes, max_image_bytes, "resize destination");
        const float scale_x = static_cast<float>(source.width) /
                              static_cast<float>(destination_width);
        const float scale_y = static_cast<float>(source.height) /
                              static_cast<float>(destination_height);
        BgrImage result;
        result.width = destination_width;
        result.height = destination_height;
        result.pixels.resize(bytes);
        for (std::int32_t y = 0; y < destination_height; ++y)
        {
            const float source_y = (static_cast<float>(y) + 0.5F) * scale_y - 0.5F;
            const int y0 = (std::clamp)(static_cast<int>(std::floor(source_y)),
                                        0, source.height - 1);
            const int y1 = (std::min)(y0 + 1, source.height - 1);
            const float fy = (std::clamp)(source_y - static_cast<float>(y0), 0.0F, 1.0F);
            for (std::int32_t x = 0; x < destination_width; ++x)
            {
                const float source_x = (static_cast<float>(x) + 0.5F) * scale_x - 0.5F;
                const int x0 = (std::clamp)(static_cast<int>(std::floor(source_x)),
                                            0, source.width - 1);
                const int x1 = (std::min)(x0 + 1, source.width - 1);
                const float fx = (std::clamp)(source_x - static_cast<float>(x0), 0.0F, 1.0F);
                const std::size_t destination =
                    (static_cast<std::size_t>(y) * destination_width + x) * kChannels;
                for (int channel = 0; channel < static_cast<int>(kChannels); ++channel)
                {
                    const float top = owned_channel(source, x0, y0, channel, 0.0F) *
                                          (1.0F - fx) +
                                      owned_channel(source, x1, y0, channel, 0.0F) * fx;
                    const float bottom = owned_channel(source, x0, y1, channel, 0.0F) *
                                             (1.0F - fx) +
                                         owned_channel(source, x1, y1, channel, 0.0F) * fx;
                    result.pixels[destination + static_cast<std::size_t>(channel)] =
                        static_cast<std::uint8_t>((std::clamp)(
                            top * (1.0F - fy) + bottom * fy, 0.0F, 255.0F));
                }
            }
        }
        return result;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("resize destination allocation failed");
    }
}

BgrImage CpuImageProcessor::warp_affine_bgr(
    const BgrImage& source, std::int32_t destination_width,
    std::int32_t destination_height, const AffineTransform& destination_to_source,
    float border_value, std::size_t max_image_bytes)
{
    try
    {
        validate_owned(source, max_image_bytes);
        const std::size_t bytes = packed_bytes(destination_width, destination_height,
                                               "destination image");
        validate_limit(bytes, max_image_bytes, "destination image");
        if (!std::isfinite(border_value) || border_value < 0.0F || border_value > 255.0F)
        {
            throw_invalid("affine border value must be finite within [0,255]");
        }
        for (float coefficient : destination_to_source.destination_to_source)
        {
            if (!std::isfinite(coefficient))
            {
                throw_invalid("affine coefficients must be finite");
            }
        }

        BgrImage result;
        result.width  = destination_width;
        result.height = destination_height;
        result.pixels.resize(bytes);
        const auto& matrix = destination_to_source.destination_to_source;
        for (std::int32_t y = 0; y < destination_height; ++y)
        {
            for (std::int32_t x = 0; x < destination_width; ++x)
            {
                const float source_x = matrix[0] * static_cast<float>(x) +
                                       matrix[1] * static_cast<float>(y) + matrix[2];
                const float source_y = matrix[3] * static_cast<float>(x) +
                                       matrix[4] * static_cast<float>(y) + matrix[5];
                const std::size_t destination =
                    (static_cast<std::size_t>(y) * destination_width + x) * kChannels;
                for (int channel = 0; channel < static_cast<int>(kChannels); ++channel)
                {
                    const float value = bilinear_owned(source, source_x, source_y, channel,
                                                       border_value);
                    result.pixels[destination + static_cast<std::size_t>(channel)] =
                        static_cast<std::uint8_t>((std::clamp)(value, 0.0F, 255.0F));
                }
            }
        }
        return result;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("affine destination allocation failed");
    }
}

std::vector<float> CpuImageProcessor::to_nchw(const BgrImage& source,
                                               const PreprocessOptions& options,
                                               std::size_t max_tensor_bytes)
{
    try
    {
        validate_owned(source, (std::numeric_limits<std::size_t>::max)());
        validate_options(options);
        const std::size_t elements = checked_multiply(
            checked_multiply(static_cast<std::size_t>(source.width),
                             static_cast<std::size_t>(source.height), "tensor"),
            kChannels, "tensor");
        validate_limit(checked_multiply(elements, sizeof(float), "tensor"),
                       max_tensor_bytes, "tensor");
        std::vector<float> result(elements);
        const std::size_t plane = elements / kChannels;
        for (std::size_t pixel = 0; pixel < plane; ++pixel)
        {
            for (std::size_t output_channel = 0; output_channel < kChannels; ++output_channel)
            {
                const std::size_t source_channel =
                    options.output_format == PixelFormat::Rgb8 ? 2U - output_channel
                                                               : output_channel;
                result[output_channel * plane + pixel] = normalize(
                    static_cast<float>(source.pixels[pixel * kChannels + source_channel]),
                    output_channel, options);
            }
        }
        return result;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("tensor allocation failed");
    }
}

std::vector<float> CpuImageProcessor::letterbox_nchw(
    const ImageView& source, std::int32_t destination_width,
    std::int32_t destination_height, const PreprocessOptions& options,
    std::size_t max_source_bytes, std::size_t max_tensor_bytes,
    LetterboxTransform* transform)
{
    try
    {
        (void)validate_borrowed(source, max_source_bytes);
        validate_options(options);
        if (transform == nullptr)
        {
            throw_invalid("letterbox transform output must not be null");
        }
        if (destination_width <= 0 || destination_height <= 0)
        {
            throw_invalid("letterbox destination dimensions must be positive");
        }
        const float scale = (std::min)(
            static_cast<float>(destination_width) / static_cast<float>(source.width),
            static_cast<float>(destination_height) / static_cast<float>(source.height));
        const float pad_x =
            options.center_letterbox
                ? (static_cast<float>(destination_width) -
                   source.width * scale) * 0.5F
                : 0.0F;
        const float pad_y =
            options.center_letterbox
                ? (static_cast<float>(destination_height) -
                   source.height * scale) * 0.5F
                : 0.0F;
        *transform = {
            scale,
            pad_x,
            pad_y,
            source.width,
            source.height,
        };
        std::size_t elements = checked_multiply(static_cast<std::size_t>(destination_width),
                                                static_cast<std::size_t>(destination_height),
                                                "letterbox tensor");
        elements = checked_multiply(elements, kChannels, "letterbox tensor");
        validate_limit(checked_multiply(elements, sizeof(float), "letterbox tensor"),
                       max_tensor_bytes, "letterbox tensor");
        std::vector<float> result(elements);
        const std::size_t plane = elements / kChannels;
        const float content_right = transform->pad_x + source.width * transform->scale;
        const float content_bottom = transform->pad_y + source.height * transform->scale;
        for (std::int32_t y = 0; y < destination_height; ++y)
        {
            for (std::int32_t x = 0; x < destination_width; ++x)
            {
                const float center_x = static_cast<float>(x) + 0.5F;
                const float center_y = static_cast<float>(y) + 0.5F;
                const bool border = center_x < transform->pad_x || center_x >= content_right ||
                                    center_y < transform->pad_y || center_y >= content_bottom;
                float source_x = 0.0F;
                float source_y = 0.0F;
                if (!border)
                {
                    source_x = (center_x - transform->pad_x) / transform->scale - 0.5F;
                    source_y = (center_y - transform->pad_y) / transform->scale - 0.5F;
                    source_x = (std::clamp)(source_x, 0.0F,
                                            static_cast<float>(source.width - 1));
                    source_y = (std::clamp)(source_y, 0.0F,
                                            static_cast<float>(source.height - 1));
                    if (options.mirror_horizontal)
                    {
                        source_x = static_cast<float>(source.width - 1) - source_x;
                    }
                }
                const int x0 = static_cast<int>(std::floor(source_x));
                const int y0 = static_cast<int>(std::floor(source_y));
                const int x1 = (std::min)(x0 + 1, source.width - 1);
                const int y1 = (std::min)(y0 + 1, source.height - 1);
                const float fx = source_x - static_cast<float>(x0);
                const float fy = source_y - static_cast<float>(y0);
                const std::size_t pixel = static_cast<std::size_t>(y) * destination_width + x;
                const bool yuv_source = source.pixel_format == PixelFormat::Nv12 ||
                                        source.pixel_format == PixelFormat::I420 ||
                                        source.pixel_format == PixelFormat::Nv21 ||
                                        source.pixel_format == PixelFormat::Yuy2 ||
                                        source.pixel_format == PixelFormat::Uyvy;
                std::array<std::uint8_t, 3> top_left_rgb {};
                std::array<std::uint8_t, 3> top_right_rgb {};
                std::array<std::uint8_t, 3> bottom_left_rgb {};
                std::array<std::uint8_t, 3> bottom_right_rgb {};
                if (!border && yuv_source)
                {
                    top_left_rgb = yuv_rgb(source, x0, y0);
                    if (fx == 0.0F && fy == 0.0F)
                    {
                        top_right_rgb = top_left_rgb;
                        bottom_left_rgb = top_left_rgb;
                        bottom_right_rgb = top_left_rgb;
                    }
                    else
                    {
                        top_right_rgb = yuv_rgb(source, x1, y0);
                        bottom_left_rgb = yuv_rgb(source, x0, y1);
                        bottom_right_rgb = yuv_rgb(source, x1, y1);
                    }
                }
                for (std::size_t channel = 0; channel < kChannels; ++channel)
                {
                    float value = options.border_value;
                    if (!border)
                    {
                        float top = 0.0F;
                        float bottom = 0.0F;
                        if (yuv_source)
                        {
                            const std::size_t semantic_channel =
                                options.output_format == PixelFormat::Rgb8
                                    ? channel
                                    : 2U - channel;
                            top = static_cast<float>(top_left_rgb[semantic_channel]) *
                                      (1.0F - fx) +
                                  static_cast<float>(top_right_rgb[semantic_channel]) * fx;
                            bottom = static_cast<float>(bottom_left_rgb[semantic_channel]) *
                                         (1.0F - fx) +
                                     static_cast<float>(bottom_right_rgb[semantic_channel]) * fx;
                        }
                        else
                        {
                            top = borrowed_channel(source, x0, y0,
                                                   static_cast<int>(channel),
                                                   options.output_format,
                                                   options.border_value) * (1.0F - fx) +
                                  borrowed_channel(source, x1, y0,
                                                   static_cast<int>(channel),
                                                   options.output_format,
                                                   options.border_value) * fx;
                            bottom = borrowed_channel(source, x0, y1,
                                                      static_cast<int>(channel),
                                                      options.output_format,
                                                      options.border_value) * (1.0F - fx) +
                                     borrowed_channel(source, x1, y1,
                                                      static_cast<int>(channel),
                                                      options.output_format,
                                                      options.border_value) * fx;
                        }
                        value = top * (1.0F - fy) + bottom * fy;
                    }
                    result[channel * plane + pixel] = normalize(value, channel, options);
                }
            }
        }
        return result;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("letterbox tensor allocation failed");
    }
}

} // namespace kfcore::image
