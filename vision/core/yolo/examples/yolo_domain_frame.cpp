#include "yolo_domain_frame.hpp"

#include "kfcore/image_processor/cpu.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace kfcore::yolo::demo
{
namespace
{

std::size_t checked_image_bytes(int width, int height, std::size_t channels,
                                std::size_t limit, const char* operation)
{
    if (width <= 0 || height <= 0 || channels == 0U || limit == 0U)
    {
        throw std::invalid_argument(std::string(operation) +
                                    " dimensions and byte limit must be positive");
    }
    const std::size_t converted_width = static_cast<std::size_t>(width);
    const std::size_t converted_height = static_cast<std::size_t>(height);
    if (converted_height > (std::numeric_limits<std::size_t>::max)() /
                               converted_width ||
        converted_width * converted_height >
            (std::numeric_limits<std::size_t>::max)() / channels)
    {
        throw std::length_error(std::string(operation) + " byte count overflow");
    }
    const std::size_t bytes = converted_width * converted_height * channels;
    if (bytes > limit)
    {
        throw std::length_error(std::string(operation) +
                                " bytes exceed the configured limit");
    }
    return bytes;
}

std::string lower_extension(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char character)
                   { return static_cast<char>(std::tolower(character)); });
    return extension;
}

kfcore::image::PixelFormat processor_format(int format)
{
    switch (format)
    {
    case SALTS_VIDEO_CAPTURE_FORMAT_RGB24:
        return kfcore::image::PixelFormat::Rgb8;
    case SALTS_VIDEO_CAPTURE_FORMAT_NV12:
        return kfcore::image::PixelFormat::Nv12;
    case SALTS_VIDEO_CAPTURE_FORMAT_I420:
        return kfcore::image::PixelFormat::I420;
    default:
        throw std::invalid_argument("captured frame format has no ImageProcessor mapping");
    }
}

PixelFormat yolo_format(int format)
{
    switch (format)
    {
    case SALTS_VIDEO_CAPTURE_FORMAT_RGB24:
        return PixelFormat::Rgb8;
    case SALTS_VIDEO_CAPTURE_FORMAT_NV12:
        return PixelFormat::Nv12;
    case SALTS_VIDEO_CAPTURE_FORMAT_I420:
        return PixelFormat::I420;
    default:
        throw std::invalid_argument("captured frame format has no YOLO image mapping");
    }
}

void validate_captured(const CapturedFrame& frame)
{
    const auto expected = packed_frame_bytes(frame.width, frame.height, frame.format);
    if (!expected.has_value() || frame.pixels.empty() ||
        frame.pixels.size() != *expected)
    {
        throw std::invalid_argument(
            "captured frame is empty, malformed, or uses an unsupported format");
    }
}

} // namespace

kfcore::image::BgrImage to_bgr(const CapturedFrame& frame,
                               std::size_t max_image_bytes)
{
    validate_captured(frame);
    if (frame.format == SALTS_VIDEO_CAPTURE_FORMAT_BGRA)
    {
        const std::size_t destination_bytes = checked_image_bytes(
            frame.width, frame.height, 3U, max_image_bytes, "captured BGR image");
        kfcore::image::BgrImage result;
        result.width = frame.width;
        result.height = frame.height;
        result.pixels.resize(destination_bytes);
        const std::size_t pixels = destination_bytes / 3U;
        for (std::size_t index = 0U; index < pixels; ++index)
        {
            result.pixels[index * 3U] = frame.pixels[index * 4U];
            result.pixels[index * 3U + 1U] = frame.pixels[index * 4U + 1U];
            result.pixels[index * 3U + 2U] = frame.pixels[index * 4U + 2U];
        }
        return result;
    }

    kfcore::image::ImageView source {
        frame.pixels.data(), frame.pixels.size(), frame.width, frame.height,
        static_cast<std::size_t>(frame.width), processor_format(frame.format),
        kfcore::image::MemoryKind::Host,
    };
    if (frame.format == SALTS_VIDEO_CAPTURE_FORMAT_RGB24)
    {
        source.row_stride *= 3U;
    }
    return kfcore::image::CpuImageProcessor::copy_bgr(source, max_image_bytes);
}

kfcore::image::BgrImage load_bgr(const std::filesystem::path& path,
                                 std::size_t max_file_bytes,
                                 std::size_t max_image_bytes)
{
    if (path.empty() || max_file_bytes == 0U || max_image_bytes == 0U)
    {
        throw std::invalid_argument("image path and byte limits must be non-empty and positive");
    }
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
    {
        throw std::runtime_error("image is not a readable regular file: " + path.string());
    }
    const std::uintmax_t file_size = std::filesystem::file_size(path, error);
    if (error || file_size == 0U || file_size > max_file_bytes ||
        file_size > static_cast<std::uintmax_t>((std::numeric_limits<int>::max)()))
    {
        throw std::length_error("encoded image is empty or exceeds the configured limit");
    }
    std::vector<std::uint8_t> encoded(static_cast<std::size_t>(file_size));
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(encoded.data()),
                    static_cast<std::streamsize>(encoded.size())))
    {
        throw std::runtime_error("failed to read encoded image: " + path.string());
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    if (stbi_info_from_memory(encoded.data(), static_cast<int>(encoded.size()),
                              &width, &height, &channels) == 0)
    {
        throw std::runtime_error("failed to inspect encoded image: " + path.string());
    }
    const std::size_t decoded_bytes = checked_image_bytes(
        width, height, 3U, max_image_bytes, "decoded image");
    const int inspected_width = width;
    const int inspected_height = height;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> decoded(
        stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()),
                              &width, &height, &channels, 3),
        &stbi_image_free);
    if (!decoded)
    {
        const char* reason = stbi_failure_reason();
        throw std::runtime_error("failed to decode image: " + path.string() +
                                 (reason != nullptr ? " (" + std::string(reason) + ")" : ""));
    }
    if (width != inspected_width || height != inspected_height)
    {
        throw std::runtime_error("decoded image dimensions changed after validation");
    }

    kfcore::image::BgrImage result;
    result.width = width;
    result.height = height;
    result.pixels.assign(decoded.get(), decoded.get() + decoded_bytes);
    for (std::size_t index = 0U; index < decoded_bytes; index += 3U)
    {
        std::swap(result.pixels[index], result.pixels[index + 2U]);
    }
    return result;
}

void save_bgr(const std::filesystem::path& path,
              const kfcore::image::BgrImage& image,
              std::size_t max_image_bytes)
{
    const std::size_t bytes = checked_image_bytes(
        image.width, image.height, 3U, max_image_bytes, "output image");
    if (path.empty() || image.pixels.size() != bytes)
    {
        throw std::invalid_argument("output path or packed BGR image is invalid");
    }
    std::vector<std::uint8_t> rgb = image.pixels;
    for (std::size_t index = 0U; index < rgb.size(); index += 3U)
    {
        std::swap(rgb[index], rgb[index + 2U]);
    }
    const std::string extension = lower_extension(path);
    const std::string native_path = path.string();
    int result = 0;
    if (extension == ".png")
    {
        result = stbi_write_png(native_path.c_str(), image.width, image.height, 3,
                                rgb.data(), image.width * 3);
    }
    else if (extension == ".jpg" || extension == ".jpeg")
    {
        constexpr int kJpegQuality = 95;
        result = stbi_write_jpg(native_path.c_str(), image.width, image.height, 3,
                                rgb.data(), kJpegQuality);
    }
    else if (extension == ".bmp")
    {
        result = stbi_write_bmp(native_path.c_str(), image.width, image.height, 3,
                                rgb.data());
    }
    else
    {
        throw std::invalid_argument("output image extension must be PNG, JPEG, or BMP");
    }
    if (result == 0)
    {
        throw std::runtime_error("failed to write output image: " + path.string());
    }
}

ImageView capture_image_view(const CapturedFrame& frame)
{
    validate_captured(frame);
    std::size_t stride = static_cast<std::size_t>(frame.width);
    if (frame.format == SALTS_VIDEO_CAPTURE_FORMAT_RGB24)
    {
        stride *= 3U;
    }
    return { frame.pixels.data(), frame.width, frame.height, stride,
             yolo_format(frame.format), MemoryKind::Host };
}

ImageView bgr_image_view(const kfcore::image::BgrImage& image)
{
    const std::size_t bytes = checked_image_bytes(
        image.width, image.height, 3U,
        (std::numeric_limits<std::size_t>::max)(), "BGR image view");
    if (image.pixels.size() != bytes)
    {
        throw std::invalid_argument("BGR image storage does not match its dimensions");
    }
    return { image.pixels.data(), image.width, image.height,
             static_cast<std::size_t>(image.width) * 3U,
             PixelFormat::Bgr8, MemoryKind::Host };
}

void mirror_bgr_horizontal(kfcore::image::BgrImage& image)
{
    const std::size_t bytes = checked_image_bytes(
        image.width, image.height, 3U,
        (std::numeric_limits<std::size_t>::max)(), "BGR mirror");
    if (image.pixels.size() != bytes)
    {
        throw std::invalid_argument("BGR mirror storage does not match its dimensions");
    }
    const std::size_t width = static_cast<std::size_t>(image.width);
    for (std::int32_t y = 0; y < image.height; ++y)
    {
        std::uint8_t* row = image.pixels.data() +
            static_cast<std::size_t>(y) * width * 3U;
        for (std::size_t left = 0U, right = width - 1U; left < right;
             ++left, --right)
        {
            for (std::size_t channel = 0U; channel < 3U; ++channel)
            {
                std::swap(row[left * 3U + channel], row[right * 3U + channel]);
            }
        }
    }
}

} // namespace kfcore::yolo::demo
