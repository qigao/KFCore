#include "hand_appearance.hpp"

#include "kfcore/vision_models/error.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace kfcore::vision_models::detail
{
namespace
{

constexpr std::size_t kPalmRows = 6U;
constexpr std::size_t kPalmColumns = 8U;
constexpr std::size_t kFingerRows = 8U;
constexpr std::size_t kFingerColumns = 2U;
constexpr float kTextureStandardDeviationScale = 3.0F;
constexpr float kMinimumStandardDeviation = 1.0e-4F;
constexpr float kColorSumEpsilon = 1.0e-6F;
constexpr float kFingerHalfWidthPalmRatio = 0.055F;
constexpr float kMinimumFingerLengthPalmRatio = 0.15F;
constexpr std::array<std::size_t, 5U> kPalmIndices = { 0U, 5U, 9U, 13U, 17U };
constexpr std::array<std::array<std::size_t, 4U>, 5U> kFingerChains = {{
    {{1U, 2U, 3U, 4U}},
    {{5U, 6U, 7U, 8U}},
    {{9U, 10U, 11U, 12U}},
    {{13U, 14U, 15U, 16U}},
    {{17U, 18U, 19U, 20U}},
}};

static_assert(kPalmRows * kPalmColumns * 2U ==
              kPalmAppearanceFeatureCount);
static_assert(kFingerRows * kFingerColumns * 2U ==
              kFingerAppearanceFeatureCount);

struct RgbSample
{
    float red = 0.0F;
    float green = 0.0F;
    float blue = 0.0F;
    bool valid = false;
};

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw VisionModelError(VisionModelErrorCode::InvalidArgument,
                           "hand appearance stage: " + detail);
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char* description)
{
    if (left != 0U && right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_invalid(std::string(description) + " byte count overflows size_t");
    }
    return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right,
                        const char* description)
{
    if (right > (std::numeric_limits<std::size_t>::max)() - left)
    {
        throw_invalid(std::string(description) + " byte count overflows size_t");
    }
    return left + right;
}

bool finite(const HandLandmark& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) &&
           std::isfinite(point.z);
}

RgbSample sample_bilinear(const image::ImageView& image, float x, float y)
{
    if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0F || y < 0.0F ||
        x > static_cast<float>(image.width - 1) ||
        y > static_cast<float>(image.height - 1))
    {
        return {};
    }
    const std::int32_t x0 = static_cast<std::int32_t>(std::floor(x));
    const std::int32_t y0 = static_cast<std::int32_t>(std::floor(y));
    const std::int32_t x1 = std::min(x0 + 1, image.width - 1);
    const std::int32_t y1 = std::min(y0 + 1, image.height - 1);
    const float wx = x - static_cast<float>(x0);
    const float wy = y - static_cast<float>(y0);
    const auto pixel = [&](std::int32_t px, std::int32_t py) {
        const auto* bytes = static_cast<const std::uint8_t*>(image.data);
        if (image.pixel_format == image::PixelFormat::Nv12 ||
            image.pixel_format == image::PixelFormat::I420 ||
            image.pixel_format == image::PixelFormat::Nv21 ||
            image.pixel_format == image::PixelFormat::Yuy2 ||
            image.pixel_format == image::PixelFormat::Uyvy)
        {
            const std::size_t sx = static_cast<std::size_t>(px);
            const std::size_t sy = static_cast<std::size_t>(py);
            const std::size_t height = static_cast<std::size_t>(image.height);
            int y_value = 0;
            int u_value = 0;
            int v_value = 0;
            if (image.pixel_format == image::PixelFormat::Yuy2 ||
                image.pixel_format == image::PixelFormat::Uyvy)
            {
                const auto* pair = bytes + sy * image.row_stride + (sx / 2U) * 4U;
                if (image.pixel_format == image::PixelFormat::Yuy2)
                {
                    y_value = pair[(sx & 1U) == 0U ? 0U : 2U];
                    u_value = pair[1];
                    v_value = pair[3];
                }
                else
                {
                    y_value = pair[(sx & 1U) == 0U ? 1U : 3U];
                    u_value = pair[0];
                    v_value = pair[2];
                }
            }
            else
            {
                const std::size_t y_storage = image.row_stride * height;
                y_value = bytes[sy * image.row_stride + sx];
                if (image.pixel_format == image::PixelFormat::Nv12 ||
                    image.pixel_format == image::PixelFormat::Nv21)
                {
                    const std::size_t offset = y_storage + (sy / 2U) * image.row_stride +
                                               (sx / 2U) * 2U;
                    const bool uv_order = image.pixel_format == image::PixelFormat::Nv12;
                    u_value = bytes[offset + (uv_order ? 0U : 1U)];
                    v_value = bytes[offset + (uv_order ? 1U : 0U)];
                }
                else
                {
                    const std::size_t chroma_stride = image.row_stride / 2U;
                    const std::size_t chroma_rows = height / 2U;
                    const std::size_t offset = (sy / 2U) * chroma_stride + sx / 2U;
                    u_value = bytes[y_storage + offset];
                    v_value = bytes[y_storage + chroma_stride * chroma_rows + offset];
                }
            }
            const int c = std::max(0, y_value - 16);
            const int d = u_value - 128;
            const int e = v_value - 128;
            const auto channel = [](int value) {
                return static_cast<float>(std::clamp(value, 0, 255));
            };
            return RgbSample {
                channel((298 * c + 409 * e + 128) / 256),
                channel((298 * c - 100 * d - 208 * e + 128) / 256),
                channel((298 * c + 516 * d + 128) / 256), true,
            };
        }
        const std::size_t offset = static_cast<std::size_t>(py) * image.row_stride +
                                   static_cast<std::size_t>(px) * 3U;
        const bool bgr = image.pixel_format == image::PixelFormat::Bgr8;
        return RgbSample {
            static_cast<float>(bytes[offset + (bgr ? 2U : 0U)]),
            static_cast<float>(bytes[offset + 1U]),
            static_cast<float>(bytes[offset + (bgr ? 0U : 2U)]), true,
        };
    };
    const RgbSample top_left = pixel(x0, y0);
    const RgbSample top_right = pixel(x1, y0);
    const RgbSample bottom_left = pixel(x0, y1);
    const RgbSample bottom_right = pixel(x1, y1);
    const auto interpolate = [&](float RgbSample::*component) {
        const float top = top_left.*component * (1.0F - wx) +
                          top_right.*component * wx;
        const float bottom = bottom_left.*component * (1.0F - wx) +
                             bottom_right.*component * wx;
        return top * (1.0F - wy) + bottom * wy;
    };
    return { interpolate(&RgbSample::red), interpolate(&RgbSample::green),
             interpolate(&RgbSample::blue), true };
}

template <std::size_t Rows, std::size_t Columns>
bool encode_part(const std::array<RgbSample, Rows * Columns>& samples,
                 HandAppearancePart part, float minimum_coverage,
                 HandAppearanceDescriptor& descriptor)
{
    static_assert(Columns % 2U == 0U);
    const std::size_t valid_count = static_cast<std::size_t>(std::count_if(
        samples.begin(), samples.end(), [](const RgbSample& sample) {
            return sample.valid;
        }));
    const float coverage = static_cast<float>(valid_count) /
                           static_cast<float>(samples.size());
    if (valid_count == 0U || coverage < minimum_coverage)
    {
        return false;
    }

    std::array<float, Rows * Columns> luminance {};
    float luminance_sum = 0.0F;
    for (std::size_t index = 0U; index < samples.size(); ++index)
    {
        if (!samples[index].valid)
        {
            continue;
        }
        luminance[index] = 0.2126F * samples[index].red +
                           0.7152F * samples[index].green +
                           0.0722F * samples[index].blue;
        luminance_sum += luminance[index];
    }
    const float mean = luminance_sum / static_cast<float>(valid_count);
    float squared_sum = 0.0F;
    for (std::size_t index = 0U; index < samples.size(); ++index)
    {
        if (samples[index].valid)
        {
            const float delta = luminance[index] - mean;
            squared_sum += delta * delta;
        }
    }
    const float standard_deviation =
        std::sqrt(squared_sum / static_cast<float>(valid_count));
    const float denominator =
        std::max(standard_deviation, kMinimumStandardDeviation) *
        kTextureStandardDeviationScale;

    const std::size_t feature_offset = hand_appearance_feature_offset(part);
    const std::size_t sample_count = samples.size();
    const std::size_t chroma_count = sample_count / 2U;
    for (std::size_t index = 0U; index < samples.size(); ++index)
    {
        descriptor.values[feature_offset + index] = samples[index].valid
            ? std::clamp((luminance[index] - mean) / denominator, -1.0F, 1.0F)
            : 0.0F;
    }
    for (std::size_t row = 0U; row < Rows; ++row)
    {
        for (std::size_t pair = 0U; pair < Columns / 2U; ++pair)
        {
            float red_chroma = 0.0F;
            float green_chroma = 0.0F;
            std::size_t chroma_samples = 0U;
            for (std::size_t column_offset = 0U; column_offset < 2U;
                 ++column_offset)
            {
                const RgbSample& sample =
                    samples[row * Columns + pair * 2U + column_offset];
                const float sum = sample.red + sample.green + sample.blue;
                if (sample.valid && sum > kColorSumEpsilon)
                {
                    red_chroma += sample.red / sum;
                    green_chroma += sample.green / sum;
                    ++chroma_samples;
                }
            }
            const std::size_t cell = row * (Columns / 2U) + pair;
            if (chroma_samples > 0U)
            {
                descriptor.values[feature_offset + sample_count + cell] =
                    red_chroma / static_cast<float>(chroma_samples);
                descriptor.values[feature_offset + sample_count + chroma_count +
                                  cell] =
                    green_chroma / static_cast<float>(chroma_samples);
            }
        }
    }
    const std::size_t part_index = static_cast<std::size_t>(part);
    descriptor.valid_parts |= hand_appearance_part_bit(part);
    descriptor.quality[part_index] = coverage;
    return true;
}

std::array<RgbSample, kPalmRows * kPalmColumns> sample_palm(
    const image::ImageView& image,
    const std::array<HandLandmark, kHandLandmarkCount>& landmarks,
    float center_x, float center_y)
{
    const float width_x = landmarks[17].x - landmarks[5].x;
    const float width_y = landmarks[17].y - landmarks[5].y;
    const float height_x = landmarks[0].x - landmarks[9].x;
    const float height_y = landmarks[0].y - landmarks[9].y;
    std::array<RgbSample, kPalmRows * kPalmColumns> samples {};
    for (std::size_t row = 0U; row < kPalmRows; ++row)
    {
        const float v = (static_cast<float>(row) + 0.5F) /
                            static_cast<float>(kPalmRows) -
                        0.5F;
        for (std::size_t column = 0U; column < kPalmColumns; ++column)
        {
            const float u = (static_cast<float>(column) + 0.5F) /
                                static_cast<float>(kPalmColumns) -
                            0.5F;
            samples[row * kPalmColumns + column] = sample_bilinear(
                image, center_x + u * width_x + v * height_x,
                center_y + u * width_y + v * height_y);
        }
    }
    return samples;
}

std::array<RgbSample, kFingerRows * kFingerColumns> sample_finger(
    const image::ImageView& image,
    const std::array<HandLandmark, kHandLandmarkCount>& landmarks,
    const std::array<std::size_t, 4U>& chain, float half_width,
    float minimum_length)
{
    std::array<RgbSample, kFingerRows * kFingerColumns> samples {};
    std::array<float, 3U> lengths {};
    float total_length = 0.0F;
    for (std::size_t segment = 0U; segment < lengths.size(); ++segment)
    {
        const HandLandmark& first = landmarks[chain[segment]];
        const HandLandmark& last = landmarks[chain[segment + 1U]];
        if (!finite(first) || !finite(last))
        {
            return samples;
        }
        lengths[segment] = std::hypot(last.x - first.x, last.y - first.y);
        if (!std::isfinite(lengths[segment]))
        {
            return samples;
        }
        total_length += lengths[segment];
    }
    if (total_length < minimum_length)
    {
        return samples;
    }

    for (std::size_t row = 0U; row < kFingerRows; ++row)
    {
        const float target = (static_cast<float>(row) + 0.5F) /
                             static_cast<float>(kFingerRows) * total_length;
        float preceding = 0.0F;
        std::size_t segment = 0U;
        while (segment + 1U < lengths.size() &&
               target > preceding + lengths[segment])
        {
            preceding += lengths[segment++];
        }
        if (lengths[segment] <= 0.0F)
        {
            continue;
        }
        const HandLandmark& first = landmarks[chain[segment]];
        const HandLandmark& last = landmarks[chain[segment + 1U]];
        const float fraction = std::clamp(
            (target - preceding) / lengths[segment], 0.0F, 1.0F);
        const float center_x = first.x + (last.x - first.x) * fraction;
        const float center_y = first.y + (last.y - first.y) * fraction;
        const float tangent_x = (last.x - first.x) / lengths[segment];
        const float tangent_y = (last.y - first.y) / lengths[segment];
        for (std::size_t column = 0U; column < kFingerColumns; ++column)
        {
            const float side = column == 0U ? -0.5F : 0.5F;
            samples[row * kFingerColumns + column] = sample_bilinear(
                image, center_x - tangent_y * half_width * side,
                center_y + tangent_x * half_width * side);
        }
    }
    return samples;
}

} // namespace

void validate_hand_appearance_source(const image::ImageView& image)
{
    if (image.memory_kind != image::MemoryKind::Host ||
        (image.pixel_format != image::PixelFormat::Bgr8 &&
         image.pixel_format != image::PixelFormat::Rgb8 &&
         image.pixel_format != image::PixelFormat::Nv12 &&
         image.pixel_format != image::PixelFormat::I420 &&
         image.pixel_format != image::PixelFormat::Nv21 &&
         image.pixel_format != image::PixelFormat::Yuy2 &&
         image.pixel_format != image::PixelFormat::Uyvy))
    {
        throw_invalid(
            "built-in extraction requires a host BGR8/RGB8/NV12/I420/NV21/YUY2/UYVY image");
    }
    const std::size_t width = static_cast<std::size_t>(image.width);
    const std::size_t height = static_cast<std::size_t>(image.height);
    const bool yuv420 = image.pixel_format == image::PixelFormat::Nv12 ||
                        image.pixel_format == image::PixelFormat::I420 ||
                        image.pixel_format == image::PixelFormat::Nv21;
    const bool yuv422 = image.pixel_format == image::PixelFormat::Yuy2 ||
                        image.pixel_format == image::PixelFormat::Uyvy;
    if (yuv420 && ((image.width & 1) != 0 || (image.height & 1) != 0))
    {
        throw_invalid("NV12/I420/NV21 dimensions must be even");
    }
    if (yuv422 && (image.width & 1) != 0)
    {
        throw_invalid("YUY2/UYVY width must be even");
    }
    const std::size_t bytes_per_pixel = yuv422 ? 2U : 3U;
    if ((!yuv420 && width > (std::numeric_limits<std::size_t>::max)() / bytes_per_pixel) ||
        (yuv420 && (image.row_stride < width || (image.row_stride & 1U) != 0U)))
    {
        throw_invalid(yuv420 ? "YUV420 row_stride must be even and at least image width"
                             : "image row byte count overflows size_t");
    }
    const std::size_t row_bytes = yuv420 ? width : width * bytes_per_pixel;
    if (image.row_stride < row_bytes)
    {
        throw_invalid("row_stride is smaller than one packed image row");
    }
    if (height > 1U &&
        image.row_stride > ((std::numeric_limits<std::size_t>::max)() - row_bytes) /
                               (height - 1U))
    {
        throw_invalid("image byte_size calculation overflows size_t");
    }
    std::size_t required = checked_add(
        checked_multiply(height - 1U, image.row_stride, "image"), row_bytes, "image");
    if (yuv420)
    {
        const std::size_t chroma_rows = height / 2U;
        const std::size_t y_storage = checked_multiply(image.row_stride, height, "Y plane");
        if (image.pixel_format == image::PixelFormat::Nv12 ||
            image.pixel_format == image::PixelFormat::Nv21)
        {
            required = checked_add(
                y_storage,
                checked_add(checked_multiply(chroma_rows - 1U, image.row_stride,
                                              "NV12 chroma plane"),
                            width, "NV12 chroma plane"),
                "NV12 image");
        }
        else
        {
            const std::size_t chroma_stride = image.row_stride / 2U;
            required = checked_add(
                checked_add(y_storage,
                            checked_multiply(chroma_stride, chroma_rows, "I420 U plane"),
                            "I420 image"),
                checked_add(checked_multiply(chroma_rows - 1U, chroma_stride,
                                              "I420 V plane"),
                            width / 2U, "I420 V plane"),
                "I420 image");
        }
    }
    if (image.byte_size < required)
    {
        throw_invalid("image byte_size is smaller than the declared rows");
    }
}

std::optional<HandAppearanceDescriptor> make_hand_appearance_descriptor(
    const image::ImageView& image,
    const std::array<HandLandmark, kHandLandmarkCount>& landmarks,
    const HandAppearanceOptions& options)
{
    for (std::size_t index : kPalmIndices)
    {
        if (!finite(landmarks[index]))
        {
            return std::nullopt;
        }
    }
    float center_x = 0.0F;
    float center_y = 0.0F;
    for (std::size_t index : kPalmIndices)
    {
        center_x += landmarks[index].x;
        center_y += landmarks[index].y;
    }
    center_x /= static_cast<float>(kPalmIndices.size());
    center_y /= static_cast<float>(kPalmIndices.size());
    const float width_span = std::hypot(landmarks[17].x - landmarks[5].x,
                                        landmarks[17].y - landmarks[5].y);
    const float height_span = std::hypot(landmarks[0].x - landmarks[9].x,
                                         landmarks[0].y - landmarks[9].y);
    if (!std::isfinite(width_span) || !std::isfinite(height_span) ||
        width_span < options.minimum_palm_span_pixels ||
        height_span < options.minimum_palm_span_pixels)
    {
        return std::nullopt;
    }

    HandAppearanceDescriptor descriptor;
    const auto palm_samples = sample_palm(image, landmarks, center_x, center_y);
    (void)encode_part<kPalmRows, kPalmColumns>(
        palm_samples, HandAppearancePart::Palm,
        options.minimum_part_in_frame_sample_ratio, descriptor);

    const float half_width = width_span * kFingerHalfWidthPalmRatio;
    const float minimum_length = width_span * kMinimumFingerLengthPalmRatio;
    for (std::size_t finger = 0U; finger < kFingerChains.size(); ++finger)
    {
        const auto samples = sample_finger(image, landmarks, kFingerChains[finger],
                                           half_width, minimum_length);
        const auto part = static_cast<HandAppearancePart>(finger + 1U);
        (void)encode_part<kFingerRows, kFingerColumns>(
            samples, part, options.minimum_part_in_frame_sample_ratio, descriptor);
    }
    return descriptor.valid_parts != 0U
        ? std::optional<HandAppearanceDescriptor>(descriptor)
        : std::nullopt;
}

} // namespace kfcore::vision_models::detail
