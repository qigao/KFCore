#include "kfcore/image_processor/image_processor.hpp"
#include "tinytest.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using namespace kfcore::image;

namespace
{

template <typename Callable>
void expect_processor_error(Callable&& callable, ImageProcessorErrorCode code,
                            const char* message_fragment)
{
    try
    {
        callable();
        check(false);
    }
    catch (const ImageProcessorError& error)
    {
        check(error.code() == code);
        check(std::string(error.what()).find(message_fragment) != std::string::npos);
    }
}

ImageView host_view(const void* data, std::int32_t width, std::int32_t height, std::size_t stride,
                    PixelFormat format = PixelFormat::Rgb8)
{
    const std::size_t byte_size =
        (static_cast<std::size_t>(height) - 1U) * stride + static_cast<std::size_t>(width) * 3U;
    return { data, byte_size, width, height, stride, format, MemoryKind::Host };
}

ImageView gray_host_view(const void* data, std::int32_t width, std::int32_t height,
                         std::size_t stride)
{
    const std::size_t byte_size =
        (static_cast<std::size_t>(height) - 1U) * stride + static_cast<std::size_t>(width);
    return { data, byte_size, width, height, stride, PixelFormat::Gray8, MemoryKind::Host };
}

ImageView yuv_host_view(const void* data, std::size_t byte_size, std::int32_t width,
                        std::int32_t height, std::size_t stride, PixelFormat format)
{
    return { data, byte_size, width, height, stride, format, MemoryKind::Host };
}

TensorView device_tensor(void* data, std::size_t bytes, std::int32_t batch, std::int32_t height,
                         std::int32_t width, TensorElementType type = TensorElementType::Float32)
{
    return {
        data, bytes, batch, 3, height, width, type, TensorLayout::Nchw, MemoryKind::CudaDevice
    };
}

} // namespace

spec("ImageProcessor CPU contract")
{
    it("stages padded RGB8 and BGR8 images as deterministic packed grayscale")
    {
        constexpr std::size_t kSourceLimit = 1024;
        const std::array<std::uint8_t, 8> rgb = {
            255, 0, 0, 0, 255, 0, 90, 91,
        };
        const std::array<std::uint8_t, 8> bgr = {
            255, 0, 0, 0, 0, 255, 92, 93,
        };
        std::array<std::uint8_t, 2> rgb_gray {};
        std::array<std::uint8_t, 2> bgr_gray {};

        const ImageView rgb_view = host_view(rgb.data(), 2, 1, rgb.size(), PixelFormat::Rgb8);
        const ImageView bgr_view = host_view(bgr.data(), 2, 1, bgr.size(), PixelFormat::Bgr8);
        check(ImageProcessor::packed_grayscale_bytes(rgb_view, kSourceLimit) == rgb_gray.size());
        check(ImageProcessor::packed_grayscale_bytes(bgr_view, kSourceLimit) == bgr_gray.size());

        ImageProcessor::stage_host_grayscale(
            rgb_view, { rgb_gray.data(), rgb_gray.size() }, kSourceLimit);
        ImageProcessor::stage_host_grayscale(
            bgr_view, { bgr_gray.data(), bgr_gray.size() }, kSourceLimit);

        const std::array<std::uint8_t, 2> expected_rgb = { 76, 150 };
        const std::array<std::uint8_t, 2> expected_bgr = { 29, 76 };
        check(rgb_gray == expected_rgb);
        check(bgr_gray == expected_bgr);
    }

    it("copies padded Gray8 rows without copying padding bytes")
    {
        constexpr std::size_t kSourceLimit = 1024;
        const std::array<std::uint8_t, 8> source = {
            1, 2, 3, 90, 4, 5, 6, 91,
        };
        std::array<std::uint8_t, 6> destination {};
        const ImageView image = gray_host_view(source.data(), 3, 2, 4);

        check(ImageProcessor::packed_grayscale_bytes(image, kSourceLimit) == destination.size());
        ImageProcessor::stage_host_grayscale(
            image, { destination.data(), destination.size() }, kSourceLimit);

        const std::array<std::uint8_t, 6> expected = { 1, 2, 3, 4, 5, 6 };
        check(destination == expected);
    }

    it("rejects invalid grayscale staging contracts before reading input")
    {
        constexpr std::size_t kSourceLimit = 1024;
        std::array<std::uint8_t, 6> pixels {};
        std::array<std::uint8_t, 2> destination {};
        const ImageView valid = host_view(pixels.data(), 2, 1, 6);

        ImageView invalid = valid;
        invalid.memory_kind = MemoryKind::CudaDevice;
        expect_processor_error(
            [&] { (void)ImageProcessor::packed_grayscale_bytes(invalid, kSourceLimit); },
            ImageProcessorErrorCode::InvalidArgument, "Host memory");

        invalid = valid;
        invalid.row_stride = 5;
        expect_processor_error(
            [&] { (void)ImageProcessor::packed_grayscale_bytes(invalid, kSourceLimit); },
            ImageProcessorErrorCode::InvalidArgument, "row stride");

        invalid = valid;
        invalid.byte_size = 5;
        expect_processor_error(
            [&] { (void)ImageProcessor::packed_grayscale_bytes(invalid, kSourceLimit); },
            ImageProcessorErrorCode::InvalidArgument, "image capacity");

        expect_processor_error(
            [&] { (void)ImageProcessor::packed_grayscale_bytes(valid, 5); },
            ImageProcessorErrorCode::ResourceLimitExceeded, "source bytes");

        expect_processor_error(
            [&]
            {
                ImageProcessor::stage_host_grayscale(
                    valid, { destination.data(), destination.size() - 1U }, kSourceLimit);
            },
            ImageProcessorErrorCode::InvalidArgument, "destination capacity");

        expect_processor_error(
            [&]
            {
                ImageProcessor::stage_host_grayscale(
                    valid, { pixels.data(), destination.size() }, kSourceLimit);
            },
            ImageProcessorErrorCode::InvalidArgument, "must not overlap");

        const std::uint8_t pixel = 0;
        const ImageView huge = {
            &pixel,
            (std::numeric_limits<std::size_t>::max)(),
            (std::numeric_limits<std::int32_t>::max)(),
            (std::numeric_limits<std::int32_t>::max)(),
            (std::numeric_limits<std::size_t>::max)(),
            PixelFormat::Gray8,
            MemoryKind::Host,
        };
        expect_processor_error(
            [&]
            {
                (void)ImageProcessor::packed_grayscale_bytes(
                    huge, (std::numeric_limits<std::size_t>::max)());
            },
            ImageProcessorErrorCode::ResourceLimitExceeded, "overflow");
    }

    it("plans a bounded mixed host and CUDA-device batch")
    {
        std::array<std::uint8_t, 24>     host_pixels {};
        std::array<std::uint8_t, 18>     device_placeholder {};
        std::array<float, 3 * 2 * 4 * 2> tensor_placeholder {};
        const std::vector<ImageView>     images = {
            host_view(host_pixels.data(), 2, 3, 8, PixelFormat::Bgr8),
            { device_placeholder.data(), device_placeholder.size(), 3, 2, 9, PixelFormat::Rgb8,
                  MemoryKind::CudaDevice },
        };
        const BatchPlan plan = ImageProcessor::plan(
            images, device_tensor(tensor_placeholder.data(), sizeof(tensor_placeholder), 2, 2, 4),
            1024, 1024);

        check(plan.images.size() == 2);
        check(plan.host_staging_bytes == 18);
        check(plan.device_staging_bytes == 18);
        check(plan.tensor_bytes == sizeof(tensor_placeholder));
        check(plan.images[0].requires_staging);
        check(plan.images[0].staging_offset == 0);
        check(plan.images[0].packed_bytes == 18);
        check(!plan.images[1].requires_staging);
        check(plan.images[0].transform.source_width == 2);
        check(plan.images[0].transform.source_height == 3);
        check(plan.images[0].transform.scale > 0.6666f);
        check(plan.images[0].transform.scale < 0.6667f);
        check(plan.images[0].transform.pad_x > 1.3333f);
        check(plan.images[0].transform.pad_x < 1.3334f);
        check(plan.images[0].transform.pad_y == 0.0f);
    }

    it("packs padded host rows without copying padding bytes")
    {
        constexpr std::uint8_t             kSentinel = 0xee;
        const std::array<std::uint8_t, 16> source    = {
            1, 2, 3, 4, 5, 6, 90, 91, 7, 8, 9, 10, 11, 12, 92, 93,
        };
        std::array<std::uint8_t, 12> packed {};
        packed.fill(kSentinel);
        std::array<float, 12>        output_placeholder {};
        const std::vector<ImageView> images = { host_view(source.data(), 2, 2, 8) };
        const BatchPlan              plan   = ImageProcessor::plan(
            images, device_tensor(output_placeholder.data(), sizeof(output_placeholder), 1, 2, 2),
            1024, 1024);

        ImageProcessor::stage_host_inputs(images, plan, { packed.data(), packed.size() });

        const std::array<std::uint8_t, 12> expected = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        };
        check(packed == expected);
    }

    it("plans and packs padded NV12 and I420 planes without padding")
    {
        constexpr std::uint8_t kPadding = 0xeeU;
        const std::array<std::uint8_t, 16> nv12 = {
            1, 2, 3, 4, kPadding, kPadding,
            5, 6, 7, 8, kPadding, kPadding,
            9, 10, 11, 12,
        };
        const std::array<std::uint8_t, 17> i420 = {
            1, 2, 3, 4, kPadding, kPadding,
            5, 6, 7, 8, kPadding, kPadding,
            9, 10, kPadding,
            11, 12,
        };
        std::array<float, 3 * 2 * 4> tensor_placeholder {};
        const TensorView tensor = device_tensor(
            tensor_placeholder.data(), sizeof(tensor_placeholder), 1, 2, 4);

        for (const ImageView image : {
                 yuv_host_view(nv12.data(), nv12.size(), 4, 2, 6, PixelFormat::Nv12),
                 yuv_host_view(i420.data(), i420.size(), 4, 2, 6, PixelFormat::I420),
             })
        {
            const BatchPlan plan = ImageProcessor::plan({ image }, tensor, 1024, 1024);
            check(plan.images.size() == 1U);
            check(plan.images[0].source_span_bytes == image.byte_size);
            check(plan.images[0].packed_bytes == 12U);
            check(plan.host_staging_bytes == 12U);

            std::array<std::uint8_t, 12> packed {};
            ImageProcessor::stage_host_inputs(
                { image }, plan, { packed.data(), packed.size() });
            const std::array<std::uint8_t, 12> expected = {
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            };
            check(packed == expected);
        }
    }

    it("rejects malformed packed YUV dimensions strides and capacities")
    {
        std::array<std::uint8_t, 12> pixels {};
        std::array<float, 3 * 2 * 4> tensor_placeholder {};
        const TensorView tensor = device_tensor(
            tensor_placeholder.data(), sizeof(tensor_placeholder), 1, 2, 4);
        const ImageView valid = yuv_host_view(
            pixels.data(), pixels.size(), 4, 2, 4, PixelFormat::Nv12);

        ImageView invalid = valid;
        invalid.width = 3;
        expect_processor_error(
            [&] { (void)ImageProcessor::plan({ invalid }, tensor, 1024, 1024); },
            ImageProcessorErrorCode::InvalidArgument, "even");

        invalid = valid;
        invalid.height = 3;
        expect_processor_error(
            [&] { (void)ImageProcessor::plan({ invalid }, tensor, 1024, 1024); },
            ImageProcessorErrorCode::InvalidArgument, "even");

        invalid = valid;
        invalid.row_stride = 3;
        expect_processor_error(
            [&] { (void)ImageProcessor::plan({ invalid }, tensor, 1024, 1024); },
            ImageProcessorErrorCode::InvalidArgument, "row stride");

        invalid = valid;
        invalid.byte_size = pixels.size() - 1U;
        expect_processor_error(
            [&] { (void)ImageProcessor::plan({ invalid }, tensor, 1024, 1024); },
            ImageProcessorErrorCode::InvalidArgument, "image capacity");
    }

    it("rejects invalid images, tensor contracts, limits, and workspace capacity")
    {
        std::array<std::uint8_t, 12> pixels {};
        std::array<float, 12>        output_placeholder {};
        const TensorView             output =
            device_tensor(output_placeholder.data(), sizeof(output_placeholder), 1, 2, 2);
        const ImageView valid = host_view(pixels.data(), 2, 2, 6);

        expect_processor_error([&] { (void)ImageProcessor::plan({}, output, 1024, 1024); },
                               ImageProcessorErrorCode::InvalidArgument, "batch must not be empty");

        ImageView invalid    = valid;
        invalid.pixel_format = static_cast<PixelFormat>(99);
        expect_processor_error([&] { (void)ImageProcessor::plan({ invalid }, output, 1024, 1024); },
                               ImageProcessorErrorCode::InvalidArgument, "pixel format");

        invalid           = valid;
        invalid.byte_size = 11;
        expect_processor_error([&] { (void)ImageProcessor::plan({ invalid }, output, 1024, 1024); },
                               ImageProcessorErrorCode::InvalidArgument, "image capacity");

        invalid            = valid;
        invalid.row_stride = 5;
        expect_processor_error([&] { (void)ImageProcessor::plan({ invalid }, output, 1024, 1024); },
                               ImageProcessorErrorCode::InvalidArgument, "row stride");

        TensorView bad_tensor = output;
        bad_tensor.layout     = static_cast<TensorLayout>(99);
        expect_processor_error([&]
                               { (void)ImageProcessor::plan({ valid }, bad_tensor, 1024, 1024); },
                               ImageProcessorErrorCode::InvalidArgument, "tensor layout");

        bad_tensor = output;
        bad_tensor.byte_size -= sizeof(float);
        expect_processor_error([&]
                               { (void)ImageProcessor::plan({ valid }, bad_tensor, 1024, 1024); },
                               ImageProcessorErrorCode::InvalidArgument, "tensor capacity");

        expect_processor_error([&] { (void)ImageProcessor::plan({ valid }, output, 11, 1024); },
                               ImageProcessorErrorCode::ResourceLimitExceeded, "source bytes");

        const BatchPlan              plan = ImageProcessor::plan({ valid }, output, 1024, 1024);
        std::array<std::uint8_t, 11> too_small {};
        expect_processor_error(
            [&]
            {
                ImageProcessor::stage_host_inputs({ valid }, plan,
                                                  { too_small.data(), too_small.size() });
            },
            ImageProcessorErrorCode::InvalidArgument, "workspace capacity");

        ImageView changed_after_planning  = valid;
        changed_after_planning.row_stride = 5;
        std::array<std::uint8_t, 12> workspace {};
        expect_processor_error(
            [&]
            {
                ImageProcessor::stage_host_inputs({ changed_after_planning }, plan,
                                                  { workspace.data(), workspace.size() });
            },
            ImageProcessorErrorCode::InvalidArgument, "row stride");

        std::array<std::uint8_t, 15> shape_pixels {};
        const ImageView              original_shape = host_view(shape_pixels.data(), 1, 4, 4);
        const BatchPlan shape_plan = ImageProcessor::plan({ original_shape }, output, 1024, 1024);
        ImageView       changed_shape = original_shape;
        changed_shape.width           = 2;
        changed_shape.height          = 2;
        changed_shape.row_stride      = 9;
        expect_processor_error(
            [&]
            {
                ImageProcessor::stage_host_inputs({ changed_shape }, shape_plan,
                                                  { workspace.data(), workspace.size() });
            },
            ImageProcessorErrorCode::InvalidArgument, "image dimensions");
    }

    it("rejects byte-count overflow before reading input memory")
    {
        const std::uint8_t   pixel = 0;
        std::array<float, 3> output_placeholder {};
        const ImageView      huge = {
            &pixel,
            (std::numeric_limits<std::size_t>::max)(),
            (std::numeric_limits<std::int32_t>::max)(),
            (std::numeric_limits<std::int32_t>::max)(),
            (std::numeric_limits<std::size_t>::max)(),
            PixelFormat::Rgb8,
            MemoryKind::Host,
        };
        const TensorView output =
            device_tensor(output_placeholder.data(), sizeof(output_placeholder), 1, 1, 1);

        expect_processor_error(
            [&]
            {
                (void)ImageProcessor::plan({ huge }, output,
                                           (std::numeric_limits<std::size_t>::max)(), 1024);
            },
            ImageProcessorErrorCode::ResourceLimitExceeded, "overflow");
    }
}
