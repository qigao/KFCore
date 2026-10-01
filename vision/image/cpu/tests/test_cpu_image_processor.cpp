#include "kfcore/image_processor/cpu.hpp"
#include "tinytest.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace kfcore::image;

namespace
{

bool approximately_equal(float actual, float expected, float tolerance = 1.0e-6F)
{
    return std::fabs(actual - expected) <= tolerance;
}

} // namespace

spec("CPU image processor")
{
    it("copies padded borrowed BGR rows into owned packed storage")
    {
        const std::array<std::uint8_t, 16> source = {
            1, 2, 3, 4, 5, 6, 90, 91,
            7, 8, 9, 10, 11, 12, 92, 93,
        };
        const ImageView view { source.data(), source.size(), 2, 2, 8,
                               PixelFormat::Bgr8, MemoryKind::Host };

        const BgrImage image = CpuImageProcessor::copy_bgr(view, 1024);

        const std::vector<std::uint8_t> expected = {
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12,
        };
        check(image.width == 2);
        check(image.height == 2);
        check_eq_container(image.pixels, expected);
        check(image.view().row_stride == std::size_t { 6 });
    }

    it("warps BGR pixels with destination-to-source affine sampling")
    {
        BgrImage source;
        source.width  = 2;
        source.height = 2;
        source.pixels = {
            10, 20, 30, 40, 50, 60,
            70, 80, 90, 100, 110, 120,
        };
        AffineTransform destination_to_source;
        destination_to_source.destination_to_source = { 1.0F, 0.0F, -1.0F,
                                                        0.0F, 1.0F, 0.0F };

        const BgrImage shifted = CpuImageProcessor::warp_affine_bgr(
            source, 3, 2, destination_to_source, 0.0F, 1024);

        const std::vector<std::uint8_t> expected = {
            0, 0, 0, 10, 20, 30, 40, 50, 60,
            0, 0, 0, 70, 80, 90, 100, 110, 120,
        };
        check_eq_container(shifted.pixels, expected);
    }

    it("converts BGR to normalized RGB planar floats")
    {
        BgrImage image;
        image.width  = 1;
        image.height = 1;
        image.pixels = { 0, 127, 255 };
        PreprocessOptions options;
        options.output_format = PixelFormat::Rgb8;
        options.mean           = { 0.5F, 0.5F, 0.5F };
        options.stddev         = { 0.5F, 0.5F, 0.5F };

        const std::vector<float> tensor = CpuImageProcessor::to_nchw(image, options, 1024);

        check(tensor.size() == std::size_t { 3 });
        check_true(approximately_equal(tensor[0], 1.0F));
        check_true(approximately_equal(tensor[1], 127.0F / 127.5F - 1.0F));
        check_true(approximately_equal(tensor[2], -1.0F));
    }

    it("letterboxes host BGR into the same normalized tensor contract as CUDA")
    {
        const std::array<std::uint8_t, 6> source = { 0, 0, 255, 0, 255, 0 };
        const ImageView view { source.data(), source.size(), 2, 1, 6,
                               PixelFormat::Bgr8, MemoryKind::Host };
        PreprocessOptions options;
        options.output_format = PixelFormat::Rgb8;
        options.border_value  = 114.0F;
        LetterboxTransform transform;

        const std::vector<float> tensor = CpuImageProcessor::letterbox_nchw(
            view, 2, 2, options, 1024, 1024, &transform);

        check(tensor.size() == std::size_t { 12 });
        check_true(approximately_equal(transform.scale, 1.0F));
        check_true(approximately_equal(transform.pad_y, 0.5F));
        check_true(approximately_equal(tensor[0], 1.0F));
        check_true(approximately_equal(tensor[1], 0.0F));
        check_true(approximately_equal(tensor[2], 114.0F / 255.0F));
        check_true(approximately_equal(tensor[3], 114.0F / 255.0F));
    }

    it("supports top-left letterbox placement for detector contracts")
    {
        const std::array<std::uint8_t, 6> source = {
            10, 20, 30, 40, 50, 60
        };
        const ImageView view {
            source.data(), source.size(), 2, 1, 6,
            PixelFormat::Bgr8, MemoryKind::Host
        };
        PreprocessOptions options;
        options.output_format = PixelFormat::Bgr8;
        options.center_letterbox = false;
        options.stddev = {
            1.0F / 255.0F,
            1.0F / 255.0F,
            1.0F / 255.0F,
        };
        LetterboxTransform transform;

        const std::vector<float> tensor =
            CpuImageProcessor::letterbox_nchw(
                view, 2, 2, options,
                1024, 1024, &transform);

        check_true(approximately_equal(transform.scale, 1.0F));
        check_true(approximately_equal(transform.pad_x, 0.0F));
        check_true(approximately_equal(transform.pad_y, 0.0F));
        check_true(approximately_equal(tensor[0], 10.0F, 1.0e-4F));
        check_true(approximately_equal(tensor[1], 40.0F, 1.0e-4F));
        check_true(approximately_equal(tensor[2], 114.0F, 1.0e-4F));
        check_true(approximately_equal(tensor[3], 114.0F, 1.0e-4F));
        check_true(approximately_equal(tensor[4], 20.0F, 1.0e-4F));
        check_true(approximately_equal(tensor[8], 30.0F, 1.0e-4F));
    }

    it("converts packed NV12 and I420 red pixels to identical BGR and RGB tensors")
    {
        const std::array<std::uint8_t, 6> nv12 = {
            81, 81, 81, 81, 90, 240,
        };
        const std::array<std::uint8_t, 6> i420 = {
            81, 81, 81, 81, 90, 240,
        };
        const std::array<ImageView, 2> views = {
            ImageView { nv12.data(), nv12.size(), 2, 2, 2,
                        PixelFormat::Nv12, MemoryKind::Host },
            ImageView { i420.data(), i420.size(), 2, 2, 2,
                        PixelFormat::I420, MemoryKind::Host },
        };

        std::vector<float> reference;
        for (const ImageView& view : views)
        {
            const BgrImage bgr = CpuImageProcessor::copy_bgr(view, 1024);
            check_eq_container(bgr.pixels,
                               std::vector<std::uint8_t>({ 0, 0, 255, 0, 0, 255,
                                                           0, 0, 255, 0, 0, 255 }));

            PreprocessOptions options;
            options.output_format = PixelFormat::Rgb8;
            LetterboxTransform transform;
            const std::vector<float> tensor = CpuImageProcessor::letterbox_nchw(
                view, 2, 2, options, 1024, 1024, &transform);
            check_true(approximately_equal(tensor[0], 1.0F));
            check_true(approximately_equal(tensor[4], 0.0F));
            check_true(approximately_equal(tensor[8], 0.0F));
            if (reference.empty())
            {
                reference = tensor;
            }
            else
            {
                check_eq_container(tensor, reference);
            }
        }
    }

    it("converts NV21 VU chroma into BGR pixels")
    {
        const std::array<std::uint8_t, 6> nv21 = {
            81, 81, 81, 81, 240, 90,
        };
        const ImageView view { nv21.data(), nv21.size(), 2, 2, 2,
                               PixelFormat::Nv21, MemoryKind::Host };

        const BgrImage bgr = CpuImageProcessor::copy_bgr(view, 1024);

        check_eq_container(bgr.pixels,
                           std::vector<std::uint8_t>({ 0, 0, 255, 0, 0, 255,
                                                       0, 0, 255, 0, 0, 255 }));
    }

    it("converts packed YUY2 and UYVY 422 pairs into BGR pixels")
    {
        const std::array<std::uint8_t, 4> yuy2 = { 81, 90, 81, 240 };
        const std::array<std::uint8_t, 4> uyvy = { 90, 81, 240, 81 };
        const std::array<ImageView, 2> views = {
            ImageView { yuy2.data(), yuy2.size(), 2, 1, 4,
                        PixelFormat::Yuy2, MemoryKind::Host },
            ImageView { uyvy.data(), uyvy.size(), 2, 1, 4,
                        PixelFormat::Uyvy, MemoryKind::Host },
        };

        for (const ImageView& view : views)
        {
            const BgrImage bgr = CpuImageProcessor::copy_bgr(view, 1024);
            check_eq_container(bgr.pixels,
                               std::vector<std::uint8_t>({ 0, 0, 255, 0, 0, 255 }));
        }
    }

    it("mirrors packed NV12 in fused letterbox source coordinates")
    {
        const std::array<std::uint8_t, 12> nv12 = {
            16, 235, 81, 145,
            16, 235, 81, 145,
            128, 128, 128, 128,
        };
        const ImageView view { nv12.data(), nv12.size(), 4, 2, 4,
                               PixelFormat::Nv12, MemoryKind::Host };
        PreprocessOptions options;
        options.output_format = PixelFormat::Rgb8;
        options.mirror_horizontal = true;
        LetterboxTransform transform;

        const std::vector<float> tensor = CpuImageProcessor::letterbox_nchw(
            view, 4, 2, options, 1024, 1024, &transform);

        check_true(approximately_equal(tensor[0], 150.0F / 255.0F));
        check_true(approximately_equal(tensor[1], 76.0F / 255.0F));
        check_true(approximately_equal(tensor[2], 1.0F));
        check_true(approximately_equal(tensor[3], 0.0F));
        check_true(approximately_equal(tensor[4], tensor[0]));
        check_true(approximately_equal(tensor[8], tensor[0]));
        check_true(approximately_equal(tensor[16], tensor[0]));
    }

    it("rejects device input and malformed owned storage")
    {
        const std::array<std::uint8_t, 3> source = { 0, 0, 0 };
        const ImageView device { source.data(), source.size(), 1, 1, 3,
                                 PixelFormat::Bgr8, MemoryKind::CudaDevice };
        check_throws_as(CpuImageProcessor::copy_bgr(device, 1024), ImageProcessorError);

        BgrImage malformed;
        malformed.width  = 2;
        malformed.height = 2;
        malformed.pixels.resize(3);
        check_throws_as(CpuImageProcessor::to_nchw(malformed, {}, 1024),
                        ImageProcessorError);
    }
}
