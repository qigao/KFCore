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
