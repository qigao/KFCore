#include "kfcore/image_processor/image_processor.hpp"
#include "kfcore/image_processor/cpu.hpp"
#include "tinytest.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace kfcore::image;

namespace
{

class DeviceAllocation final
{
public:
    explicit DeviceAllocation(std::size_t bytes)
    {
        check(cudaMalloc(&data_, bytes) == cudaSuccess);
    }

    ~DeviceAllocation()
    {
        if (data_ != nullptr)
        {
            (void)cudaFree(data_);
        }
    }

    DeviceAllocation(const DeviceAllocation&)            = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    void* get() const noexcept
    {
        return data_;
    }

private:
    void* data_ = nullptr;
};

class PinnedAllocation final
{
public:
    explicit PinnedAllocation(std::size_t bytes)
    {
        check(cudaMallocHost(&data_, bytes) == cudaSuccess);
    }

    ~PinnedAllocation()
    {
        if (data_ != nullptr)
        {
            (void)cudaFreeHost(data_);
        }
    }

    PinnedAllocation(const PinnedAllocation&)            = delete;
    PinnedAllocation& operator=(const PinnedAllocation&) = delete;

    void* get() const noexcept
    {
        return data_;
    }

private:
    void* data_ = nullptr;
};

class ExplicitStream final
{
public:
    ExplicitStream()
    {
        check(cudaStreamCreate(&stream_) == cudaSuccess);
    }

    ~ExplicitStream()
    {
        if (stream_ != nullptr)
        {
            (void)cudaStreamDestroy(stream_);
        }
    }

    ExplicitStream(const ExplicitStream&)            = delete;
    ExplicitStream& operator=(const ExplicitStream&) = delete;

    cudaStream_t get() const noexcept
    {
        return stream_;
    }

private:
    cudaStream_t stream_ = nullptr;
};

TensorView fp32_tensor(void* data, std::size_t bytes, std::int32_t batch, std::int32_t height,
                       std::int32_t width)
{
    return { data,
             bytes,
             batch,
             3,
             height,
             width,
             TensorElementType::Float32,
             TensorLayout::Nchw,
             MemoryKind::CudaDevice };
}

template <std::size_t Size>
void check_fp32(const std::array<float, Size>& actual, const std::array<float, Size>& expected,
                float tolerance = 1.0e-5f)
{
    for (std::size_t index = 0; index < Size; ++index)
    {
        check(std::fabs(actual[index] - expected[index]) < tolerance);
    }
}

} // namespace

spec("ImageProcessor CUDA contract")
{
    it("owns bounded writable inference tensor storage")
    {
        CudaImageProcessorOptions options;
        options.max_tensor_bytes = 48;
        auto processor           = CudaImageProcessor::create(options);

        const TensorView output =
            processor->acquire_tensor(1, 3, 2, 2, TensorElementType::Float32);
        check(output.data != nullptr);
        check(output.byte_size == 48);
        check(output.batch == 1);
        check(output.channels == 3);
        check(output.height == 2);
        check(output.width == 2);
        check(output.memory_kind == MemoryKind::CudaDevice);

        const TensorView reused =
            processor->acquire_tensor(1, 3, 1, 1, TensorElementType::Float32);
        check(reused.data == output.data);
        check_throws_as(
            processor->acquire_tensor(1, 3, 3, 2, TensorElementType::Float32),
            ImageProcessorError);
    }

    it("composites RGB CHW through an affine mask and downloads packed BGR once")
    {
        const std::array<std::uint8_t, 12> base_pixels = {
            10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30,
        };
        const std::array<float, 12> rgb = {
            1.0F, 1.0F, 1.0F, 1.0F,
            0.5F, 0.5F, 0.5F, 0.5F,
            0.0F, 0.0F, 0.0F, 0.0F,
        };
        std::array<float, 4> mask = { 1.0F, 0.0F, 0.5F, 1.0F };

        auto processor = CudaImageProcessor::create();
        const ImageView base = processor->stage(
            { base_pixels.data(), base_pixels.size(), 2, 2, 6, PixelFormat::Bgr8,
              MemoryKind::Host });
        const TensorView aligned =
            processor->acquire_tensor(1, 3, 2, 2, TensorElementType::Float32);
        check(cudaMemcpy(aligned.data, rgb.data(), aligned.byte_size, cudaMemcpyHostToDevice) ==
              cudaSuccess);
        const TensorView alpha = {
            mask.data(), mask.size() * sizeof(float), 1, 1, 2, 2,
            TensorElementType::Float32, TensorLayout::Nchw, MemoryKind::Host,
        };

        const ImageView composed = processor->composite_affine(base, aligned, alpha, {}, {});
        check(composed.data != base.data);
        check(composed.memory_kind == MemoryKind::CudaDevice);
        check(composed.pixel_format == PixelFormat::Bgr8);

        std::array<std::uint8_t, 12> downloaded {};
        processor->download_bgr(composed, { downloaded.data(), downloaded.size() });
        const std::array<std::uint8_t, 12> expected = {
            0, 127, 255, 10, 20, 30, 5, 73, 142, 0, 127, 255,
        };
        check(downloaded == expected);

        TensorCompositeOptions quarter;
        quarter.strength = 0.25F;
        const ImageView recomposed =
            processor->composite_affine(composed, aligned, alpha, {}, quarter);
        check(recomposed.data == composed.data);
        processor->download_bgr(recomposed, { downloaded.data(), downloaded.size() });
        check(downloaded[3] == std::uint8_t { 10 });
        check(downloaded[4] == std::uint8_t { 20 });
        check(downloaded[5] == std::uint8_t { 30 });
    }

    it("samples RGB NV12 and I420 composition bases into packed BGR output")
    {
        const std::array<std::uint8_t, 12> rgb_base = {
            255, 1, 0, 255, 1, 0, 255, 1, 0, 255, 1, 0,
        };
        const std::array<std::uint8_t, 6> nv12 = { 82, 82, 82, 82, 90, 240 };
        const std::array<std::uint8_t, 6> i420 = { 82, 82, 82, 82, 90, 240 };
        const std::array<float, 12> rgb {};
        std::array<float, 4> mask {};
        const std::array<std::uint8_t, 12> expected = {
            0, 1, 255, 0, 1, 255, 0, 1, 255, 0, 1, 255,
        };

        auto processor = CudaImageProcessor::create();
        const TensorView aligned =
            processor->acquire_tensor(1, 3, 2, 2, TensorElementType::Float32);
        check(cudaMemcpy(aligned.data, rgb.data(), aligned.byte_size,
                         cudaMemcpyHostToDevice) == cudaSuccess);
        const TensorView alpha = {
            mask.data(), mask.size() * sizeof(float), 1, 1, 2, 2,
            TensorElementType::Float32, TensorLayout::Nchw, MemoryKind::Host,
        };

        const ImageView staged_rgb = processor->stage(
            { rgb_base.data(), rgb_base.size(), 2, 2, 6, PixelFormat::Rgb8,
              MemoryKind::Host });
        const ImageView composed_rgb =
            processor->composite_affine(staged_rgb, aligned, alpha, {}, {});
        std::array<std::uint8_t, 12> downloaded_rgb {};
        processor->download_bgr(composed_rgb,
                                { downloaded_rgb.data(), downloaded_rgb.size() });
        check(downloaded_rgb == expected);

        for (const auto format : { PixelFormat::Nv12, PixelFormat::I420 })
        {
            const auto& bytes = format == PixelFormat::Nv12 ? nv12 : i420;
            const ImageView base = processor->stage(
                { bytes.data(), bytes.size(), 2, 2, 2, format, MemoryKind::Host });
            const ImageView composed =
                processor->composite_affine(base, aligned, alpha, {}, {});
            check(composed.pixel_format == PixelFormat::Bgr8);
            check(composed.row_stride == 6U);
            std::array<std::uint8_t, 12> downloaded {};
            processor->download_bgr(composed,
                                    { downloaded.data(), downloaded.size() });
            check(downloaded == expected);
        }
    }

    it("rejects malformed affine composition inputs")
    {
        const std::array<std::uint8_t, 3> pixel = { 0, 0, 0 };
        std::array<float, 3> rgb = { 0.0F, 0.0F, 0.0F };
        std::array<float, 1> mask = { 1.0F };
        auto processor = CudaImageProcessor::create();
        const ImageView base = processor->stage(
            { pixel.data(), pixel.size(), 1, 1, 3, PixelFormat::Bgr8, MemoryKind::Host });
        const TensorView aligned =
            processor->acquire_tensor(1, 3, 1, 1, TensorElementType::Float32);
        check(cudaMemcpy(aligned.data, rgb.data(), aligned.byte_size, cudaMemcpyHostToDevice) ==
              cudaSuccess);
        TensorView alpha = {
            mask.data(), sizeof(float), 1, 1, 1, 1, TensorElementType::Float32,
            TensorLayout::Nchw, MemoryKind::Host,
        };

        TensorCompositeOptions invalid_strength;
        invalid_strength.strength = 1.1F;
        check_throws_as(
            processor->composite_affine(base, aligned, alpha, {}, invalid_strength),
            ImageProcessorError);
        alpha.channels = 3;
        check_throws_as(processor->composite_affine(base, aligned, alpha, {}, {}),
                        ImageProcessorError);
        check_throws_as(processor->download_bgr(base, { nullptr, 0 }), ImageProcessorError);
    }

    it("uses destination-to-aligned coordinates for affine composition")
    {
        const std::array<std::uint8_t, 9> base_pixels {};
        const std::array<float, 3> rgb = { 1.0F, 0.0F, 0.0F };
        std::array<float, 1> mask = { 1.0F };
        auto processor = CudaImageProcessor::create();
        const ImageView base = processor->stage(
            { base_pixels.data(), base_pixels.size(), 3, 1, 9, PixelFormat::Bgr8,
              MemoryKind::Host });
        const TensorView aligned =
            processor->acquire_tensor(1, 3, 1, 1, TensorElementType::Float32);
        check(cudaMemcpy(aligned.data, rgb.data(), aligned.byte_size, cudaMemcpyHostToDevice) ==
              cudaSuccess);
        const TensorView alpha = {
            mask.data(), sizeof(float), 1, 1, 1, 1, TensorElementType::Float32,
            TensorLayout::Nchw, MemoryKind::Host,
        };
        AffineTransform destination_to_aligned;
        destination_to_aligned.destination_to_source = {
            1.0F, 0.0F, -1.0F, 0.0F, 1.0F, 0.0F,
        };
        const ImageView composed = processor->composite_affine(
            base, aligned, alpha, destination_to_aligned);
        std::array<std::uint8_t, 9> downloaded {};
        processor->download_bgr(composed, { downloaded.data(), downloaded.size() });
        const std::array<std::uint8_t, 9> expected = {
            0, 0, 0, 0, 0, 255, 0, 0, 0,
        };
        check(downloaded == expected);
    }

    it("owns reusable storage for synchronous affine and normalized host preprocessing")
    {
        const std::array<std::uint8_t, 12> bgr = {
            0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
        };
        const ImageView image = {
            bgr.data(), bgr.size(), 2, 2, 6, PixelFormat::Bgr8, MemoryKind::Host,
        };
        CudaImageProcessorOptions processor_options;
        processor_options.max_source_bytes = 1024;
        processor_options.max_tensor_bytes = 1024;
        auto processor                     = CudaImageProcessor::create(processor_options);

        const ImageView staged = processor->stage(image);
        check(staged.memory_kind == MemoryKind::CudaDevice);
        check(staged.data != image.data);
        check(staged.row_stride == 6);
        check(staged.byte_size == bgr.size());

        PreprocessOptions options;
        options.output_format = PixelFormat::Rgb8;
        options.mean          = { 0.5F, 0.5F, 0.5F };
        options.stddev        = { 0.5F, 0.5F, 0.5F };
        options.border_value  = 0.0F;
        const AffineTransform identity;
        const TensorView      first =
            processor->process_affine(staged, 2, 2, identity, options, TensorElementType::Float32);

        check(first.memory_kind == MemoryKind::CudaDevice);
        check(first.batch == 1);
        check(first.channels == 3);
        check(first.height == 2);
        check(first.width == 2);
        check(first.byte_size == 12 * sizeof(float));
        std::array<float, 12> output {};
        check(cudaMemcpy(output.data(), first.data, first.byte_size, cudaMemcpyDeviceToHost) ==
              cudaSuccess);
        const std::array<float, 12> expected = {
            20.0F / 127.5F - 1.0F,  50.0F / 127.5F - 1.0F,  80.0F / 127.5F - 1.0F,
            110.0F / 127.5F - 1.0F, 10.0F / 127.5F - 1.0F,  40.0F / 127.5F - 1.0F,
            70.0F / 127.5F - 1.0F,  100.0F / 127.5F - 1.0F, 0.0F / 127.5F - 1.0F,
            30.0F / 127.5F - 1.0F,  60.0F / 127.5F - 1.0F,  90.0F / 127.5F - 1.0F,
        };
        check_fp32(output, expected);

        const TensorView second =
            processor->process_affine(staged, 1, 1, identity, options, TensorElementType::Float32);
        check(second.data == first.data);
    }

    it("uses destination-to-source affine coordinates and constant borders")
    {
        const std::array<std::uint8_t, 3> rgb   = { 10, 20, 30 };
        const ImageView                   image = {
            rgb.data(), rgb.size(), 1, 1, 3, PixelFormat::Rgb8, MemoryKind::Host,
        };
        auto            processor = CudaImageProcessor::create();
        AffineTransform translated;
        translated.destination_to_source = { 1.0F, 0.0F, -1.0F, 0.0F, 1.0F, 0.0F };
        PreprocessOptions options;
        options.output_format = PixelFormat::Rgb8;
        options.border_value  = 7.0F;
        const TensorView tensor =
            processor->process_affine(image, 2, 1, translated, options, TensorElementType::Float32);

        std::array<float, 6> output {};
        check(cudaMemcpy(output.data(), tensor.data, tensor.byte_size, cudaMemcpyDeviceToHost) ==
              cudaSuccess);
        const std::array<float, 6> expected = {
            7.0F / 255.0F,  10.0F / 255.0F, 7.0F / 255.0F,
            20.0F / 255.0F, 7.0F / 255.0F,  30.0F / 255.0F,
        };
        check_fp32(output, expected);
    }

    it("stages packed NV12 and I420 once for affine preprocessing")
    {
        const std::array<std::uint8_t, 12> nv12 = {
            16, 81, 145, 235,
            32, 96, 160, 224,
            90, 240, 128, 128,
        };
        const std::array<std::uint8_t, 12> i420 = {
            16, 81, 145, 235,
            32, 96, 160, 224,
            90, 128,
            240, 128,
        };
        auto processor = CudaImageProcessor::create();
        PreprocessOptions options;
        options.output_format = PixelFormat::Rgb8;

        const auto verify = [&](const auto& pixels, PixelFormat format) {
            const ImageView host = {
                pixels.data(), pixels.size(), 4, 2, 4, format, MemoryKind::Host,
            };
            const ImageView staged = processor->stage(host);
            check(staged.memory_kind == MemoryKind::CudaDevice);
            check(staged.pixel_format == format);
            check(staged.row_stride == (std::size_t)4U);
            check(staged.byte_size == pixels.size());

            std::array<std::uint8_t, 12> staged_pixels {};
            check(cudaMemcpy(staged_pixels.data(), staged.data, staged.byte_size,
                             cudaMemcpyDeviceToHost) == cudaSuccess);
            check(staged_pixels == pixels);

            const BgrImage bgr = CpuImageProcessor::copy_bgr(host, 1024U);
            const std::vector<float> expected =
                CpuImageProcessor::to_nchw(bgr, options, 1024U);
            const TensorView tensor = processor->process_affine(
                staged, 4, 2, {}, options, TensorElementType::Float32);
            std::vector<float> actual(expected.size());
            check(cudaMemcpy(actual.data(), tensor.data, tensor.byte_size,
                             cudaMemcpyDeviceToHost) == cudaSuccess);
            for (std::size_t index = 0U; index < actual.size(); ++index)
            {
                check(std::fabs(actual[index] - expected[index]) < 1.0e-5F);
            }
        };

        verify(nv12, PixelFormat::Nv12);
        verify(i420, PixelFormat::I420);
    }

    it("packs padded host YUV planes into one compact CUDA image")
    {
        const std::array<std::uint8_t, 16> nv12 = {
            16, 81, 145, 235, 1, 2,
            32, 96, 160, 224, 3, 4,
            90, 240, 128, 128,
        };
        const std::array<std::uint8_t, 17> i420 = {
            16, 81, 145, 235, 1, 2,
            32, 96, 160, 224, 3, 4,
            90, 128, 5,
            240, 128,
        };
        const std::array<std::uint8_t, 12> expected_nv12 = {
            16, 81, 145, 235,
            32, 96, 160, 224,
            90, 240, 128, 128,
        };
        const std::array<std::uint8_t, 12> expected_i420 = {
            16, 81, 145, 235,
            32, 96, 160, 224,
            90, 128, 240, 128,
        };
        auto processor = CudaImageProcessor::create();
        const auto verify = [&](const auto& pixels, PixelFormat format,
                                const auto& expected) {
            const ImageView source = {
                pixels.data(), pixels.size(), 4, 2, 6, format, MemoryKind::Host,
            };
            const ImageView staged = processor->stage(source);
            check(staged.byte_size == expected.size());
            check(staged.row_stride == (std::size_t)4U);
            std::array<std::uint8_t, 12> actual {};
            check(cudaMemcpy(actual.data(), staged.data, actual.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess);
            check(actual == expected);
        };
        verify(nv12, PixelFormat::Nv12, expected_nv12);
        verify(i420, PixelFormat::I420, expected_i420);
    }

    it("packs padded NV21 YUY2 and UYVY host rows into compact CUDA images")
    {
        struct Case
        {
            std::vector<std::uint8_t> storage;
            std::vector<std::uint8_t> expected;
            std::int32_t width;
            std::int32_t height;
            std::size_t stride;
            PixelFormat format;
        };
        const std::array<Case, 3> cases = {
            Case { { 16, 81, 145, 235, 1, 2,
                     32, 96, 160, 224, 3, 4,
                     240, 90, 128, 128 },
                   { 16, 81, 145, 235, 32, 96, 160, 224, 240, 90, 128, 128 },
                   4, 2, 6, PixelFormat::Nv21 },
            Case { { 81, 90, 81, 240, 145, 128, 145, 128, 1, 2,
                     81, 90, 81, 240, 145, 128, 145, 128, 3, 4 },
                   { 81, 90, 81, 240, 145, 128, 145, 128,
                     81, 90, 81, 240, 145, 128, 145, 128 },
                   4, 2, 10, PixelFormat::Yuy2 },
            Case { { 90, 81, 240, 81, 128, 145, 128, 145, 1, 2,
                     90, 81, 240, 81, 128, 145, 128, 145, 3, 4 },
                   { 90, 81, 240, 81, 128, 145, 128, 145,
                     90, 81, 240, 81, 128, 145, 128, 145 },
                   4, 2, 10, PixelFormat::Uyvy },
        };
        auto processor = CudaImageProcessor::create();

        for (const Case& item : cases)
        {
            const ImageView source = { item.storage.data(), item.storage.size(),
                                       item.width, item.height, item.stride,
                                       item.format, MemoryKind::Host };
            const ImageView staged = processor->stage(source);
            check(staged.pixel_format == item.format);
            check(staged.byte_size == item.expected.size());
            std::vector<std::uint8_t> actual(item.expected.size());
            check(cudaMemcpy(actual.data(), staged.data, actual.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess);
            check_eq_container(actual, item.expected);
        }
    }

    it("matches CPU preprocessing for NV21 YUY2 and UYVY")
    {
        struct Case
        {
            std::vector<std::uint8_t> pixels;
            std::int32_t width;
            std::int32_t height;
            std::size_t stride;
            PixelFormat format;
        };
        const std::array<Case, 3> cases = {
            Case { { 81, 81, 81, 81, 240, 90 }, 2, 2, 2, PixelFormat::Nv21 },
            Case { { 81, 90, 81, 240 }, 2, 1, 4, PixelFormat::Yuy2 },
            Case { { 90, 81, 240, 81 }, 2, 1, 4, PixelFormat::Uyvy },
        };
        auto processor = CudaImageProcessor::create();
        PreprocessOptions options;
        options.output_format = PixelFormat::Rgb8;

        for (const Case& item : cases)
        {
            const ImageView source = { item.pixels.data(), item.pixels.size(),
                                       item.width, item.height, item.stride,
                                       item.format, MemoryKind::Host };
            LetterboxTransform cpu_transform;
            const std::vector<float> expected = CpuImageProcessor::letterbox_nchw(
                source, item.width, item.height, options, 1024, 1024, &cpu_transform);
            const TensorView tensor = processor->process_affine(
                source, item.width, item.height, {}, options, TensorElementType::Float32);
            std::vector<float> actual(expected.size());
            check(cudaMemcpy(actual.data(), tensor.data, tensor.byte_size,
                             cudaMemcpyDeviceToHost) == cudaSuccess);
            for (std::size_t index = 0U; index < actual.size(); ++index)
            {
                check(std::fabs(actual[index] - expected[index]) < 1.0e-5F);
            }
        }
    }

    it("fails fast when affine preprocessing exceeds configured source capacity")
    {
        const std::array<std::uint8_t, 3> rgb   = { 1, 2, 3 };
        const ImageView                   image = {
            rgb.data(), rgb.size(), 1, 1, 3, PixelFormat::Rgb8, MemoryKind::Host,
        };
        CudaImageProcessorOptions options;
        options.max_source_bytes = 2;
        options.max_tensor_bytes = 1024;
        auto processor           = CudaImageProcessor::create(options);
        check_throws_as(processor->process_affine(image, 1, 1, {}, {}, TensorElementType::Float32),
                        ImageProcessorError);
    }

    it("letterboxes a CUDA RGB image into normalized FP32 NCHW on an explicit stream")
    {
        constexpr std::size_t                         kSourceStride = 8;
        const std::array<std::uint8_t, kSourceStride> rgb           = {
            255, 0, 0, 0, 128, 255, 17, 19,
        };
        const std::array<float, 18> expected = {
            0.6941177f, 0.6941177f, 1.8f,   -0.2f,      0.6941177f, 0.6941177f,
            0.9882353f, 0.9882353f, -0.8f,  1.2078432f, 0.9882353f, 0.9882353f,
            0.0735294f, 0.0735294f, -0.15f, 0.35f,      0.0735294f, 0.0735294f,
        };
        DeviceAllocation                   source(rgb.size());
        DeviceAllocation                   destination(expected.size() * sizeof(float));
        ExplicitStream                     stream;
        std::array<float, expected.size()> output {};
        check(cudaMemcpyAsync(source.get(), rgb.data(), rgb.size(), cudaMemcpyHostToDevice,
                              stream.get()) == cudaSuccess);

        const std::vector<ImageView> images = {
            { source.get(), rgb.size(), 2, 1, kSourceStride, PixelFormat::Rgb8,
              MemoryKind::CudaDevice },
        };
        const TensorView tensor =
            fp32_tensor(destination.get(), expected.size() * sizeof(float), 1, 3, 2);
        const BatchPlan   plan = ImageProcessor::plan(images, tensor, 1024, 1024);
        PreprocessOptions options;
        options.mean   = { 0.1f, 0.2f, 0.3f };
        options.stddev = { 0.5f, 0.25f, 2.0f };

        ImageProcessor::enqueue(images, tensor, plan, {}, {}, options, stream.get());
        check(cudaMemcpyAsync(output.data(), destination.get(), output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        check_fp32(output, expected);
    }

    it("supports BGR tensor channel order independently of source format")
    {
        const std::array<std::uint8_t, 3> rgb      = { 10, 20, 30 };
        const std::array<float, 3>        expected = {
            30.0f / 255.0f,
            20.0f / 255.0f,
            10.0f / 255.0f,
        };
        DeviceAllocation     source(rgb.size());
        DeviceAllocation     destination(expected.size() * sizeof(float));
        ExplicitStream       stream;
        std::array<float, 3> output {};
        check(cudaMemcpyAsync(source.get(), rgb.data(), rgb.size(), cudaMemcpyHostToDevice,
                              stream.get()) == cudaSuccess);

        const std::vector<ImageView> images = {
            { source.get(), rgb.size(), 1, 1, 3, PixelFormat::Rgb8, MemoryKind::CudaDevice },
        };
        const TensorView  tensor = fp32_tensor(destination.get(), sizeof(output), 1, 1, 1);
        const BatchPlan   plan   = ImageProcessor::plan(images, tensor, 1024, 1024);
        PreprocessOptions options;
        options.output_format = PixelFormat::Bgr8;

        ImageProcessor::enqueue(images, tensor, plan, {}, {}, options, stream.get());
        check(cudaMemcpyAsync(output.data(), destination.get(), sizeof(output),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        check_fp32(output, expected);
    }

    it("fuses staged NV12 conversion into the CUDA tensor")
    {
        const std::array<std::uint8_t, 6> nv12 = {
            81, 81, 81, 81, 90, 240,
        };
        const ImageView image = {
            nv12.data(), nv12.size(), 2, 2, 2, PixelFormat::Nv12, MemoryKind::Host,
        };
        PreprocessOptions options;
        options.output_format = PixelFormat::Rgb8;
        LetterboxTransform cpu_transform;
        const std::vector<float> expected = CpuImageProcessor::letterbox_nchw(
            image, 2, 2, options, 1024, 1024, &cpu_transform);

        DeviceAllocation destination(expected.size() * sizeof(float));
        DeviceAllocation device_workspace(nv12.size());
        PinnedAllocation pinned_workspace(nv12.size());
        ExplicitStream stream;
        std::array<float, 12> output {};
        const TensorView tensor = fp32_tensor(
            destination.get(), expected.size() * sizeof(float), 1, 2, 2);
        const BatchPlan plan = ImageProcessor::plan({ image }, tensor, 1024, 1024);
        ImageProcessor::stage_host_inputs(
            { image }, plan, { pinned_workspace.get(), plan.host_staging_bytes });

        ImageProcessor::enqueue(
            { image }, tensor, plan,
            { pinned_workspace.get(), plan.host_staging_bytes },
            { device_workspace.get(), plan.device_staging_bytes }, options, stream.get());
        check(cudaMemcpyAsync(output.data(), destination.get(), sizeof(output),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        for (std::size_t index = 0U; index < output.size(); ++index)
        {
            check(std::fabs(output[index] - expected[index]) < 1.0e-5F);
        }
    }

    it("matches CPU I420 preprocessing with horizontal mirror")
    {
        const std::array<std::uint8_t, 12> i420 = {
            16, 235, 81, 145,
            16, 235, 81, 145,
            128, 128,
            128, 128,
        };
        DeviceAllocation source(i420.size());
        DeviceAllocation destination(24U * sizeof(float));
        ExplicitStream stream;
        check(cudaMemcpyAsync(source.get(), i420.data(), i420.size(),
                              cudaMemcpyHostToDevice, stream.get()) == cudaSuccess);
        const ImageView host_image = {
            i420.data(), i420.size(), 4, 2, 4, PixelFormat::I420, MemoryKind::Host,
        };
        const ImageView device_image = {
            source.get(), i420.size(), 4, 2, 4, PixelFormat::I420,
            MemoryKind::CudaDevice,
        };
        PreprocessOptions options;
        options.output_format = PixelFormat::Rgb8;
        options.mirror_horizontal = true;
        LetterboxTransform cpu_transform;
        const std::vector<float> expected = CpuImageProcessor::letterbox_nchw(
            host_image, 4, 2, options, 1024, 1024, &cpu_transform);
        std::array<float, 24> output {};
        const TensorView tensor = fp32_tensor(
            destination.get(), sizeof(output), 1, 2, 4);
        const BatchPlan plan = ImageProcessor::plan(
            { device_image }, tensor, 1024, 1024);

        ImageProcessor::enqueue(
            { device_image }, tensor, plan, {}, {}, options, stream.get());
        check(cudaMemcpyAsync(output.data(), destination.get(), sizeof(output),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        for (std::size_t index = 0U; index < output.size(); ++index)
        {
            check(std::fabs(output[index] - expected[index]) < 1.0e-5F);
        }
    }

    it("bilinearly samples the midpoint during non-square FP32 scaling")
    {
        constexpr std::size_t                         kSourceStride = 8;
        const std::array<std::uint8_t, kSourceStride> rgb           = {
            0, 64, 128, 200, 192, 0, 31, 37,
        };
        const std::array<float, 18> expected = {
            0.0f,       0.3921569f, 0.7843137f, 0.0f,       0.3921569f, 0.7843137f,
            0.2509804f, 0.5019608f, 0.7529412f, 0.2509804f, 0.5019608f, 0.7529412f,
            0.5019608f, 0.2509804f, 0.0f,       0.5019608f, 0.2509804f, 0.0f,
        };
        DeviceAllocation                   source(rgb.size());
        DeviceAllocation                   destination(expected.size() * sizeof(float));
        ExplicitStream                     stream;
        std::array<float, expected.size()> output {};
        check(cudaMemcpyAsync(source.get(), rgb.data(), rgb.size(), cudaMemcpyHostToDevice,
                              stream.get()) == cudaSuccess);

        const std::vector<ImageView> images = {
            { source.get(), rgb.size(), 2, 1, kSourceStride, PixelFormat::Rgb8,
              MemoryKind::CudaDevice },
        };
        const TensorView tensor =
            fp32_tensor(destination.get(), expected.size() * sizeof(float), 1, 2, 3);
        const BatchPlan plan = ImageProcessor::plan(images, tensor, 1024, 1024);

        ImageProcessor::enqueue(images, tensor, plan, {}, {}, {}, stream.get());
        check(cudaMemcpyAsync(output.data(), destination.get(), output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        check_fp32(output, expected);
    }

    it("writes horizontal borders and image pixels to FP16 NCHW")
    {
        constexpr std::size_t                             kSourceStride = 5;
        const std::array<std::uint8_t, kSourceStride * 2> rgb           = {
            64, 128, 255, 41, 43, 255, 0, 128, 47, 53,
        };
        const std::array<float, 18> expected = {
            0.4470588f, 0.2509804f, 0.4470588f, 0.4470588f, 1.0f,       0.4470588f,
            0.4470588f, 0.5019608f, 0.4470588f, 0.4470588f, 0.0f,       0.4470588f,
            0.4470588f, 1.0f,       0.4470588f, 0.4470588f, 0.5019608f, 0.4470588f,
        };
        DeviceAllocation                    source(rgb.size());
        DeviceAllocation                    destination(expected.size() * sizeof(__half));
        ExplicitStream                      stream;
        std::array<__half, expected.size()> output {};
        check(cudaMemcpyAsync(source.get(), rgb.data(), rgb.size(), cudaMemcpyHostToDevice,
                              stream.get()) == cudaSuccess);

        const std::vector<ImageView> images = {
            { source.get(), rgb.size(), 1, 2, kSourceStride, PixelFormat::Rgb8,
              MemoryKind::CudaDevice },
        };
        const TensorView tensor = {
            destination.get(),
            expected.size() * sizeof(__half),
            1,
            3,
            2,
            3,
            TensorElementType::Float16,
            TensorLayout::Nchw,
            MemoryKind::CudaDevice,
        };
        const BatchPlan plan = ImageProcessor::plan(images, tensor, 1024, 1024);

        ImageProcessor::enqueue(images, tensor, plan, {}, {}, {}, stream.get());
        check(cudaMemcpyAsync(output.data(), destination.get(), output.size() * sizeof(__half),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        for (std::size_t index = 0; index < output.size(); ++index)
        {
            check(std::fabs(__half2float(output[index]) - expected[index]) < 8.0e-4f);
        }
    }

    it("uploads padded host input and mixes it with a CUDA input in one FP16 batch")
    {
        const std::array<std::uint8_t, 5> host_rgb   = { 64, 128, 255, 41, 43 };
        const std::array<std::uint8_t, 3> device_bgr = { 255, 128, 64 };
        const std::array<float, 6>        expected   = {
            64.0f / 255.0f, 128.0f / 255.0f, 1.0f, 64.0f / 255.0f, 128.0f / 255.0f, 1.0f,
        };
        DeviceAllocation                    device_source(device_bgr.size());
        DeviceAllocation                    device_workspace(3);
        DeviceAllocation                    destination(expected.size() * sizeof(__half));
        PinnedAllocation                    pinned_workspace(3);
        ExplicitStream                      stream;
        std::array<__half, expected.size()> output {};
        check(cudaMemcpyAsync(device_source.get(), device_bgr.data(), device_bgr.size(),
                              cudaMemcpyHostToDevice, stream.get()) == cudaSuccess);

        const std::vector<ImageView> images = {
            { host_rgb.data(), host_rgb.size(), 1, 1, host_rgb.size(), PixelFormat::Rgb8,
              MemoryKind::Host },
            { device_source.get(), device_bgr.size(), 1, 1, 3, PixelFormat::Bgr8,
              MemoryKind::CudaDevice },
        };
        const TensorView tensor = {
            destination.get(),
            expected.size() * sizeof(__half),
            2,
            3,
            1,
            1,
            TensorElementType::Float16,
            TensorLayout::Nchw,
            MemoryKind::CudaDevice,
        };
        const BatchPlan plan = ImageProcessor::plan(images, tensor, 1024, 1024);
        ImageProcessor::stage_host_inputs(images, plan,
                                          { pinned_workspace.get(), plan.host_staging_bytes });

        ImageProcessor::enqueue(
            images, tensor, plan, { pinned_workspace.get(), plan.host_staging_bytes },
            { device_workspace.get(), plan.device_staging_bytes }, {}, stream.get());
        check(cudaMemcpyAsync(output.data(), destination.get(), output.size() * sizeof(__half),
                              cudaMemcpyDeviceToHost, stream.get()) == cudaSuccess);
        check(cudaStreamSynchronize(stream.get()) == cudaSuccess);
        for (std::size_t index = 0; index < output.size(); ++index)
        {
            check(std::fabs(__half2float(output[index]) - expected[index]) < 8.0e-4f);
        }
    }
}
