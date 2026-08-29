#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "kfcore/face_applications/cpu.hpp"
#include "kfcore/image_processor/cpu.hpp"
#include "tinytest.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>

using namespace kfcore::face_applications;

namespace
{

kfcore::image::BgrImage read_bgr(const std::filesystem::path& path)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* rgb = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
    if (rgb == nullptr)
    {
        throw std::runtime_error("failed to decode integration image");
    }
    const std::size_t bytes = static_cast<std::size_t>(width) * height * 3U;
    const kfcore::image::ImageView view { rgb, bytes, width, height,
                                          static_cast<std::size_t>(width) * 3U,
                                          kfcore::image::PixelFormat::Rgb8,
                                          kfcore::image::MemoryKind::Host };
    kfcore::image::BgrImage image =
        kfcore::image::CpuImageProcessor::copy_bgr(view, 64U * 1024U * 1024U);
    stbi_image_free(rgb);
    return image;
}

CpuFaceApplicationModelPaths model_paths()
{
    CpuFaceApplicationModelPaths paths;
    paths.detector_model = KFCORE_FACE_CPU_TEST_DETECTOR;
    paths.face68_model = KFCORE_FACE_CPU_TEST_FACE68;
    paths.arcface_model = KFCORE_FACE_CPU_TEST_ARCFACE;
    paths.inswapper_model = KFCORE_FACE_CPU_TEST_INSWAPPER;
    paths.inswapper_matrix = KFCORE_FACE_CPU_TEST_MATRIX;
    paths.gfpgan_model = std::filesystem::path(KFCORE_FACE_CPU_TEST_GFPGAN);
    paths.age_gender_model = std::filesystem::path(KFCORE_FACE_CPU_TEST_AGE_GENDER);
    return paths;
}

struct OwnedNv12
{
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> bytes;

    [[nodiscard]] kfcore::image::ImageView view() const noexcept
    {
        return { bytes.data(), bytes.size(), width, height,
                 static_cast<std::size_t>(width), kfcore::image::PixelFormat::Nv12,
                 kfcore::image::MemoryKind::Host };
    }
};

std::uint8_t yuv_byte(int value)
{
    return static_cast<std::uint8_t>((std::clamp)(value, 0, 255));
}

OwnedNv12 to_nv12(const kfcore::image::BgrImage& image)
{
    if ((image.width & 1) != 0 || (image.height & 1) != 0)
    {
        throw std::runtime_error("NV12 integration fixture requires even dimensions");
    }
    OwnedNv12 result;
    result.width = image.width;
    result.height = image.height;
    const std::size_t y_bytes = static_cast<std::size_t>(image.width) * image.height;
    result.bytes.resize(y_bytes + y_bytes / 2U);
    for (int y = 0; y < image.height; ++y)
    {
        for (int x = 0; x < image.width; ++x)
        {
            const std::size_t pixel =
                (static_cast<std::size_t>(y) * image.width + x) * 3U;
            const int blue = image.pixels[pixel];
            const int green = image.pixels[pixel + 1U];
            const int red = image.pixels[pixel + 2U];
            result.bytes[static_cast<std::size_t>(y) * image.width + x] =
                yuv_byte(((66 * red + 129 * green + 25 * blue + 128) >> 8) + 16);
        }
    }
    for (int y = 0; y < image.height; y += 2)
    {
        for (int x = 0; x < image.width; x += 2)
        {
            int red = 0;
            int green = 0;
            int blue = 0;
            for (int offset_y = 0; offset_y < 2; ++offset_y)
            {
                for (int offset_x = 0; offset_x < 2; ++offset_x)
                {
                    const std::size_t pixel =
                        (static_cast<std::size_t>(y + offset_y) * image.width +
                         x + offset_x) * 3U;
                    blue += image.pixels[pixel];
                    green += image.pixels[pixel + 1U];
                    red += image.pixels[pixel + 2U];
                }
            }
            red /= 4;
            green /= 4;
            blue /= 4;
            const std::size_t uv = y_bytes +
                static_cast<std::size_t>(y / 2) * image.width + x;
            result.bytes[uv] = yuv_byte(((-38 * red - 74 * green + 112 * blue + 128) >> 8) + 128);
            result.bytes[uv + 1U] = yuv_byte(((112 * red - 94 * green - 18 * blue + 128) >> 8) + 128);
        }
    }
    return result;
}

} // namespace

spec("ONNX Runtime CPU face pipeline integration")
{
    it("loads local models and returns owned analysis and swap results")
    {
        const kfcore::image::BgrImage source = read_bgr(KFCORE_FACE_CPU_TEST_SOURCE_IMAGE);
        const kfcore::image::BgrImage target = read_bgr(KFCORE_FACE_CPU_TEST_TARGET_IMAGE);
        const std::vector<std::uint8_t> source_before = source.pixels;
        const std::vector<std::uint8_t> target_before = target.pixels;
        const OwnedNv12 source_nv12 = to_nv12(source);
        const OwnedNv12 target_nv12 = to_nv12(target);
        const std::vector<std::uint8_t> source_nv12_before = source_nv12.bytes;
        const std::vector<std::uint8_t> target_nv12_before = target_nv12.bytes;
        CpuFaceSwapOptions options;
        options.intra_op_threads = 1;
        options.inter_op_threads = 1;
        const auto application = OnnxFaceSwapApplication::load(model_paths(), options);

        const CpuFaceAnalysis analysis = application->analyze(source_nv12.view());
        const ProfiledCpuFaceSwapResult result =
            application->swap_profiled(source_nv12.view(), target_nv12.view());

        check(analysis.detection.score >= options.detector_score_threshold);
        check(analysis.age_gender_logits.has_value());
        check(result.image.width == target.width);
        check(result.image.height == target.height);
        check(result.image.pixels.size() == target.pixels.size());
        check(result.timings.total.count() > 0);
        check(result.timings.source_analysis.detection.count() > 0);
        check(result.timings.inswapper_inference_and_decode.count() > 0);
        check(result.timings.gfpgan_inference_and_decode.has_value());
        check_eq_container(source.pixels, source_before);
        check_eq_container(target.pixels, target_before);
        check_eq_container(source_nv12.bytes, source_nv12_before);
        check_eq_container(target_nv12.bytes, target_nv12_before);
    }
}
