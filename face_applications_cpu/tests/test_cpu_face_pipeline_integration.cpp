#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "kfcore/face_applications/cpu.hpp"
#include "kfcore/image_processor/cpu.hpp"
#include "tinytest.hpp"

#include <cstdint>
#include <filesystem>
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

} // namespace

spec("ONNX Runtime CPU face pipeline integration")
{
    it("loads local models and returns owned analysis and swap results")
    {
        const kfcore::image::BgrImage source = read_bgr(KFCORE_FACE_CPU_TEST_SOURCE_IMAGE);
        const kfcore::image::BgrImage target = read_bgr(KFCORE_FACE_CPU_TEST_TARGET_IMAGE);
        const std::vector<std::uint8_t> source_before = source.pixels;
        const std::vector<std::uint8_t> target_before = target.pixels;
        CpuFaceSwapOptions options;
        options.intra_op_threads = 1;
        options.inter_op_threads = 1;
        const auto application = OnnxFaceSwapApplication::load(model_paths(), options);

        const CpuFaceAnalysis analysis = application->analyze(source);
        const ProfiledCpuFaceSwapResult result = application->swap_profiled(source, target);

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
    }
}
