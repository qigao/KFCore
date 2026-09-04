#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "kfcore/face_applications/cpu.hpp"
#include "kfcore/image_processor/cpu.hpp"
#include "tinytest.hpp"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>

using namespace kfcore::face_applications;

namespace
{

constexpr std::size_t kMeasuredSamples = 3U;
using Clock = std::chrono::steady_clock;

double milliseconds(CpuFaceSwapDuration duration)
{
    return std::chrono::duration<double, std::milli>(duration).count();
}

double elapsed_milliseconds(Clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

kfcore::image::BgrImage read_bgr(const std::filesystem::path& path)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    using StbiPixels = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;
    StbiPixels rgb(stbi_load(path.string().c_str(), &width, &height, &channels, 3),
                   &stbi_image_free);
    if (!rgb)
    {
        throw std::runtime_error("failed to decode benchmark image");
    }
    const std::size_t bytes = static_cast<std::size_t>(width) * height * 3U;
    const kfcore::image::ImageView view {
        rgb.get(), bytes, width, height, static_cast<std::size_t>(width) * 3U,
        kfcore::image::PixelFormat::Rgb8, kfcore::image::MemoryKind::Host
    };
    return kfcore::image::CpuImageProcessor::copy_bgr(view, 64U * 1024U * 1024U);
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

void print_analysis(const char* prefix, const CpuFaceAnalysisTimingReport& report)
{
    std::cout << "      " << prefix
              << ": staging=" << milliseconds(report.initial_staging)
              << " ms, detection=" << milliseconds(report.detection)
              << " ms, face68_preprocess=" << milliseconds(report.face68_preprocess)
              << " ms, face68_inference_post="
              << milliseconds(report.face68_inference_and_postprocess)
              << " ms, arcface_preprocess=" << milliseconds(report.arcface_preprocess)
              << " ms, arcface_inference=" << milliseconds(report.arcface_inference)
              << " ms, age_gender="
              << (report.age_gender ? milliseconds(*report.age_gender) : -1.0)
              << " ms, total=" << milliseconds(report.total) << " ms\n";
}

void print_report(const CpuFaceSwapTimingReport& report)
{
    print_analysis("source analysis", report.source_analysis);
    print_analysis("target analysis", report.target_analysis);
    std::cout << "      swap: projection=" << milliseconds(report.embedding_projection)
              << " ms, inswapper_preprocess="
              << milliseconds(report.inswapper_preprocess)
              << " ms, inswapper_inference_decode="
              << milliseconds(report.inswapper_inference_and_decode)
              << " ms, inswapper_composition="
              << milliseconds(report.inswapper_composition)
              << " ms, gfpgan_preprocess="
              << (report.gfpgan_preprocess ? milliseconds(*report.gfpgan_preprocess) : -1.0)
              << " ms, gfpgan_inference_decode="
              << (report.gfpgan_inference_and_decode
                      ? milliseconds(*report.gfpgan_inference_and_decode)
                      : -1.0)
              << " ms, gfpgan_composition="
              << (report.gfpgan_composition ? milliseconds(*report.gfpgan_composition) : -1.0)
              << " ms, total=" << milliseconds(report.total) << " ms\n";
}

} // namespace

suite("ONNX Runtime CPU face pipeline performance")
{
    bench("complete CPU face swap")
    {
        const auto decode_started = Clock::now();
        const kfcore::image::BgrImage source = read_bgr(KFCORE_FACE_CPU_TEST_SOURCE_IMAGE);
        const kfcore::image::BgrImage target = read_bgr(KFCORE_FACE_CPU_TEST_TARGET_IMAGE);
        const double decode_ms = elapsed_milliseconds(decode_started);

        const auto load_started = Clock::now();
        auto application = OnnxFaceSwapApplication::load(model_paths());
        const double load_ms = elapsed_milliseconds(load_started);

        const auto first_started = Clock::now();
        ProfiledCpuFaceSwapResult result = application->swap_profiled(source, target);
        const double first_swap_ms = elapsed_milliseconds(first_started);
        std::cout << std::fixed << std::setprecision(3)
                  << "      cold phase: image_decode=" << decode_ms
                  << " ms, model_load=" << load_ms
                  << " ms, first_swap=" << first_swap_ms
                  << " ms, load_plus_first_swap=" << load_ms + first_swap_ms << " ms\n";
        print_report(result.timings);

        benchmark_batch("ONNX Runtime CPU face swap with GFPGAN", kMeasuredSamples)
        {
            result = application->swap_profiled(source, target);
        }
        check(result.image.width == target.width);
        check(result.image.height == target.height);
        check(result.timings.total.count() > 0);
        print_report(result.timings);
    }
}
