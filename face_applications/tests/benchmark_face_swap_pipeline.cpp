#include "face_swap_demo_metrics.hpp"
#include "opencv_image_adapter.hpp"

#include "kfcore/face_applications/tensorrt.hpp"
#include "tinytest.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

using namespace kfcore::face_applications;

namespace
{

constexpr std::size_t kWarmupIterations = 5U;
constexpr std::size_t kMeasuredSamples = 50U;

using Clock = std::chrono::steady_clock;

double elapsed_milliseconds(Clock::time_point started)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

double milliseconds(const std::optional<FaceSwapDuration>& duration)
{
    return duration ? std::chrono::duration<double, std::milli>(*duration).count() : -1.0;
}

void print_stage_percentiles(const demo::TimingHistory& history)
{
    std::cout << "      | stage | current(ms) | P50(ms) | P95(ms) |\n"
                 "      | :--- | ---: | ---: | ---: |\n";
    for (const demo::TimingRow& row : history.rows())
    {
        std::cout << "      | " << row.label << " | ";
        if (row.current)
        {
            std::cout << std::fixed << std::setprecision(3) << milliseconds(row.current);
        }
        else
        {
            std::cout << '-';
        }
        std::cout << " | ";
        if (row.p50)
        {
            std::cout << std::fixed << std::setprecision(3) << milliseconds(row.p50);
        }
        else
        {
            std::cout << '-';
        }
        std::cout << " | ";
        if (row.p95)
        {
            std::cout << std::fixed << std::setprecision(3) << milliseconds(row.p95);
        }
        else
        {
            std::cout << '-';
        }
        std::cout << " |\n";
    }
}

void run_pipeline(const char* name, const cv::Mat& source, const cv::Mat& target,
                  const FaceApplicationModelPaths& paths)
{
    const auto load_started = Clock::now();
    auto application = TensorRtFaceSwapApplication::load(paths);
    const double load_ms = elapsed_milliseconds(load_started);
    const kfcore::image::ImageView source_view =
        demo::borrowed_bgr(source, "source");
    const kfcore::image::ImageView target_view =
        demo::borrowed_bgr(target, "target");

    const auto first_swap_started = Clock::now();
    kfcore::image::BgrImage output = application->swap(source_view, target_view);
    const double first_swap_ms = elapsed_milliseconds(first_swap_started);
    std::cout << "      cold phase " << name << ": load=" << std::fixed
              << std::setprecision(3) << load_ms << " ms, first_swap=" << first_swap_ms
              << " ms, load_plus_first_swap=" << load_ms + first_swap_ms << " ms\n";
    check(load_ms > 0.0);
    check(first_swap_ms > 0.0);

    for (std::size_t iteration = 1U; iteration < kWarmupIterations; ++iteration)
    {
        output = application->swap(source_view, target_view);
    }

    benchmark_batch(name, kMeasuredSamples)
    {
        output = application->swap(source_view, target_view);
    }
    check_false(output.empty());
    check(output.width == target.cols);
    check(output.height == target.rows);

    demo::TimingHistory history;
    for (std::size_t sample = 0U; sample < kMeasuredSamples; ++sample)
    {
        ProfiledFaceSwapResult profiled =
            application->swap_profiled(source_view, target_view);
        history.add(profiled.timings);
        output = std::move(profiled.image);
    }
    check(history.sample_count() == kMeasuredSamples);
    print_stage_percentiles(history);
}

FaceApplicationModelPaths required_paths()
{
    return { KFCORE_TEST_12FACE_ENGINE, KFCORE_TEST_FACE68_ENGINE,
             KFCORE_TEST_ARCFACE_ENGINE, KFCORE_TEST_INSWAPPER_ENGINE,
             KFCORE_TEST_INSWAPPER_MATRIX, std::nullopt, std::nullopt };
}

} // namespace

suite("face swap real-model performance")
{
    bench("warmed TensorRT pipeline")
    {
        const auto decode_started = Clock::now();
        const cv::Mat source = cv::imread(KFCORE_TEST_SOURCE_IMAGE, cv::IMREAD_COLOR);
        const cv::Mat target = cv::imread(KFCORE_TEST_TARGET_IMAGE, cv::IMREAD_COLOR);
        const double decode_ms = elapsed_milliseconds(decode_started);
        check_false(source.empty());
        check_false(target.empty());
        std::cout << "      cold phase source+target image decode: " << std::fixed
                  << std::setprecision(3) << decode_ms << " ms\n";
        check(decode_ms > 0.0);

        run_pipeline("TensorRT face swap without GFPGAN", source, target, required_paths());

        const std::string gfpgan_path = KFCORE_TEST_GFPGAN_ENGINE;
        if (!gfpgan_path.empty())
        {
            FaceApplicationModelPaths paths = required_paths();
            paths.gfpgan_engine = gfpgan_path;
            run_pipeline("TensorRT face swap with GFPGAN", source, target, paths);
        }
    }
}
