#include "face_swap_cli.hpp"
#include "face_swap_demo_metrics.hpp"
#include "face_swap_demo_ui.hpp"
#include "opencv_image_adapter.hpp"

#include "kfcore/face_applications/tensorrt.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr const char* kWindowName = "KFCore TensorRT Face Swap";
constexpr int kEventPollMilliseconds = 30;
constexpr int kDisplayRefreshMilliseconds = 1;

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error(message);
}

cv::Mat read_bgr(const std::string& path, const char* role)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty() || image.type() != CV_8UC3)
    {
        fail(std::string("failed to decode ") + role + " as BGR image: " + path);
    }
    return image;
}

void require_valid_output(const kfcore::image::BgrImage& output, const cv::Mat& target)
{
    if (output.width != target.cols || output.height != target.rows || output.empty())
    {
        fail("swap returned an invalid output image");
    }
}

void require_valid_output(const cv::Mat& output, const cv::Mat& target)
{
    if (output.empty() || output.type() != CV_8UC3 || output.size() != target.size())
    {
        fail("display output image is invalid");
    }
}

void show(const cv::Mat& source, const cv::Mat& target, const cv::Mat& result,
          const std::string& status,
          const std::vector<kfcore::face_applications::demo::TimingRow>& timing_rows = {},
          std::size_t timing_samples = 0U)
{
    cv::imshow(kWindowName,
               kfcore::face_applications::demo::compose_canvas(
                   source, target, result, status, timing_rows, timing_samples));
    (void)cv::waitKey(kDisplayRefreshMilliseconds);
}

std::string completed_status(kfcore::face_applications::FaceSwapDuration elapsed)
{
    const double milliseconds =
        std::chrono::duration<double, std::milli>(elapsed).count();
    std::ostringstream stream;
    stream << "Completed in " << std::fixed << std::setprecision(1) << milliseconds
           << " ms. Press S to save.";
    return stream.str();
}

void close_window() noexcept
{
    try
    {
        cv::destroyWindow(kWindowName);
    }
    catch (...)
    {
        // Cleanup must not replace the original UI or inference error.
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::vector<std::string> values;
        values.reserve(static_cast<std::size_t>(argc));
        for (int index = 0; index < argc; ++index)
        {
            values.emplace_back(argv[index]);
        }
        const auto arguments =
            kfcore::face_applications::cli::parse_arguments(values);
        const cv::Mat source = read_bgr(arguments.source, "--source");
        const cv::Mat target = read_bgr(arguments.target, "--target");

        cv::namedWindow(kWindowName, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::resizeWindow(kWindowName, kfcore::face_applications::demo::kCanvasWidth,
                         kfcore::face_applications::demo::kCanvasHeight);
        show(source, target, {}, "Loading TensorRT engines...");

        auto application =
            kfcore::face_applications::TensorRtFaceSwapApplication::load(
                kfcore::face_applications::FaceSwapOptions {});
        cv::Mat result;
        std::string status;
        kfcore::face_applications::demo::TimingHistory timing_history;
        std::vector<kfcore::face_applications::demo::TimingRow> timing_rows;
        const auto run_swap = [&]() {
            show(source, target, result, "Running TensorRT face swap...", timing_rows,
                 timing_history.sample_count());
            auto profiled = application->swap_profiled(
                kfcore::face_applications::demo::borrowed_bgr(source, "source"),
                kfcore::face_applications::demo::borrowed_bgr(target, "target"));
            require_valid_output(profiled.image, target);
            result = kfcore::face_applications::demo::borrowed_bgr(
                         profiled.image, "result").clone();
            status = completed_status(profiled.timings.total);
            timing_history.add(profiled.timings);
            timing_rows = timing_history.rows();
            show(source, target, result, status, timing_rows, timing_history.sample_count());
        };
        run_swap();

        while (cv::getWindowProperty(kWindowName, cv::WND_PROP_VISIBLE) >= 1.0)
        {
            using kfcore::face_applications::demo::DemoAction;
            const DemoAction action = kfcore::face_applications::demo::action_from_key(
                cv::waitKeyEx(kEventPollMilliseconds));
            if (action == DemoAction::Quit)
            {
                break;
            }
            if (action == DemoAction::Rerun)
            {
                run_swap();
                continue;
            }
            if (action == DemoAction::Save)
            {
                require_valid_output(result, target);
                if (!cv::imwrite(arguments.output, result))
                {
                    fail("failed to write output image: " + arguments.output);
                }
                status = "Saved: " + arguments.output;
                show(source, target, result, status, timing_rows,
                     timing_history.sample_count());
            }
        }
        close_window();
        return 0;
    }
    catch (const std::exception& error)
    {
        close_window();
        std::cerr << "face_swap_demo: " << error.what() << '\n';
        return 1;
    }
}
