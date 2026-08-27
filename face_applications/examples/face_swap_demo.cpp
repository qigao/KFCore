#include "face_swap_cli.hpp"
#include "face_swap_demo_ui.hpp"

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

void require_valid_output(const cv::Mat& output, const cv::Mat& target)
{
    if (output.empty() || output.type() != CV_8UC3 || output.size() != target.size())
    {
        fail("swap returned an invalid output image");
    }
}

void show(const cv::Mat& source, const cv::Mat& target, const cv::Mat& result,
          const std::string& status)
{
    cv::imshow(kWindowName,
               kfcore::face_applications::demo::compose_canvas(source, target, result, status));
    (void)cv::waitKey(kDisplayRefreshMilliseconds);
}

std::string completed_status(std::chrono::steady_clock::duration elapsed)
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
        const auto arguments = kfcore::face_applications::cli::parse_arguments(values);
        const cv::Mat source = read_bgr(arguments.source, "--source");
        const cv::Mat target = read_bgr(arguments.target, "--target");

        cv::namedWindow(kWindowName, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::resizeWindow(kWindowName, kfcore::face_applications::demo::kCanvasWidth,
                         kfcore::face_applications::demo::kCanvasHeight);
        show(source, target, {}, "Loading TensorRT engines...");

        kfcore::face_applications::FaceApplicationModelPaths paths {
            arguments.detector, arguments.face68, arguments.arcface, arguments.inswapper,
            arguments.matrix, arguments.gfpgan, arguments.age_gender
        };
        auto application = kfcore::face_applications::TensorRtFaceSwapApplication::load(paths);
        cv::Mat result;
        std::string status;
        const auto run_swap = [&]() {
            show(source, target, result, "Running TensorRT face swap...");
            const auto started = std::chrono::steady_clock::now();
            cv::Mat next = application->swap(source, target);
            const auto finished = std::chrono::steady_clock::now();
            require_valid_output(next, target);
            result = std::move(next);
            status = completed_status(finished - started);
            show(source, target, result, status);
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
                show(source, target, result, status);
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
