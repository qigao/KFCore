#include "face_swap_cli.hpp"
#include "opencv_image_adapter.hpp"

#include "kfcore/face_applications/tensorrt.hpp"

#include <opencv2/imgcodecs.hpp>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace
{

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
        kfcore::face_applications::FaceApplicationModelPaths paths {
            arguments.detector, arguments.face68, arguments.arcface, arguments.inswapper,
            arguments.matrix, arguments.gfpgan, arguments.age_gender
        };
        auto application =
            kfcore::face_applications::TensorRtFaceSwapApplication::load(paths);
        kfcore::image::BgrImage output = application->swap(
            kfcore::face_applications::demo::borrowed_bgr(source, "source"),
            kfcore::face_applications::demo::borrowed_bgr(target, "target"));
        if (output.width != target.cols || output.height != target.rows)
        {
            fail("swap returned an invalid output image");
        }
        if (!cv::imwrite(arguments.output,
                         kfcore::face_applications::demo::borrowed_bgr(output, "output")))
        {
            fail("failed to write output image: " + arguments.output);
        }
        std::cout << "face_swap_image: wrote " << arguments.output << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "face_swap_image: " << error.what() << '\n';
        return 1;
    }
}
