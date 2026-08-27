#include "kfcore/face_applications/tensorrt.hpp"
#include "tinytest.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <string>

using namespace kfcore::face_applications;

namespace
{

bool identical(const cv::Mat& left, const cv::Mat& right)
{
    if (left.size() != right.size() || left.type() != right.type())
    {
        return false;
    }
    cv::Mat difference;
    cv::compare(left, right, difference, cv::CMP_NE);
    return cv::countNonZero(difference.reshape(1)) == 0;
}

void verify_swap(const cv::Mat& source, const cv::Mat& target,
                 const FaceApplicationModelPaths& paths)
{
    const cv::Mat source_before = source.clone();
    const cv::Mat target_before = target.clone();
    auto application = TensorRtFaceSwapApplication::load(paths);
    const cv::Mat output = application->swap(source, target);

    check_false(output.empty());
    check(output.type() == CV_8UC3);
    check(output.size() == target.size());
    check_true(identical(source, source_before));
    check_true(identical(target, target_before));

    char* output_path = tt_make_temp_file("face-swap-e2e", ".png");
    check_not_null(output_path);
    if (output_path != nullptr)
    {
        check_true(cv::imwrite(output_path, output));
        const cv::Mat decoded = cv::imread(output_path, cv::IMREAD_COLOR);
        check_false(decoded.empty());
        check(decoded.type() == CV_8UC3);
        check(decoded.size() == target.size());
        (void)tt_remove_file(output_path);
        std::free(output_path);
    }
}

FaceApplicationModelPaths required_paths()
{
    return { KFCORE_TEST_12FACE_ENGINE, KFCORE_TEST_FACE68_ENGINE,
             KFCORE_TEST_ARCFACE_ENGINE, KFCORE_TEST_INSWAPPER_ENGINE,
             KFCORE_TEST_INSWAPPER_MATRIX, std::nullopt, std::nullopt };
}

} // namespace

spec("TensorRT face swap application integration")
{
    it("runs YOLOv12-face through InSwapper without mutating inputs")
    {
        const cv::Mat source = cv::imread(KFCORE_TEST_SOURCE_IMAGE, cv::IMREAD_COLOR);
        const cv::Mat target = cv::imread(KFCORE_TEST_TARGET_IMAGE, cv::IMREAD_COLOR);
        check_false(source.empty());
        check_false(target.empty());
        verify_swap(source, target, required_paths());
    }

    it("runs the explicit GFPGAN enhancement path when configured")
    {
        const std::string gfpgan_path = KFCORE_TEST_GFPGAN_ENGINE;
        if (!gfpgan_path.empty())
        {
            const cv::Mat source = cv::imread(KFCORE_TEST_SOURCE_IMAGE, cv::IMREAD_COLOR);
            const cv::Mat target = cv::imread(KFCORE_TEST_TARGET_IMAGE, cv::IMREAD_COLOR);
            FaceApplicationModelPaths paths = required_paths();
            paths.gfpgan_engine = gfpgan_path;
            verify_swap(source, target, paths);
        }
    }
}
