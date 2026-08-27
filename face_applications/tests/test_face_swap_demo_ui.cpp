#include "face_swap_demo_ui.hpp"
#include "tinytest.hpp"

#include <opencv2/core.hpp>

#include <stdexcept>
#include <string>

using kfcore::face_applications::demo::DemoAction;
using kfcore::face_applications::demo::action_from_key;
using kfcore::face_applications::demo::compose_canvas;

namespace
{

bool identical(const cv::Mat& left, const cv::Mat& right)
{
    cv::Mat difference;
    cv::compare(left, right, difference, cv::CMP_NE);
    return cv::countNonZero(difference.reshape(1)) == 0;
}

} // namespace

spec("face swap demo UI")
{
    it("maps supported keys to explicit actions")
    {
        check(action_from_key('q') == DemoAction::Quit);
        check(action_from_key('Q') == DemoAction::Quit);
        check(action_from_key(27) == DemoAction::Quit);
        check(action_from_key('r') == DemoAction::Rerun);
        check(action_from_key('R') == DemoAction::Rerun);
        check(action_from_key('s') == DemoAction::Save);
        check(action_from_key('S') == DemoAction::Save);
        check(action_from_key(-1) == DemoAction::None);
        check(action_from_key('x') == DemoAction::None);
    }

    it("composes source target and a pending result without mutating inputs")
    {
        const cv::Mat source(48, 96, CV_8UC3, cv::Scalar(10, 20, 30));
        const cv::Mat target(96, 48, CV_8UC3, cv::Scalar(40, 50, 60));
        const cv::Mat source_before = source.clone();
        const cv::Mat target_before = target.clone();

        const cv::Mat canvas = compose_canvas(source, target, {}, "Loading engines...");

        check_false(canvas.empty());
        check(canvas.type() == CV_8UC3);
        check(canvas.cols == kfcore::face_applications::demo::kCanvasWidth);
        check(canvas.rows == kfcore::face_applications::demo::kCanvasHeight);
        check_true(identical(source, source_before));
        check_true(identical(target, target_before));
    }

    it("rejects invalid source target and result image types")
    {
        const cv::Mat bgr(32, 32, CV_8UC3, cv::Scalar::all(0));
        const cv::Mat gray(32, 32, CV_8UC1, cv::Scalar::all(0));
        bool source_threw = false;
        bool target_threw = false;
        bool result_threw = false;
        try
        {
            (void)compose_canvas(gray, bgr, {}, "status");
        }
        catch (const std::invalid_argument&)
        {
            source_threw = true;
        }
        try
        {
            (void)compose_canvas(bgr, gray, {}, "status");
        }
        catch (const std::invalid_argument&)
        {
            target_threw = true;
        }
        try
        {
            (void)compose_canvas(bgr, bgr, gray, "status");
        }
        catch (const std::invalid_argument&)
        {
            result_threw = true;
        }
        check_true(source_threw);
        check_true(target_threw);
        check_true(result_threw);
    }
}
