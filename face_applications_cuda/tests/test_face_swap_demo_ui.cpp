#include "face_swap_demo_ui.hpp"
#include "face_swap_demo_metrics.hpp"
#include "tinytest.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

using kfcore::face_applications::demo::DemoAction;
using kfcore::face_applications::demo::action_from_key;
using kfcore::face_applications::demo::compose_canvas;
using kfcore::face_applications::demo::TimingHistory;
using kfcore::face_applications::FaceSwapDuration;
using kfcore::face_applications::FaceSwapTimingReport;

namespace
{

bool identical(const cv::Mat& left, const cv::Mat& right)
{
    cv::Mat difference;
    cv::compare(left, right, difference, cv::CMP_NE);
    return cv::countNonZero(difference.reshape(1)) == 0;
}

FaceSwapTimingReport report_with_total(std::chrono::milliseconds total)
{
    FaceSwapTimingReport report;
    report.total = total;
    report.source_analysis.total = total / 2;
    return report;
}

} // namespace

spec("face swap demo UI")
{
    it("reports the latest value and nearest-rank percentiles")
    {
        TimingHistory history;
        history.add(report_with_total(std::chrono::milliseconds(30)));
        history.add(report_with_total(std::chrono::milliseconds(10)));
        history.add(report_with_total(std::chrono::milliseconds(20)));

        const auto rows = history.rows();
        const auto total = std::find_if(rows.begin(), rows.end(), [](const auto& row) {
            return row.label == "Total";
        });
        check(total != rows.end());
        check(total->current == FaceSwapDuration(std::chrono::milliseconds(20)));
        check(total->p50 == FaceSwapDuration(std::chrono::milliseconds(20)));
        check(total->p95 == FaceSwapDuration(std::chrono::milliseconds(30)));
        check(history.sample_count() == 3U);
    }

    it("keeps optional stages absent until samples contain them")
    {
        TimingHistory history;
        history.add(report_with_total(std::chrono::milliseconds(10)));
        auto rows = history.rows();
        auto gfpgan = std::find_if(rows.begin(), rows.end(), [](const auto& row) {
            return row.label == "GFPGAN preprocess";
        });
        check(gfpgan != rows.end());
        check_false(gfpgan->current.has_value());
        check_false(gfpgan->p50.has_value());
        check_false(gfpgan->p95.has_value());

        FaceSwapTimingReport report = report_with_total(std::chrono::milliseconds(11));
        report.gfpgan_preprocess = std::chrono::milliseconds(4);
        history.add(report);
        rows = history.rows();
        gfpgan = std::find_if(rows.begin(), rows.end(), [](const auto& row) {
            return row.label == "GFPGAN preprocess";
        });
        check(gfpgan->current == FaceSwapDuration(std::chrono::milliseconds(4)));
        check(gfpgan->p50 == FaceSwapDuration(std::chrono::milliseconds(4)));
        check(gfpgan->p95 == FaceSwapDuration(std::chrono::milliseconds(4)));
    }

    it("bounds percentile history to the latest 120 samples")
    {
        TimingHistory history;
        for (int value = 1; value <= 121; ++value)
        {
            history.add(report_with_total(std::chrono::milliseconds(value)));
        }
        const auto rows = history.rows();
        const auto total = std::find_if(rows.begin(), rows.end(), [](const auto& row) {
            return row.label == "Total";
        });
        check(history.sample_count() == 120U);
        check(total->current == FaceSwapDuration(std::chrono::milliseconds(121)));
        check(total->p50 == FaceSwapDuration(std::chrono::milliseconds(61)));
        check(total->p95 == FaceSwapDuration(std::chrono::milliseconds(115)));
    }

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

    it("renders timing rows in the expanded diagnostics footer")
    {
        const cv::Mat image(32, 32, CV_8UC3, cv::Scalar::all(0));
        TimingHistory history;
        history.add(report_with_total(std::chrono::milliseconds(42)));

        const cv::Mat canvas = compose_canvas(image, image, image, "Completed",
                                              history.rows(), history.sample_count());

        check_false(canvas.empty());
        check(canvas.cols == kfcore::face_applications::demo::kCanvasWidth);
        check(canvas.rows == kfcore::face_applications::demo::kCanvasHeight);
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
