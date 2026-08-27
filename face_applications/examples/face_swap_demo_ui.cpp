#include "face_swap_demo_ui.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace kfcore::face_applications::demo
{
namespace
{

const cv::Scalar kCanvasColor { 24, 24, 24 };
const cv::Scalar kPanelColor { 40, 40, 40 };
const cv::Scalar kPrimaryTextColor { 235, 235, 235 };
const cv::Scalar kSecondaryTextColor { 165, 190, 210 };
constexpr double kLabelScale = 0.62;
constexpr double kStatusScale = 0.52;
constexpr double kTimingScale = 0.39;
constexpr int kTextThickness = 1;
constexpr int kPanelTop = kCanvasGap + kHeaderHeight;
constexpr int kTimingLineHeight = 18;

void require_bgr(const cv::Mat& image, const char* role, bool allow_empty)
{
    if (image.empty())
    {
        if (allow_empty)
        {
            return;
        }
        throw std::invalid_argument(std::string(role) + " image is empty");
    }
    if (image.type() != CV_8UC3)
    {
        throw std::invalid_argument(std::string(role) + " image must be CV_8UC3");
    }
}

void render_panel(cv::Mat& canvas, const cv::Mat& image, int left, const char* label)
{
    cv::putText(canvas, label, cv::Point(left, kCanvasGap + 27), cv::FONT_HERSHEY_SIMPLEX,
                kLabelScale, kPrimaryTextColor, kTextThickness, cv::LINE_AA);

    const cv::Rect panel_bounds(left, kPanelTop, kPanelWidth, kPanelHeight);
    cv::Mat panel = canvas(panel_bounds);
    panel.setTo(kPanelColor);
    if (image.empty())
    {
        constexpr const char* kPendingText = "Pending inference";
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(kPendingText, cv::FONT_HERSHEY_SIMPLEX,
                                                   kLabelScale, kTextThickness, &baseline);
        const cv::Point origin((kPanelWidth - text_size.width) / 2,
                               (kPanelHeight + text_size.height) / 2);
        cv::putText(panel, kPendingText, origin, cv::FONT_HERSHEY_SIMPLEX, kLabelScale,
                    kSecondaryTextColor, kTextThickness, cv::LINE_AA);
        return;
    }

    const double scale =
        std::min(static_cast<double>(kPanelWidth) / static_cast<double>(image.cols),
                 static_cast<double>(kPanelHeight) / static_cast<double>(image.rows));
    const cv::Size display_size(
        std::max(1, static_cast<int>(std::lround(static_cast<double>(image.cols) * scale))),
        std::max(1, static_cast<int>(std::lround(static_cast<double>(image.rows) * scale))));
    cv::Mat display;
    cv::resize(image, display, display_size, 0.0, 0.0, cv::INTER_AREA);
    const int x = (kPanelWidth - display.cols) / 2;
    const int y = (kPanelHeight - display.rows) / 2;
    display.copyTo(panel(cv::Rect(x, y, display.cols, display.rows)));
}

std::string format_duration(const std::optional<FaceSwapDuration>& duration)
{
    if (!duration)
    {
        return "--";
    }
    const double milliseconds =
        std::chrono::duration<double, std::milli>(*duration).count();
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << milliseconds;
    return stream.str();
}

void render_timings(cv::Mat& canvas, int footer_top,
                    const std::vector<TimingRow>& timing_rows,
                    std::size_t timing_samples)
{
    if (timing_rows.empty())
    {
        return;
    }

    const std::string heading = "STAGE  current / P50 / P95 (ms)    samples: " +
                                std::to_string(timing_samples);
    constexpr int kHeadingOffset = 82;
    cv::putText(canvas, heading, cv::Point(kCanvasGap, footer_top + kHeadingOffset),
                cv::FONT_HERSHEY_SIMPLEX, kTimingScale, kPrimaryTextColor,
                kTextThickness, cv::LINE_AA);

    const std::size_t rows_per_column = (timing_rows.size() + 1U) / 2U;
    const int column_width = kCanvasWidth / 2;
    constexpr int kFirstRowOffset = kHeadingOffset + 23;
    for (std::size_t index = 0U; index < timing_rows.size(); ++index)
    {
        const std::size_t column = index / rows_per_column;
        const std::size_t row = index % rows_per_column;
        const TimingRow& timing = timing_rows[index];
        const std::string text = std::string(timing.label) + "  " +
                                 format_duration(timing.current) + " / " +
                                 format_duration(timing.p50) + " / " +
                                 format_duration(timing.p95);
        const cv::Point origin(
            kCanvasGap + static_cast<int>(column) * column_width,
            footer_top + kFirstRowOffset + static_cast<int>(row) * kTimingLineHeight);
        cv::putText(canvas, text, origin, cv::FONT_HERSHEY_SIMPLEX, kTimingScale,
                    kSecondaryTextColor, kTextThickness, cv::LINE_AA);
    }
}

} // namespace

DemoAction action_from_key(int key) noexcept
{
    if (key < 0)
    {
        return DemoAction::None;
    }
    switch (key & 0xff)
    {
    case 27:
    case 'q':
    case 'Q':
        return DemoAction::Quit;
    case 'r':
    case 'R':
        return DemoAction::Rerun;
    case 's':
    case 'S':
        return DemoAction::Save;
    default:
        return DemoAction::None;
    }
}

cv::Mat compose_canvas(const cv::Mat& source, const cv::Mat& target,
                       const cv::Mat& result, std::string_view status,
                       const std::vector<TimingRow>& timing_rows,
                       std::size_t timing_samples)
{
    require_bgr(source, "source", false);
    require_bgr(target, "target", false);
    require_bgr(result, "result", true);

    cv::Mat canvas(kCanvasHeight, kCanvasWidth, CV_8UC3, kCanvasColor);
    const std::array<const cv::Mat*, 3> images { &source, &target, &result };
    constexpr std::array<const char*, 3> kLabels { "SOURCE FACE", "TARGET IMAGE", "RESULT" };
    for (std::size_t index = 0; index < images.size(); ++index)
    {
        const int left = kCanvasGap + static_cast<int>(index) * (kPanelWidth + kCanvasGap);
        render_panel(canvas, *images[index], left, kLabels[index]);
    }

    const int footer_top = kPanelTop + kPanelHeight;
    cv::putText(canvas, "R  rerun     S  save     Q / Esc  quit",
                cv::Point(kCanvasGap, footer_top + 26), cv::FONT_HERSHEY_SIMPLEX,
                kStatusScale, kPrimaryTextColor, kTextThickness, cv::LINE_AA);
    cv::putText(canvas, std::string(status), cv::Point(kCanvasGap, footer_top + 49),
                cv::FONT_HERSHEY_SIMPLEX, kStatusScale, kSecondaryTextColor,
                kTextThickness, cv::LINE_AA);
    render_timings(canvas, footer_top, timing_rows, timing_samples);
    return canvas;
}

} // namespace kfcore::face_applications::demo
