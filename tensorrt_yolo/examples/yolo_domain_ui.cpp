#include "yolo_domain_ui.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace kfcore::yolo::demo
{
namespace
{

constexpr int kHeaderHeight = 123;
constexpr int kLineHeight = 23;
constexpr double kFontScale = 0.55;
constexpr int kTextThickness = 1;
constexpr int kBoxThickness = 2;

void validate_timing(const FrameTimings& timings)
{
    const std::array<double, 6> values {
        timings.capture_wait_ms, timings.convert_ms, timings.detect_ms,
        timings.track_ms, timings.render_ms, timings.total_ms
    };
    for (double value : values)
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            throw std::invalid_argument(
                "frame timing values must be finite and non-negative");
        }
    }
}

StageStatistics statistics(const std::vector<FrameTimings>& samples,
                           std::size_t current_index,
                           double FrameTimings::*member)
{
    StageStatistics result;
    result.current_ms = samples[current_index].*member;
    std::vector<double> sorted;
    sorted.reserve(samples.size());
    for (const FrameTimings& sample : samples)
    {
        sorted.push_back(sample.*member);
    }
    result.mean_ms = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
                     static_cast<double>(sorted.size());
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double fraction)
    {
        const std::size_t rank = static_cast<std::size_t>(
            std::ceil(fraction * static_cast<double>(sorted.size())));
        return sorted[(std::max)(std::size_t { 1U }, rank) - 1U];
    };
    result.p50_ms = percentile(0.50);
    result.p95_ms = percentile(0.95);
    return result;
}

cv::Scalar track_color(const TrackedDetection& tracked)
{
    static const std::array<cv::Scalar, 8> palette {
        cv::Scalar(0, 220, 255), cv::Scalar(255, 170, 0),
        cv::Scalar(70, 255, 70), cv::Scalar(255, 80, 220),
        cv::Scalar(30, 150, 255), cv::Scalar(255, 220, 40),
        cv::Scalar(180, 90, 255), cv::Scalar(80, 255, 210)
    };
    const std::uint64_t key = tracked.track_id.value_or(
        static_cast<std::uint64_t>(tracked.detection.class_id));
    return palette[static_cast<std::size_t>(key % palette.size())];
}

void draw_text_line(cv::Mat& image, const std::string& text, int line,
                    const cv::Scalar& color)
{
    cv::putText(image, text, cv::Point(10, 20 + line * kLineHeight),
                cv::FONT_HERSHEY_SIMPLEX, kFontScale, color, kTextThickness,
                cv::LINE_AA);
}

std::string current_metrics_line(const MetricsSnapshot& metrics)
{
    std::ostringstream output;
    output << "cur ms cap=" << std::fixed << std::setprecision(1)
           << metrics.capture_wait.current_ms << " cvt=" << metrics.convert.current_ms
           << " det=" << metrics.detect.current_ms << " trk="
           << metrics.track.current_ms << " rnd=" << metrics.render.current_ms
           << " all=" << metrics.total.current_ms;
    return output.str();
}

std::string percentile_metrics_line(const MetricsSnapshot& metrics)
{
    std::ostringstream output;
    output << "P50/P95 ms detect=" << std::fixed << std::setprecision(1)
           << metrics.detect.p50_ms << '/' << metrics.detect.p95_ms
           << " total=" << metrics.total.p50_ms << '/' << metrics.total.p95_ms
           << " samples=" << metrics.sample_count;
    return output.str();
}

} // namespace

TimingWindow::TimingWindow(std::size_t capacity)
    : capacity_(capacity)
{
    if (capacity_ == 0U)
    {
        throw std::invalid_argument("timing window capacity must be positive");
    }
    samples_.reserve(capacity_);
}

void TimingWindow::add(const FrameTimings& timings)
{
    validate_timing(timings);
    if (samples_.size() < capacity_)
    {
        samples_.push_back(timings);
        next_ = samples_.size() == capacity_ ? 0U : samples_.size();
    }
    else
    {
        samples_[next_] = timings;
        next_ = (next_ + 1U) % capacity_;
    }
}

MetricsSnapshot TimingWindow::snapshot() const
{
    MetricsSnapshot result;
    result.sample_count = samples_.size();
    if (samples_.empty())
    {
        return result;
    }
    const std::size_t current_index =
        samples_.size() < capacity_ ? samples_.size() - 1U
                                    : (next_ + capacity_ - 1U) % capacity_;
    result.capture_wait = statistics(samples_, current_index,
                                     &FrameTimings::capture_wait_ms);
    result.convert = statistics(samples_, current_index, &FrameTimings::convert_ms);
    result.detect = statistics(samples_, current_index, &FrameTimings::detect_ms);
    result.track = statistics(samples_, current_index, &FrameTimings::track_ms);
    result.render = statistics(samples_, current_index, &FrameTimings::render_ms);
    result.total = statistics(samples_, current_index, &FrameTimings::total_ms);
    return result;
}

std::size_t TimingWindow::size() const noexcept
{
    return samples_.size();
}

std::size_t TimingWindow::capacity() const noexcept
{
    return capacity_;
}

std::string track_label(const DomainProfile& profile,
                        const TrackedDetection& tracked)
{
    std::ostringstream output;
    output << class_label(profile, tracked.detection.class_id) << ' '
           << std::fixed << std::setprecision(1)
           << tracked.detection.score * 100.0F << "% track=";
    if (tracked.track_id.has_value())
    {
        output << *tracked.track_id;
    }
    else
    {
        output << "pending";
    }
    return output.str();
}

std::string format_metrics(const MetricsSnapshot& snapshot)
{
    const auto stage = [](const char* name, const StageStatistics& value)
    {
        std::ostringstream output;
        output << name << '=' << std::fixed << std::setprecision(2)
               << value.current_ms << '/' << value.mean_ms << '/'
               << value.p50_ms << '/' << value.p95_ms << "ms";
        return output.str();
    };
    return stage("capture", snapshot.capture_wait) + " " +
           stage("convert", snapshot.convert) + " " +
           stage("detect", snapshot.detect) + " " +
           stage("track", snapshot.track) + " " +
           stage("render", snapshot.render) + " " +
           stage("total", snapshot.total);
}

void update_current_metrics(MetricsSnapshot& snapshot,
                            const FrameTimings& timings,
                            std::size_t sample_count)
{
    validate_timing(timings);
    if (sample_count == 0U)
    {
        throw std::invalid_argument("metric sample count must be positive");
    }
    snapshot.sample_count = sample_count;
    snapshot.capture_wait.current_ms = timings.capture_wait_ms;
    snapshot.convert.current_ms = timings.convert_ms;
    snapshot.detect.current_ms = timings.detect_ms;
    snapshot.track.current_ms = timings.track_ms;
    snapshot.render.current_ms = timings.render_ms;
    snapshot.total.current_ms = timings.total_ms;
}

KeyAction decode_key(int key) noexcept
{
    if (key == 27 || key == 'q' || key == 'Q')
    {
        return KeyAction::Quit;
    }
    if (key == 'r' || key == 'R')
    {
        return KeyAction::Reset;
    }
    return KeyAction::None;
}

void draw_overlay(cv::Mat& image, const OverlayState& state)
{
    if (image.empty() || image.type() != CV_8UC3 || state.profile == nullptr ||
        state.tracks == nullptr || state.summary == nullptr)
    {
        throw std::invalid_argument("overlay image and state pointers must be valid");
    }
    if (state.tracks->image_width != image.cols ||
        state.tracks->image_height != image.rows)
    {
        throw std::invalid_argument("overlay track frame dimensions do not match the image");
    }

    for (const TrackedDetection& tracked : state.tracks->detections)
    {
        const BoxF& box = tracked.detection.box;
        const int left = static_cast<int>(std::floor(box.left));
        const int top = static_cast<int>(std::floor(box.top));
        const int right = static_cast<int>(std::ceil(box.right));
        const int bottom = static_cast<int>(std::ceil(box.bottom));
        const cv::Scalar color = track_color(tracked);
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom),
                      color, kBoxThickness, cv::LINE_AA);
        const std::string label = track_label(*state.profile, tracked);
        int baseline = 0;
        const cv::Size size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                              kFontScale, kTextThickness, &baseline);
        const int label_top = (std::max)(0, top - size.height - 8);
        const int label_right = (std::min)(image.cols - 1, left + size.width + 8);
        cv::rectangle(image, cv::Point(left, label_top),
                      cv::Point(label_right, (std::min)(image.rows - 1, top)),
                      cv::Scalar(12, 12, 12), cv::FILLED);
        cv::putText(image, label, cv::Point(left + 4, (std::max)(size.height + 2, top - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, kFontScale, cv::Scalar(255, 255, 255),
                    kTextThickness, cv::LINE_AA);
    }

    cv::rectangle(image, cv::Point(0, 0),
                  cv::Point(image.cols - 1, (std::min)(kHeaderHeight, image.rows - 1)),
                  cv::Scalar(18, 18, 18), cv::FILLED);
    std::ostringstream first;
    first << state.profile->name << " backend=" << state.backend
          << " FPS=" << std::fixed << std::setprecision(1) << state.fps
          << " load=" << std::setprecision(2) << state.model_load_ms << "ms";
    draw_text_line(image, first.str(), 0, cv::Scalar(90, 255, 255));
    draw_text_line(image, format_summary(*state.profile, *state.summary), 1,
                   cv::Scalar(255, 255, 255));
    draw_text_line(image, current_metrics_line(state.metrics), 2,
                   cv::Scalar(120, 255, 120));
    draw_text_line(image, percentile_metrics_line(state.metrics), 3,
                   cv::Scalar(120, 255, 120));
    std::ostringstream capture;
    capture << "capture=" << state.capture.captured_frames
            << " consumed=" << state.capture.consumed_frames
            << " coalesced=" << state.capture.coalesced_frames
            << " rejected=" << state.capture.rejected_frames;
    draw_text_line(image, capture.str(), 4, cv::Scalar(255, 210, 120));
}

} // namespace kfcore::yolo::demo
