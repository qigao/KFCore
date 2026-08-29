#include "yolo_domain_ui.hpp"

#include <stb_easy_font.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace kfcore::yolo::demo
{
namespace
{

constexpr int kHeaderHeight = 123;
constexpr int kLineHeight = 23;
constexpr int kBoxThickness = 2;
constexpr int kFontScale = 2;
constexpr std::size_t kMaximumTextCharacters = 220U;
constexpr std::size_t kFontVertexBufferBytes = 64U * 1024U;

using Color = std::array<std::uint8_t, 3>;

struct FontVertex
{
    float x;
    float y;
    float z;
    std::uint8_t color[4];
};

void validate_image(const kfcore::image::BgrImage& image)
{
    if (image.width <= 0 || image.height <= 0)
    {
        throw std::invalid_argument("overlay image dimensions must be positive");
    }
    const std::size_t width = static_cast<std::size_t>(image.width);
    const std::size_t height = static_cast<std::size_t>(image.height);
    if (height > (std::numeric_limits<std::size_t>::max)() / width ||
        width * height > (std::numeric_limits<std::size_t>::max)() / 3U ||
        image.pixels.size() != width * height * 3U)
    {
        throw std::invalid_argument("overlay image storage is malformed");
    }
}

void fill_rectangle(kfcore::image::BgrImage& image, int left, int top,
                    int right, int bottom, const Color& color)
{
    left = (std::clamp)(left, 0, image.width);
    right = (std::clamp)(right, 0, image.width);
    top = (std::clamp)(top, 0, image.height);
    bottom = (std::clamp)(bottom, 0, image.height);
    if (left >= right || top >= bottom)
    {
        return;
    }
    for (int y = top; y < bottom; ++y)
    {
        std::uint8_t* pixel = image.pixels.data() +
            (static_cast<std::size_t>(y) * image.width + left) * 3U;
        for (int x = left; x < right; ++x)
        {
            pixel[0] = color[0];
            pixel[1] = color[1];
            pixel[2] = color[2];
            pixel += 3;
        }
    }
}

void stroke_rectangle(kfcore::image::BgrImage& image, int left, int top,
                      int right, int bottom, const Color& color)
{
    fill_rectangle(image, left, top, right, top + kBoxThickness, color);
    fill_rectangle(image, left, bottom - kBoxThickness, right, bottom, color);
    fill_rectangle(image, left, top, left + kBoxThickness, bottom, color);
    fill_rectangle(image, right - kBoxThickness, top, right, bottom, color);
}

int text_width(const std::string& text)
{
    if (text.size() > kMaximumTextCharacters)
    {
        throw std::length_error("overlay text exceeds the configured character limit");
    }
    std::array<char, kMaximumTextCharacters + 1U> ascii {};
    for (std::size_t index = 0U; index < text.size(); ++index)
    {
        const unsigned char character = static_cast<unsigned char>(text[index]);
        ascii[index] = character >= 32U && character <= 126U
                           ? static_cast<char>(character)
                           : '?';
    }
    return stb_easy_font_width(ascii.data()) * kFontScale;
}

void draw_text(kfcore::image::BgrImage& image, const std::string& text,
               int x, int y, const Color& color)
{
    if (text.size() > kMaximumTextCharacters)
    {
        throw std::length_error("overlay text exceeds the configured character limit");
    }
    std::array<char, kMaximumTextCharacters + 1U> ascii {};
    for (std::size_t index = 0U; index < text.size(); ++index)
    {
        const unsigned char character = static_cast<unsigned char>(text[index]);
        ascii[index] = character >= 32U && character <= 126U
                           ? static_cast<char>(character)
                           : '?';
    }
    alignas(float) std::array<std::byte, kFontVertexBufferBytes> vertices {};
    const int quads = stb_easy_font_print(
        0.0F, 0.0F, ascii.data(), nullptr, vertices.data(),
        static_cast<int>(vertices.size()));
    const auto* typed = reinterpret_cast<const FontVertex*>(vertices.data());
    for (int quad = 0; quad < quads; ++quad)
    {
        float minimum_x = typed[quad * 4].x;
        float maximum_x = minimum_x;
        float minimum_y = typed[quad * 4].y;
        float maximum_y = minimum_y;
        for (int vertex = 1; vertex < 4; ++vertex)
        {
            const FontVertex& value = typed[quad * 4 + vertex];
            minimum_x = (std::min)(minimum_x, value.x);
            maximum_x = (std::max)(maximum_x, value.x);
            minimum_y = (std::min)(minimum_y, value.y);
            maximum_y = (std::max)(maximum_y, value.y);
        }
        fill_rectangle(
            image, x + static_cast<int>(std::floor(minimum_x * kFontScale)),
            y + static_cast<int>(std::floor(minimum_y * kFontScale)),
            x + static_cast<int>(std::ceil(maximum_x * kFontScale)),
            y + static_cast<int>(std::ceil(maximum_y * kFontScale)), color);
    }
}

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

Color track_color(const TrackedDetection& tracked)
{
    static const std::array<Color, 8> palette {
        Color { 0, 220, 255 }, Color { 255, 170, 0 },
        Color { 70, 255, 70 }, Color { 255, 80, 220 },
        Color { 30, 150, 255 }, Color { 255, 220, 40 },
        Color { 180, 90, 255 }, Color { 80, 255, 210 }
    };
    const std::uint64_t key = tracked.track_id.value_or(
        static_cast<std::uint64_t>(tracked.detection.class_id));
    return palette[static_cast<std::size_t>(key % palette.size())];
}

void draw_text_line(kfcore::image::BgrImage& image, const std::string& text,
                    int line, const Color& color)
{
    draw_text(image, text, 10, 4 + line * kLineHeight, color);
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

void draw_overlay(kfcore::image::BgrImage& image, const OverlayState& state)
{
    validate_image(image);
    if (state.profile == nullptr || state.tracks == nullptr ||
        state.summary == nullptr)
    {
        throw std::invalid_argument("overlay image and state pointers must be valid");
    }
    if (state.tracks->image_width != image.width ||
        state.tracks->image_height != image.height)
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
        const Color color = track_color(tracked);
        stroke_rectangle(image, left, top, right, bottom, color);
        const std::string label = track_label(*state.profile, tracked);
        constexpr int kFontHeight = 12 * kFontScale;
        const int label_top = (std::max)(0, top - kFontHeight - 6);
        const int label_right = (std::min)(image.width, left + text_width(label) + 8);
        fill_rectangle(image, left, label_top, label_right,
                       (std::min)(image.height, top), Color { 12, 12, 12 });
        draw_text(image, label, left + 4, label_top + 2, Color { 255, 255, 255 });
    }

    fill_rectangle(image, 0, 0, image.width,
                   (std::min)(kHeaderHeight, image.height), Color { 18, 18, 18 });
    std::ostringstream first;
    first << state.profile->name << " backend=" << state.backend
          << " FPS=" << std::fixed << std::setprecision(1) << state.fps
          << " load=" << std::setprecision(2) << state.model_load_ms << "ms";
    draw_text_line(image, first.str(), 0, Color { 90, 255, 255 });
    draw_text_line(image, format_summary(*state.profile, *state.summary), 1,
                   Color { 255, 255, 255 });
    draw_text_line(image, current_metrics_line(state.metrics), 2,
                   Color { 120, 255, 120 });
    draw_text_line(image, percentile_metrics_line(state.metrics), 3,
                   Color { 120, 255, 120 });
    std::ostringstream capture;
    capture << "capture=" << state.capture.captured_frames
            << " consumed=" << state.capture.consumed_frames
            << " coalesced=" << state.capture.coalesced_frames
            << " rejected=" << state.capture.rejected_frames;
    draw_text_line(image, capture.str(), 4, Color { 255, 210, 120 });
}

} // namespace kfcore::yolo::demo
