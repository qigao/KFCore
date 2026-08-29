#pragma once

#include "yolo_domain_capture.hpp"
#include "yolo_domain_profile.hpp"
#include "kfcore/image_processor/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace kfcore::yolo::demo
{

struct FrameTimings
{
    double capture_wait_ms = 0.0;
    double convert_ms = 0.0;
    double detect_ms = 0.0;
    double track_ms = 0.0;
    double render_ms = 0.0;
    double total_ms = 0.0;
};

struct StageStatistics
{
    double current_ms = 0.0;
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
};

struct MetricsSnapshot
{
    std::size_t sample_count = 0U;
    StageStatistics capture_wait;
    StageStatistics convert;
    StageStatistics detect;
    StageStatistics track;
    StageStatistics render;
    StageStatistics total;
};

class TimingWindow final
{
public:
    explicit TimingWindow(std::size_t capacity);
    void add(const FrameTimings& timings);
    [[nodiscard]] MetricsSnapshot snapshot() const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;

private:
    std::vector<FrameTimings> samples_;
    std::size_t               capacity_ = 0U;
    std::size_t               next_ = 0U;
};

enum class KeyAction
{
    None,
    Quit,
    Reset,
};

struct OverlayState
{
    const DomainProfile* profile = nullptr;
    const TrackFrame*    tracks = nullptr;
    const DomainSummary* summary = nullptr;
    std::string          backend;
    double               model_load_ms = 0.0;
    double               fps = 0.0;
    MetricsSnapshot      metrics;
    CaptureCounters      capture;
};

[[nodiscard]] std::string track_label(const DomainProfile& profile,
                                      const TrackedDetection& tracked);
[[nodiscard]] std::string format_metrics(const MetricsSnapshot& snapshot);
void update_current_metrics(MetricsSnapshot& snapshot,
                            const FrameTimings& timings,
                            std::size_t sample_count);
[[nodiscard]] KeyAction decode_key(int key) noexcept;
void draw_overlay(kfcore::image::BgrImage& image, const OverlayState& state);

} // namespace kfcore::yolo::demo
