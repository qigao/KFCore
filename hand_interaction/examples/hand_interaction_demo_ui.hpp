#pragma once

#include "hand_interaction_demo_capture.hpp"

#include "kfcore/hand_interaction/hand_interaction.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

namespace kfcore::hand_interaction::demo
{

enum class DemoAction
{
    None,
    Reset,
    Quit,
};

struct DemoMetrics
{
    CaptureCounters                    capture;
    vision_models::StageTimings        model;
    vision_models::FaceMeshTimings     face;
    double                             fps        = 0.0;
    double                             convert_ms = 0.0;
    double                             thig_ms  = 0.0;
    double                             frame_ms = 0.0;
};

struct HandOverlayText
{
    std::string identity;
    std::string derived;
};

struct DemoThigStatus
{
    std::vector<thig::ActionEvent> recent_actions;
    std::string                    hand_state;
    std::string                    wave_state;
    std::string                    click_state;
};

inline constexpr std::chrono::milliseconds kActionDisplayDuration { 1500 };
inline constexpr std::size_t kMaximumDisplayedActions = 4U;

class RecentActionHistory
{
public:
    explicit RecentActionHistory(
        std::chrono::milliseconds retention = kActionDisplayDuration,
        std::size_t maximum_actions = kMaximumDisplayedActions);

    [[nodiscard]] std::vector<thig::ActionEvent> update(
        const std::vector<thig::ActionEvent>& actions,
        std::chrono::steady_clock::time_point now);
    void reset() noexcept;

private:
    struct Entry
    {
        thig::ActionEvent                    action;
        std::chrono::steady_clock::time_point expires_at;
    };

    std::chrono::milliseconds retention_;
    std::size_t               maximum_actions_;
    std::vector<Entry>        entries_;
};

[[nodiscard]] DemoAction action_from_key(int key) noexcept;

[[nodiscard]] std::array<std::string, 3> format_timing_lines(
    const DemoMetrics& metrics);

[[nodiscard]] HandOverlayText format_hand_overlay_text(
    int canonical_id, const vision_models::HandResult& hand,
    const PrimitiveFrame& primitives);

[[nodiscard]] std::string format_thig_state_line(const DemoThigStatus& status);
[[nodiscard]] std::string format_action_line(const thig::ActionEvent& action);

// Source must be a non-empty CV_8UC3 image. The returned overlay is mirrored,
// owns its pixels, and source is never modified.
[[nodiscard]] cv::Mat compose_overlay(
    const cv::Mat& source, const vision_models::HandFrame& hands,
    const HandInteractionFrame& interaction,
    const DemoThigStatus& status,
    const vision_models::FaceMeshFrame* face, const DemoMetrics& metrics);

} // namespace kfcore::hand_interaction::demo
