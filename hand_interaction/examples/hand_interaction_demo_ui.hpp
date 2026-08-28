#pragma once

#include "hand_interaction_demo_capture.hpp"

#include "kfcore/hand_interaction/hand_interaction.hpp"

#include <opencv2/core.hpp>

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
    double                             thig_ms  = 0.0;
    double                             frame_ms = 0.0;
};

[[nodiscard]] DemoAction action_from_key(int key) noexcept;

// Source must be a non-empty CV_8UC3 image. The returned overlay owns its
// pixels and source is never modified.
[[nodiscard]] cv::Mat compose_overlay(
    const cv::Mat& source, const vision_models::HandFrame& hands,
    const HandInteractionFrame& interaction, const DemoMetrics& metrics);

} // namespace kfcore::hand_interaction::demo
