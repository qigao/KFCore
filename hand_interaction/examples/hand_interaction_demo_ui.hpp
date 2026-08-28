#pragma once

#include "hand_interaction_demo_capture.hpp"

#include "kfcore/hand_interaction/hand_interaction.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <string>

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
    double                             convert_ms = 0.0;
    double                             thig_ms  = 0.0;
    double                             frame_ms = 0.0;
};

[[nodiscard]] DemoAction action_from_key(int key) noexcept;

[[nodiscard]] std::array<std::string, 3> format_timing_lines(
    const DemoMetrics& metrics);

// Source must be a non-empty CV_8UC3 image. The returned overlay owns its
// pixels and source is never modified.
[[nodiscard]] cv::Mat compose_overlay(
    const cv::Mat& source, const vision_models::HandFrame& hands,
    const HandInteractionFrame& interaction,
    const vision_models::FaceMeshFrame* face, const DemoMetrics& metrics);

} // namespace kfcore::hand_interaction::demo
