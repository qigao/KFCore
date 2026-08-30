#pragma once

#include "hand_interaction_demo_capture.hpp"

#include "kfcore/face_models/core.hpp"
#include "kfcore/hand_interaction/hand_interaction.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
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
    hand_models::StageTimings model;
    face_models::FaceMeshTimings face;
    double                             fps        = 0.0;
    double                             convert_ms = 0.0;
    double                             thig_ms  = 0.0;
    double                             frame_ms = 0.0;
};

struct HandOverlayText
{
    std::string identity;
    std::string derived;
    std::string dynamics;
};

struct DemoThigStatus
{
    std::vector<thig::ActionEvent> recent_actions;
    std::string                    hand_state;
    std::string                    wave_state;
    std::string                    click_state;
};

struct HandIdentityDiagnostic
{
    std::size_t             input_index = 0U;
    int                     raw_track_id = -1;
    int                     canonical_id = 0;
    HandIdentityAssociation association =
        HandIdentityAssociation::UnreliableObservation;
    hand_models::Handedness handedness = hand_models::Handedness::Unknown;
    std::uint8_t appearance_parts = 0U;

    [[nodiscard]] bool operator==(const HandIdentityDiagnostic& other) const noexcept;
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
    const CanonicalHand& identity, const hand_models::HandResult& hand,
    const PrimitiveFrame& primitives);

[[nodiscard]] std::string format_thig_state_line(const DemoThigStatus& status);
[[nodiscard]] std::string format_action_line(const thig::ActionEvent& action);

[[nodiscard]] std::vector<HandIdentityDiagnostic> make_identity_diagnostics(
    const hand_models::HandFrame& hands,
    const HandInteractionFrame& interaction);

[[nodiscard]] std::string format_identity_diagnostic_line(
    std::uint64_t frame_serial, const hand_models::HandFrame& hands,
    const HandInteractionFrame& interaction);

// Source must be a non-empty CV_8UC3 image. The returned overlay is mirrored,
// owns its pixels, and source is never modified.
[[nodiscard]] cv::Mat compose_overlay(
    const cv::Mat& source, const hand_models::HandFrame& hands,
    const HandInteractionFrame& interaction,
    const DemoThigStatus& status,
    const face_models::FaceMeshFrame* face, const DemoMetrics& metrics);

} // namespace kfcore::hand_interaction::demo
