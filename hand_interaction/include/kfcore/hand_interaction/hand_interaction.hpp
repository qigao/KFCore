#pragma once

#include "kfcore/hand_interaction/primitive_extractor.hpp"

#include <memory>
#include <vector>

namespace kfcore::hand_interaction
{

inline constexpr char kHandInteractionSpecVersion[] = "kfcore-hand-interaction-v1";

struct HandInteractionSettings
{
    int         direction_dwell_ms                        = 0;
    int         direction_window_ms                       = 34;
    int         direction_minimum_supporting_observations = 1;
    float       direction_minimum_support_ratio           = 0.60F;
    float       direction_switch_margin                   = 0.15F;
    std::size_t direction_maximum_samples_per_source      = 1;
    int         grab_select_stable_ms                     = 167;
    int         grab_release_stable_ms                    = 167;
    int         grab_transition_max_ms                    = 2000;
    int         ok_dwell_ms                               = 800;
    int         observation_max_gap_ms                    = 350;
    int         neutral_rearm_ms                          = 0;
    int         wave_reversal_max_ms                      = 450;
    int         wave_total_max_ms                         = 1200;
    int         single_hand_v_dwell_ms                    = 500;
    int         dual_hand_dwell_ms                        = 200;
    int         dual_hand_onset_window_ms                 = 600;
    int         spatial_dwell_ms                          = 200;
    int         click_ready_dwell_ms                      = 100;
    int         click_press_dwell_ms                      = 67;
    int         click_release_dwell_ms                    = 100;
    int         click_transition_max_ms                   = 800;
    int         shape_window_ms                           = 700;
    int         shape_minimum_supporting_observations     = 2;
    float       shape_minimum_support_ratio               = 0.60F;
    float       shape_switch_margin                       = 0.15F;
    std::size_t shape_maximum_samples_per_source          = 32;
    int         region_window_ms                          = 350;
    int         region_minimum_supporting_observations    = 2;
    float       region_minimum_support_ratio              = 0.60F;
    float       region_switch_margin                      = 0.20F;
    std::size_t region_maximum_samples_per_source         = 16;
    int         history_ms                                = 5000;
    std::size_t max_observations_per_frame                = 128;
    std::size_t max_relation_events                       = 512;
    std::size_t max_observation_window_states             = 64;
    std::size_t max_action_states                         = 4096;
    bool        wave_require_horizontal_palm_axis         = false;
};

[[nodiscard]] thig::EngineSpec
build_hand_interaction_graph(const HandInteractionSettings& settings = {});

struct HandInteractionOptions
{
    HandPrimitiveOptions    primitives;
    HandInteractionSettings temporal;
};

struct HandInteractionFrame
{
    PrimitiveFrame                 primitives;
    std::vector<thig::ActionEvent> actions;
};

class HandInteractionPipeline final
{
public:
    explicit HandInteractionPipeline(const HandInteractionOptions& options = {});
    ~HandInteractionPipeline();

    HandInteractionPipeline(const HandInteractionPipeline&)            = delete;
    HandInteractionPipeline& operator=(const HandInteractionPipeline&) = delete;
    HandInteractionPipeline(HandInteractionPipeline&&) noexcept;
    HandInteractionPipeline& operator=(HandInteractionPipeline&&) noexcept;

    [[nodiscard]] HandInteractionFrame
         process(const vision_models::HandFrame& frame, const GestureFrameContext& context,
                 // External layout classification is intentionally limited to one
                 // Region relation per canonical hand and frame.
                 const std::vector<thig::Observation>& external_observations = {});
    void reset();

    [[nodiscard]] const thig::EngineSpec& graph_spec() const;
    [[nodiscard]] std::string             graph_state(const std::string& graph_id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::hand_interaction
