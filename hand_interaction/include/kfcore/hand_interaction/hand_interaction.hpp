#pragma once

#include "kfcore/hand_gesture/types.hpp"
#include "kfcore/hand_interaction/primitive_extractor.hpp"

#include <memory>
#include <vector>

namespace kfcore::hand_interaction
{

inline constexpr char kHandInteractionSpecVersion[] = "kfcore-hand-interaction-v2";

struct HandInteractionSettings
{
    int         ok_dwell_ms                               = 800;
    int         observation_max_gap_ms                    = 350;
    int         single_hand_v_dwell_ms                    = 500;
    int         dual_hand_dwell_ms                        = 200;
    int         dual_hand_onset_window_ms                 = 600;
    int         spatial_dwell_ms                          = 200;
    int         shape_window_ms                           = 700;
    int         shape_minimum_supporting_observations     = 2;
    float       shape_minimum_support_ratio               = 0.60F;
    float       shape_switch_margin                       = 0.15F;
    std::size_t shape_maximum_samples_per_source          = 32;
    int         history_ms                                = 5000;
    std::size_t max_observations_per_frame                = 128;
    std::size_t max_relation_events                       = 512;
    std::size_t max_observation_window_states             = 64;
    std::size_t max_action_states                         = 4096;
    int         rotation_cooldown_ms                      = 600;
};

[[nodiscard]] thig::EngineSpec
build_hand_interaction_graph(const HandInteractionSettings& settings = {});

struct HandInteractionOptions
{
    HandPrimitiveOptions    primitives;
    HandInteractionSettings semantic;
};

struct HandInteractionFrame
{
    PrimitiveFrame                  primitives;
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

    [[nodiscard]] HandInteractionFrame process(
        const hand_models::HandFrame& frame,
        const GestureFrameContext& context,
        const std::vector<hand_gesture::GestureEvent>& gestures,
        const std::vector<thig::Observation>& external_observations = {});

    void reset();

    [[nodiscard]] const thig::EngineSpec& graph_spec() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::hand_interaction
