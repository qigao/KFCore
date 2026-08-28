#include "kfcore/hand_interaction/hand_interaction.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kfcore::hand_interaction
{
namespace
{

    constexpr char kOkPoseGroup[]                 = "ok_pose";
    constexpr char kHandDirectionGroup[]          = "hand_direction";
    constexpr char kHandShapeGroup[]              = "hand_shape";
    constexpr char kHandRegionGroup[]             = "hand_region";
    constexpr char kIndexPressGroup[]             = "index_press";
    constexpr char kHandInteractionGraph[]        = "hand_interaction_cycle";
    constexpr char kWaveGraph[]                   = "wave_cycle";
    constexpr char kScreenClickGraph[]            = "screen_click_cycle";
    constexpr char kArmedState[]                  = "armed";
    constexpr char kAwaitReleaseState[]           = "await_release";
    constexpr char kDraggingState[]               = "dragging";
    constexpr char kAwaitNeutralState[]           = "await_neutral";
    constexpr char kBindingTimeoutRelation[]      = "thig.binding.timeout";
    constexpr char kClickBindingTimeoutRelation[] = "thig.screen_click.binding.timeout";
    constexpr char kWaveBindingTimeoutRelation[]  = "thig.wave.binding.timeout";

    thig::StateTransitionSpec Transition(std::string id, std::string from, std::string to,
                                         std::string relation, int dwellMs,
                                         thig::SourceConstraint sourceConstraint, bool bindSource,
                                         bool unbindSource, std::string action = {},
                                         int priority = 0)
    {
        thig::StateTransitionSpec transition;
        transition.id               = std::move(id);
        transition.fromState        = std::move(from);
        transition.toState          = std::move(to);
        transition.trigger          = thig::PatternGraph::Atom(std::move(relation), dwellMs);
        transition.sourceConstraint = sourceConstraint;
        transition.bindSource       = bindSource;
        transition.unbindSource     = unbindSource;
        transition.action           = std::move(action);
        transition.priority         = priority;
        return transition;
    }

    thig::PatternGraph BothRelations(std::string first, std::string second, int dwellMs)
    {
        thig::PatternGraph pattern;
        pattern.nodes = {
            { "first", thig::PatternOperator::Atom, {}, std::move(first), dwellMs },
            { "second", thig::PatternOperator::Atom, {}, std::move(second), dwellMs },
            { "both", thig::PatternOperator::Both, { 0, 1 } },
        };
        pattern.root                             = 2;
        pattern.nodes[pattern.root].minOverlapMs = dwellMs;
        return pattern;
    }

    thig::PatternGraph ThreeStrokeWavePattern(std::string firstDirection,
                                              std::string secondDirection, int strokeDwellMs,
                                              int reversalMaxMs, int totalMaxMs,
                                              bool requireHorizontalPalmAxis)
    {
        thig::PatternGraph pattern;
        pattern.nodes = {
            { "first_stroke", thig::PatternOperator::Atom, {}, firstDirection, strokeDwellMs },
            { "return_stroke", thig::PatternOperator::Atom, {}, secondDirection, strokeDwellMs },
            { "final_stroke", thig::PatternOperator::Atom, {}, firstDirection, strokeDwellMs },
            { "first_reversal", thig::PatternOperator::Seq, { 0, 1 }, {}, 0, reversalMaxMs },
            { "second_reversal", thig::PatternOperator::Seq, { 3, 2 }, {}, 0, reversalMaxMs },
            { "bounded_wave", thig::PatternOperator::Within, { 4, 4 }, {}, 0, totalMaxMs },
            { "open_palm", thig::PatternOperator::Atom, {}, "Shape Open", 0 },
            { "open_palm_wave", thig::PatternOperator::During, { 5, 6 } },
        };
        if (requireHorizontalPalmAxis)
        {
            pattern.nodes.push_back(
                { "horizontal_palm_axis", thig::PatternOperator::Atom, {},
                  "Palm Axis Horizontal", 0 });
            pattern.nodes.push_back(
                { "open_horizontal_palm", thig::PatternOperator::Both, { 6, 8 } });
            pattern.nodes[7].inputs = { 5, 9 };
        }
        pattern.root               = 7;
        pattern.forbiddenRelations = { "Direction Up", "Direction Down", "Shape Fist",
                                       "Shape Pointer", "Shape V" };
        return pattern;
    }

    thig::PatternGraph PointerClickPattern(std::string region, int readyDwellMs, int pressDwellMs,
                                           int transitionMaxMs)
    {
        thig::PatternGraph pattern;
        pattern.nodes = {
            { "ready", thig::PatternOperator::Atom, {}, "Index Extended", readyDwellMs },
            { "pressed", thig::PatternOperator::Atom, {}, "Index Pressed", pressDwellMs },
            { "stationary", thig::PatternOperator::Atom, {}, "Motion Stationary", pressDwellMs },
            { "press_terminal", thig::PatternOperator::Both, { 1, 2 } },
            { "region", thig::PatternOperator::Atom, {}, std::move(region), 0 },
            { "press_in_region", thig::PatternOperator::Both, { 3, 4 } },
            { "click", thig::PatternOperator::Seq, { 0, 5 }, {}, 0, transitionMaxMs },
        };
        pattern.root                  = 6;
        pattern.nodes[3].minOverlapMs = pressDwellMs;
        pattern.nodes[5].minOverlapMs = pressDwellMs;
        return pattern;
    }

    thig::PatternGraph DistinctSourceBothRelations(std::string first, std::string second,
                                                   int dwellMs, int onsetWindowMs)
    {
        thig::PatternGraph pattern = BothRelations(std::move(first), std::move(second), dwellMs);
        auto&              root    = pattern.nodes[pattern.root];
        root.windowMs              = onsetWindowMs;
        root.sourceJoin            = thig::PatternSourceJoin::Distinct;
        return pattern;
    }

    thig::PatternGraph ShapeTransitionWithStationarity(std::string firstShape,
                                                       std::string secondShape, int firstDwellMs,
                                                       int secondDwellMs, int transitionMaxMs)
    {
        thig::PatternGraph pattern;
        pattern.nodes = {
            { "first_shape", thig::PatternOperator::Atom, {}, std::move(firstShape), firstDwellMs },
            { "second_shape",
              thig::PatternOperator::Atom,
              {},
              std::move(secondShape),
              secondDwellMs },
            { "stationary", thig::PatternOperator::Atom, {}, "Motion Stationary", secondDwellMs },
            { "terminal", thig::PatternOperator::Both, { 1, 2 } },
            { "confirmed", thig::PatternOperator::Seq, { 0, 3 }, {}, 0, transitionMaxMs },
        };
        pattern.root                  = 4;
        pattern.nodes[3].minOverlapMs = std::max(1, secondDwellMs);
        pattern.forbiddenRelations    = { "Shape Pointer", "Shape V" };
        return pattern;
    }

    thig::StateTransitionSpec PatternTransition(std::string id, std::string from, std::string to,
                                                thig::PatternGraph     trigger,
                                                thig::SourceConstraint sourceConstraint,
                                                bool bindSource, bool unbindSource,
                                                std::string action = {}, int priority = 0)
    {
        thig::StateTransitionSpec transition;
        transition.id               = std::move(id);
        transition.fromState        = std::move(from);
        transition.toState          = std::move(to);
        transition.trigger          = std::move(trigger);
        transition.sourceConstraint = sourceConstraint;
        transition.bindSource       = bindSource;
        transition.unbindSource     = unbindSource;
        transition.action           = std::move(action);
        transition.priority         = priority;
        return transition;
    }

} // namespace

thig::EngineSpec build_hand_interaction_graph(const HandInteractionSettings& settings)
{
    if (settings.neutral_rearm_ms < 0)
    {
        throw std::invalid_argument("neutral rearm duration cannot be negative");
    }
    const auto doubled_neutral = static_cast<std::int64_t>(settings.neutral_rearm_ms) * 2;
    if (doubled_neutral > (std::numeric_limits<int>::max)())
    {
        throw std::invalid_argument("neutral rearm duration exceeds the THIG time range");
    }
    const int sourceBindingTimeoutMs =
        std::max(settings.neutral_rearm_ms, settings.observation_max_gap_ms);
    thig::EngineSpec spec;
    spec.version                 = kHandInteractionSpecVersion;
    spec.historyMs               = std::max(settings.history_ms, static_cast<int>(doubled_neutral));
    spec.maxObservationsPerFrame = settings.max_observations_per_frame;
    spec.maxRelationEvents       = settings.max_relation_events;
    spec.maxObservationWindowStates = settings.max_observation_window_states;
    spec.maxActionStates            = settings.max_action_states;
    spec.observationMaxGapMs        = settings.observation_max_gap_ms;
    spec.relations                  = {
        { "Pose OK", kOkPoseGroup },
        { "Pose Not OK", kOkPoseGroup },
        { "Direction Right", kHandDirectionGroup },
        { "Direction Left", kHandDirectionGroup },
        { "Direction Up", kHandDirectionGroup },
        { "Direction Down", kHandDirectionGroup },
        { "Direction Neutral", kHandDirectionGroup },
        { "Direction Unclassified", kHandDirectionGroup },
        { "Shape Open", kHandShapeGroup },
        { "Shape Fist", kHandShapeGroup },
        { "Shape Pointer", kHandShapeGroup },
        { "Shape V", kHandShapeGroup },
        { "Shape Unclassified", kHandShapeGroup },
        { "Index Extended", kIndexPressGroup },
        { "Index Pressed", kIndexPressGroup },
        { "Index Intermediate", kIndexPressGroup },
        { "Index Unclassified", kIndexPressGroup },
        { "Motion Stationary", "motion_stationarity" },
        { "Motion Unstable", "motion_stationarity" },
        { "Scale Increasing", "palm_scale" },
        { "Scale Decreasing", "palm_scale" },
        { "Scale Stable", "palm_scale" },
        { "Scale Unclassified", "palm_scale" },
        { "Rotation Clockwise", "palm_rotation" },
        { "Rotation CounterClockwise", "palm_rotation" },
        { "Rotation Stable", "palm_rotation" },
        { "Rotation Unclassified", "palm_rotation" },
        { "Palm Axis Horizontal", "palm_axis_orientation" },
        { "Palm Axis Diagonal", "palm_axis_orientation" },
        { "Palm Axis Vertical", "palm_axis_orientation" },
        { "Hands Distance Expanding", "hands_distance" },
        { "Hands Distance Contracting", "hands_distance" },
        { "Hands Distance Stable", "hands_distance" },
        { "Hands Distance Unclassified", "hands_distance" },
        { "Region Center", kHandRegionGroup },
        { "Region Left", kHandRegionGroup },
        { "Region Right", kHandRegionGroup },
        { "Region Top", kHandRegionGroup },
        { "Region Bottom", kHandRegionGroup },
        { "Region Unclassified", kHandRegionGroup },
        { kBindingTimeoutRelation, {} },
        { kWaveBindingTimeoutRelation, {} },
        { kClickBindingTimeoutRelation, {} },
    };
    thig::ObservationWindowSpec shapeWindow;
    shapeWindow.exclusiveGroup                = kHandShapeGroup;
    shapeWindow.windowMs                      = settings.shape_window_ms;
    shapeWindow.minimumSupportingObservations = settings.shape_minimum_supporting_observations;
    shapeWindow.minimumSupportRatio           = settings.shape_minimum_support_ratio;
    shapeWindow.switchMargin                  = settings.shape_switch_margin;
    shapeWindow.maxSamplesPerSource           = settings.shape_maximum_samples_per_source;
    shapeWindow.ignoredRelations              = { "Shape Unclassified" };
    spec.observationWindows.push_back(std::move(shapeWindow));

    thig::ObservationWindowSpec directionWindow;
    directionWindow.exclusiveGroup = kHandDirectionGroup;
    directionWindow.windowMs       = settings.direction_window_ms;
    directionWindow.minimumSupportingObservations =
        settings.direction_minimum_supporting_observations;
    directionWindow.minimumSupportRatio    = settings.direction_minimum_support_ratio;
    directionWindow.switchMargin           = settings.direction_switch_margin;
    directionWindow.maxSamplesPerSource    = settings.direction_maximum_samples_per_source;
    directionWindow.rejectStableOnConflict = true;
    spec.observationWindows.push_back(std::move(directionWindow));

    thig::ObservationWindowSpec regionWindow;
    regionWindow.exclusiveGroup                = kHandRegionGroup;
    regionWindow.windowMs                      = settings.region_window_ms;
    regionWindow.minimumSupportingObservations = settings.region_minimum_supporting_observations;
    regionWindow.minimumSupportRatio           = settings.region_minimum_support_ratio;
    regionWindow.switchMargin                  = settings.region_switch_margin;
    regionWindow.maxSamplesPerSource           = settings.region_maximum_samples_per_source;
    regionWindow.ignoredRelations              = { "Region Unclassified" };
    spec.observationWindows.push_back(std::move(regionWindow));

    thig::ActionSpec checkout;
    checkout.action   = "OK";
    checkout.pattern  = BothRelations("Pose OK", "Motion Stationary", settings.ok_dwell_ms);
    checkout.priority = 100;
    spec.actions.push_back(std::move(checkout));

    thig::ActionSpec singleHandV;
    singleHandV.action = "Single Hand V";
    singleHandV.pattern =
        BothRelations("Shape V", "Motion Stationary", settings.single_hand_v_dwell_ms);
    singleHandV.priority = 75;
    spec.actions.push_back(std::move(singleHandV));

    const auto addSpatialAction =
        [&](std::string action, thig::PatternGraph pattern, std::string exclusiveGroup,
            int cooldownMs)
    {
        thig::ActionSpec specAction;
        specAction.action         = std::move(action);
        specAction.pattern        = std::move(pattern);
        specAction.exclusiveGroup = std::move(exclusiveGroup);
        specAction.priority       = 70;
        specAction.cooldownMs     = cooldownMs;
        spec.actions.push_back(std::move(specAction));
    };
    addSpatialAction(
        "Zoom In", thig::PatternGraph::Atom("Hands Distance Expanding", settings.spatial_dwell_ms),
        "zoom_transform", 0);
    addSpatialAction(
        "Zoom Out",
        thig::PatternGraph::Atom("Hands Distance Contracting", settings.spatial_dwell_ms),
        "zoom_transform", 0);
    addSpatialAction(
        "Rotate Clockwise",
        BothRelations("Rotation Clockwise", "Motion Stationary", settings.spatial_dwell_ms),
        "rotation_transform", settings.rotation_cooldown_ms);
    addSpatialAction(
        "Rotate CounterClockwise",
        BothRelations("Rotation CounterClockwise", "Motion Stationary", settings.spatial_dwell_ms),
        "rotation_transform", settings.rotation_cooldown_ms);

    const auto addDualHandAction =
        [&](std::string action, std::string first, std::string second, int priority)
    {
        thig::ActionSpec pattern;
        pattern.action         = std::move(action);
        pattern.pattern        = DistinctSourceBothRelations(std::move(first), std::move(second),
                                                             settings.dual_hand_dwell_ms,
                                                             settings.dual_hand_onset_window_ms);
        pattern.exclusiveGroup = "dual_hand_shape";
        pattern.priority       = priority;
        spec.actions.push_back(std::move(pattern));
    };
    addDualHandAction("Two Hand V", "Shape V", "Shape V", 80);

    thig::StateGraphSpec interaction;
    interaction.id           = kHandInteractionGraph;
    interaction.initialState = kArmedState;
    interaction.states = { kArmedState, kAwaitReleaseState, kDraggingState, kAwaitNeutralState };
    interaction.bindOnRelations        = { "Direction Right", "Direction Left", "Shape Fist",
                                           "Shape Open" };
    interaction.bindingTimeoutMs       = sourceBindingTimeoutMs;
    interaction.bindingTimeoutRelation = kBindingTimeoutRelation;

    const auto boundIfPresent = thig::SourceConstraint::BoundIfPresent;
    const auto bound          = thig::SourceConstraint::Bound;
    interaction.transitions   = {
        PatternTransition("grasp_from_primitives", kArmedState, kAwaitReleaseState,
                            ShapeTransitionWithStationarity(
                              "Shape Open", "Shape Fist", settings.grab_release_stable_ms,
                              settings.grab_select_stable_ms, settings.grab_transition_max_ms),
                            boundIfPresent, true, false, "Grasp", 95),
        Transition("timeout_armed", kArmedState, kArmedState, kBindingTimeoutRelation, 0, bound,
                     false, true, {}, 110),

        PatternTransition("release_from_primitives", kAwaitReleaseState, kArmedState,
                            ShapeTransitionWithStationarity(
                              "Shape Fist", "Shape Open", settings.grab_select_stable_ms,
                              settings.grab_release_stable_ms, settings.grab_transition_max_ms),
                            bound, false, true, "Release", 120),
        PatternTransition(
            "drag_start_right", kAwaitReleaseState, kDraggingState,
            BothRelations("Direction Right", "Shape Fist", settings.direction_dwell_ms), bound,
            false, false, "Drag Start Right", 100),
        PatternTransition(
            "drag_start_left", kAwaitReleaseState, kDraggingState,
            BothRelations("Direction Left", "Shape Fist", settings.direction_dwell_ms), bound,
            false, false, "Drag Start Left", 100),
        PatternTransition("drag_start_up", kAwaitReleaseState, kDraggingState,
                            BothRelations("Direction Up", "Shape Fist", settings.direction_dwell_ms),
                            bound, false, false, "Drag Start Up", 100),
        PatternTransition(
            "drag_start_down", kAwaitReleaseState, kDraggingState,
            BothRelations("Direction Down", "Shape Fist", settings.direction_dwell_ms), bound,
            false, false, "Drag Start Down", 100),
        Transition("timeout_wait_release", kAwaitReleaseState, kAwaitNeutralState,
                     kBindingTimeoutRelation, 0, bound, false, true, {}, 110),

        PatternTransition(
            "drag_right", kDraggingState, kDraggingState,
            BothRelations("Direction Right", "Shape Fist", settings.direction_dwell_ms), bound,
            false, false, "Drag Right", 90),
        PatternTransition(
            "drag_left", kDraggingState, kDraggingState,
            BothRelations("Direction Left", "Shape Fist", settings.direction_dwell_ms), bound,
            false, false, "Drag Left", 90),
        PatternTransition("drag_up", kDraggingState, kDraggingState,
                            BothRelations("Direction Up", "Shape Fist", settings.direction_dwell_ms),
                            bound, false, false, "Drag Up", 90),
        PatternTransition(
            "drag_down", kDraggingState, kDraggingState,
            BothRelations("Direction Down", "Shape Fist", settings.direction_dwell_ms), bound,
            false, false, "Drag Down", 90),
        PatternTransition("drag_end", kDraggingState, kAwaitNeutralState,
                            ShapeTransitionWithStationarity(
                              "Shape Fist", "Shape Open", settings.grab_select_stable_ms,
                              settings.grab_release_stable_ms, settings.grab_transition_max_ms),
                            bound, false, true, "Drag End", 120),
        Transition("timeout_dragging", kDraggingState, kAwaitNeutralState, kBindingTimeoutRelation,
                     0, bound, false, true, "Drag Cancelled", 110),

        Transition("rearm", kAwaitNeutralState, kArmedState, "Direction Neutral",
                     settings.neutral_rearm_ms, boundIfPresent, false, true, {}, 100),
        Transition("timeout_wait_neutral", kAwaitNeutralState, kArmedState, kBindingTimeoutRelation,
                     0, bound, false, true, {}, 110),
    };
    for (auto& transition : interaction.transitions)
    {
        if (transition.id == "rearm")
        {
            transition.minStateDurationMs = settings.neutral_rearm_ms;
            break;
        }
    }
    spec.stateGraphs.push_back(std::move(interaction));

    thig::StateGraphSpec wave;
    wave.id                     = kWaveGraph;
    wave.initialState           = kArmedState;
    wave.states                 = { kArmedState, kAwaitNeutralState };
    wave.bindOnRelations        = { "Direction Left", "Direction Right" };
    wave.bindingTimeoutMs       = sourceBindingTimeoutMs;
    wave.bindingTimeoutRelation = kWaveBindingTimeoutRelation;
    wave.transitions            = {
        PatternTransition(
            "wave_left_right_left", kArmedState, kAwaitNeutralState,
            ThreeStrokeWavePattern("Direction Left", "Direction Right", settings.direction_dwell_ms,
                                   settings.wave_reversal_max_ms, settings.wave_total_max_ms,
                                   settings.wave_require_horizontal_palm_axis),
            boundIfPresent, true, false, "Wave", 100),
        PatternTransition(
            "wave_right_left_right", kArmedState, kAwaitNeutralState,
            ThreeStrokeWavePattern("Direction Right", "Direction Left", settings.direction_dwell_ms,
                                   settings.wave_reversal_max_ms, settings.wave_total_max_ms,
                                   settings.wave_require_horizontal_palm_axis),
            boundIfPresent, true, false, "Wave", 100),
        Transition("wave_rearm", kAwaitNeutralState, kArmedState, "Direction Neutral",
                              settings.neutral_rearm_ms, bound, false, true, {}, 100),
        Transition("wave_timeout_armed", kArmedState, kArmedState, kWaveBindingTimeoutRelation, 0,
                              bound, false, true, {}, 110),
        Transition("wave_timeout_wait_neutral", kAwaitNeutralState, kArmedState,
                              kWaveBindingTimeoutRelation, 0, bound, false, true, {}, 110),
    };
    for (auto& transition : wave.transitions)
    {
        if (transition.action == "Wave")
        {
            transition.stateLocalRelations = { "Direction Left", "Direction Right" };
        }
    }
    spec.stateGraphs.push_back(std::move(wave));

    thig::StateGraphSpec click;
    click.id                     = kScreenClickGraph;
    click.initialState           = "ready";
    click.states                 = { "ready", "pressed" };
    click.bindOnRelations        = { "Index Extended" };
    click.bindingTimeoutMs       = sourceBindingTimeoutMs;
    click.bindingTimeoutRelation = kClickBindingTimeoutRelation;
    for (const char* region : { "Center", "Left", "Right", "Top", "Bottom" })
    {
        std::string regionName(region);
        click.transitions.push_back(PatternTransition(
            "press_" + regionName, "ready", "pressed",
            PointerClickPattern("Region " + regionName, settings.click_ready_dwell_ms,
                                settings.click_press_dwell_ms, settings.click_transition_max_ms),
            boundIfPresent, true, false, "Click " + regionName, 100));
    }
    click.transitions.push_back(Transition("release", "pressed", "ready", "Index Extended",
                                           settings.click_release_dwell_ms, bound, false, true, {},
                                           100));
    click.transitions.push_back(Transition("timeout_ready", "ready", "ready",
                                           kClickBindingTimeoutRelation, 0, bound, false, true, {},
                                           110));
    click.transitions.push_back(Transition("timeout_pressed", "pressed", "ready",
                                           kClickBindingTimeoutRelation, 0, bound, false, true, {},
                                           110));
    spec.stateGraphs.push_back(std::move(click));
    return spec;
}

} // namespace kfcore::hand_interaction
