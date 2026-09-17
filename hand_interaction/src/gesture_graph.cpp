#include "kfcore/hand_interaction/hand_interaction.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace kfcore::hand_interaction
{
namespace
{

    constexpr char kOkPoseGroup[]            = "ok_pose";
    constexpr char kHandDirectionGroup[]     = "hand_direction";
    constexpr char kHandShapeGroup[]         = "hand_shape";
    constexpr char kIndexPressGroup[]        = "index_press";
    constexpr char kInteractionGraph[]       = "hand_interaction_cycle";
    constexpr char kArmedState[]             = "armed";
    constexpr char kAwaitReleaseState[]      = "await_release";
    constexpr char kDraggingState[]          = "dragging";
    constexpr char kAwaitNeutralState[]      = "await_neutral";
    constexpr char kBindingTimeoutRelation[] = "thig.binding.timeout";

    void validate_non_negative(int value, const char* name)
    {
        if (value < 0)
        {
            throw std::invalid_argument(std::string(name) + " cannot be negative");
        }
    }

    void validate_ratio(float value, const char* name)
    {
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
        {
            throw std::invalid_argument(std::string(name) + " must be finite within [0,1]");
        }
    }

    thig::PatternGraph BothRelations(std::string first, std::string second, int dwell_ms)
    {
        thig::PatternGraph pattern;
        pattern.nodes = {
            { "first", thig::PatternOperator::Atom, {}, std::move(first), dwell_ms },
            { "second", thig::PatternOperator::Atom, {}, std::move(second), dwell_ms },
            { "both", thig::PatternOperator::Both, { 0, 1 } },
        };
        pattern.root                             = 2U;
        pattern.nodes[pattern.root].minOverlapMs = dwell_ms;
        return pattern;
    }

    thig::PatternGraph DistinctStationaryShapes(std::string first_shape, std::string second_shape,
                                                int dwell_ms, int onset_window_ms)
    {
        thig::PatternGraph pattern;
        pattern.nodes = {
            { "first_shape", thig::PatternOperator::Atom, {}, std::move(first_shape), dwell_ms },
            { "first_stationary", thig::PatternOperator::Atom, {}, "Motion Stationary", dwell_ms },
            { "first_hand", thig::PatternOperator::Both, { 0, 1 } },
            { "second_shape", thig::PatternOperator::Atom, {}, std::move(second_shape), dwell_ms },
            { "second_stationary", thig::PatternOperator::Atom, {}, "Motion Stationary", dwell_ms },
            { "second_hand", thig::PatternOperator::Both, { 3, 4 } },
            { "both_hands", thig::PatternOperator::Both, { 2, 5 } },
        };
        pattern.root                             = 6U;
        pattern.nodes[2].minOverlapMs            = dwell_ms;
        pattern.nodes[5].minOverlapMs            = dwell_ms;
        pattern.nodes[pattern.root].windowMs     = onset_window_ms;
        pattern.nodes[pattern.root].sourceJoin   = thig::PatternSourceJoin::Distinct;
        pattern.nodes[pattern.root].minOverlapMs = dwell_ms;
        return pattern;
    }

    thig::PatternGraph ShapeTransitionWithStationarity(std::string first_shape,
                                                       std::string second_shape, int first_dwell_ms,
                                                       int second_dwell_ms, int transition_max_ms)
    {
        thig::PatternGraph pattern;
        pattern.nodes = {
            { "first_shape",
              thig::PatternOperator::Atom,
              {},
              std::move(first_shape),
              first_dwell_ms },
            { "second_shape",
              thig::PatternOperator::Atom,
              {},
              std::move(second_shape),
              second_dwell_ms },
            { "stationary", thig::PatternOperator::Atom, {}, "Motion Stationary", second_dwell_ms },
            { "terminal", thig::PatternOperator::Both, { 1, 2 } },
            { "confirmed", thig::PatternOperator::Seq, { 0, 3 }, {}, 0, transition_max_ms },
        };
        pattern.root                  = 4U;
        pattern.nodes[3].minOverlapMs = std::max(1, second_dwell_ms);
        pattern.forbiddenRelations    = { "Shape Pointer", "Shape V" };
        return pattern;
    }

    thig::StateTransitionSpec PatternTransition(std::string id, std::string from, std::string to,
                                                thig::PatternGraph     trigger,
                                                thig::SourceConstraint source_constraint,
                                                bool bind_source, bool unbind_source,
                                                std::string action = {}, int priority = 0)
    {
        thig::StateTransitionSpec transition;
        transition.id               = std::move(id);
        transition.fromState        = std::move(from);
        transition.toState          = std::move(to);
        transition.trigger          = std::move(trigger);
        transition.sourceConstraint = source_constraint;
        transition.bindSource       = bind_source;
        transition.unbindSource     = unbind_source;
        transition.action           = std::move(action);
        transition.priority         = priority;
        return transition;
    }

    thig::StateTransitionSpec Transition(std::string id, std::string from, std::string to,
                                         std::string relation, int dwell_ms,
                                         thig::SourceConstraint source_constraint, bool bind_source,
                                         bool unbind_source, std::string action = {},
                                         int priority = 0)
    {
        return PatternTransition(std::move(id), std::move(from), std::move(to),
                                 thig::PatternGraph::Atom(std::move(relation), dwell_ms),
                                 source_constraint, bind_source, unbind_source, std::move(action),
                                 priority);
    }

} // namespace

thig::EngineSpec build_hand_interaction_graph(const HandInteractionSettings& settings)
{
    validate_non_negative(settings.direction_dwell_ms, "direction dwell");
    validate_non_negative(settings.direction_window_ms, "direction window");
    validate_non_negative(settings.grab_select_stable_ms, "grab select stable duration");
    validate_non_negative(settings.grab_release_stable_ms, "grab release stable duration");
    validate_non_negative(settings.grab_transition_max_ms, "grab transition maximum");
    validate_non_negative(settings.ok_dwell_ms, "OK dwell");
    validate_non_negative(settings.observation_max_gap_ms, "observation max gap");
    validate_non_negative(settings.neutral_rearm_ms, "neutral rearm duration");
    validate_non_negative(settings.single_hand_v_dwell_ms, "single-hand V dwell");
    validate_non_negative(settings.dual_hand_dwell_ms, "dual-hand dwell");
    validate_non_negative(settings.dual_hand_onset_window_ms, "dual-hand onset window");
    validate_non_negative(settings.spatial_dwell_ms, "spatial dwell");
    validate_non_negative(settings.shape_window_ms, "shape window");
    validate_non_negative(settings.history_ms, "history");
    validate_non_negative(settings.rotation_cooldown_ms, "rotation cooldown");
    validate_ratio(settings.direction_minimum_support_ratio, "direction minimum support ratio");
    validate_ratio(settings.direction_switch_margin, "direction switch margin");
    validate_ratio(settings.shape_minimum_support_ratio, "shape minimum support ratio");
    validate_ratio(settings.shape_switch_margin, "shape switch margin");
    if (settings.direction_minimum_supporting_observations <= 0 ||
        settings.direction_maximum_samples_per_source == 0U ||
        settings.shape_minimum_supporting_observations <= 0 ||
        settings.shape_maximum_samples_per_source == 0U ||
        settings.max_observations_per_frame == 0U || settings.max_relation_events == 0U ||
        settings.max_observation_window_states == 0U || settings.max_action_states == 0U)
    {
        throw std::invalid_argument("hand interaction capacities must be positive");
    }
    const auto doubled_neutral = static_cast<std::int64_t>(settings.neutral_rearm_ms) * 2;
    if (doubled_neutral > (std::numeric_limits<int>::max)())
    {
        throw std::invalid_argument("neutral rearm duration exceeds the THIG time range");
    }
    const int source_binding_timeout_ms =
        std::max(settings.neutral_rearm_ms, settings.observation_max_gap_ms);

    thig::EngineSpec spec;
    spec.version                    = kHandInteractionSpecVersion;
    spec.historyMs                  = std::max({ settings.history_ms, settings.rotation_cooldown_ms,
                                                 settings.dual_hand_onset_window_ms, settings.shape_window_ms,
                                                 settings.direction_window_ms, settings.grab_transition_max_ms,
                                                 static_cast<int>(doubled_neutral) });
    spec.observationMaxGapMs        = settings.observation_max_gap_ms;
    spec.maxObservationsPerFrame    = settings.max_observations_per_frame;
    spec.maxRelationEvents          = settings.max_relation_events;
    spec.maxObservationWindowStates = settings.max_observation_window_states;
    spec.maxActionStates            = settings.max_action_states;

    spec.relations = {
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
        { kBindingTimeoutRelation, {} },
    };

    thig::ObservationWindowSpec shape_window;
    shape_window.exclusiveGroup                = kHandShapeGroup;
    shape_window.windowMs                      = settings.shape_window_ms;
    shape_window.minimumSupportingObservations = settings.shape_minimum_supporting_observations;
    shape_window.minimumSupportRatio           = settings.shape_minimum_support_ratio;
    shape_window.switchMargin                  = settings.shape_switch_margin;
    shape_window.maxSamplesPerSource           = settings.shape_maximum_samples_per_source;
    shape_window.ignoredRelations              = { "Shape Unclassified" };
    spec.observationWindows.push_back(std::move(shape_window));

    thig::ObservationWindowSpec direction_window;
    direction_window.exclusiveGroup = kHandDirectionGroup;
    direction_window.windowMs       = settings.direction_window_ms;
    direction_window.minimumSupportingObservations =
        settings.direction_minimum_supporting_observations;
    direction_window.minimumSupportRatio    = settings.direction_minimum_support_ratio;
    direction_window.switchMargin           = settings.direction_switch_margin;
    direction_window.maxSamplesPerSource    = settings.direction_maximum_samples_per_source;
    direction_window.rejectStableOnConflict = true;
    spec.observationWindows.push_back(std::move(direction_window));

    thig::ActionSpec ok;
    ok.action   = "OK";
    ok.pattern  = BothRelations("Pose OK", "Motion Stationary", settings.ok_dwell_ms);
    ok.priority = 100;
    spec.actions.push_back(std::move(ok));

    thig::ActionSpec single_hand_v;
    single_hand_v.action = "Single Hand V";
    single_hand_v.pattern =
        BothRelations("Shape V", "Motion Stationary", settings.single_hand_v_dwell_ms);
    single_hand_v.priority = 75;
    spec.actions.push_back(std::move(single_hand_v));

    const auto add_spatial_action = [&](std::string action, thig::PatternGraph pattern,
                                        std::string exclusive_group, int cooldown_ms)
    {
        thig::ActionSpec item;
        item.action         = std::move(action);
        item.pattern        = std::move(pattern);
        item.exclusiveGroup = std::move(exclusive_group);
        item.priority       = 70;
        item.cooldownMs     = cooldown_ms;
        spec.actions.push_back(std::move(item));
    };

    add_spatial_action(
        "Zoom In", thig::PatternGraph::Atom("Hands Distance Expanding", settings.spatial_dwell_ms),
        "zoom_transform", 0);
    add_spatial_action(
        "Zoom Out",
        thig::PatternGraph::Atom("Hands Distance Contracting", settings.spatial_dwell_ms),
        "zoom_transform", 0);
    add_spatial_action(
        "Rotate Clockwise",
        BothRelations("Rotation Clockwise", "Motion Stationary", settings.spatial_dwell_ms),
        "rotation_transform", settings.rotation_cooldown_ms);
    add_spatial_action(
        "Rotate CounterClockwise",
        BothRelations("Rotation CounterClockwise", "Motion Stationary", settings.spatial_dwell_ms),
        "rotation_transform", settings.rotation_cooldown_ms);

    thig::ActionSpec dual_hand_v;
    dual_hand_v.action  = "Two Hand V";
    dual_hand_v.pattern = DistinctStationaryShapes(
        "Shape V", "Shape V", settings.dual_hand_dwell_ms, settings.dual_hand_onset_window_ms);
    dual_hand_v.exclusiveGroup = "dual_hand_shape";
    dual_hand_v.priority       = 80;
    spec.actions.push_back(std::move(dual_hand_v));

    thig::ActionSpec v_fist;
    v_fist.action  = "V Fist";
    v_fist.pattern = DistinctStationaryShapes("Shape V", "Shape Fist", settings.dual_hand_dwell_ms,
                                              settings.dual_hand_onset_window_ms);
    v_fist.exclusiveGroup = "dual_hand_shape";
    v_fist.priority       = 80;
    spec.actions.push_back(std::move(v_fist));

    const auto add_swipe_action = [&](std::string action, std::string direction)
    {
        thig::ActionSpec swipe;
        swipe.action = std::move(action);
        swipe.pattern =
            BothRelations(std::move(direction), "Shape Open", settings.direction_dwell_ms);
        swipe.exclusiveGroup = "swipe_direction";
        swipe.priority       = 85;
        spec.actions.push_back(std::move(swipe));
    };
    add_swipe_action("Swipe Left", "Direction Left");
    add_swipe_action("Swipe Right", "Direction Right");

    const auto           bound_if_present = thig::SourceConstraint::BoundIfPresent;
    const auto           bound            = thig::SourceConstraint::Bound;
    thig::StateGraphSpec interaction;
    interaction.id           = kInteractionGraph;
    interaction.initialState = kArmedState;
    interaction.states = { kArmedState, kAwaitReleaseState, kDraggingState, kAwaitNeutralState };
    interaction.bindingTimeoutMs       = source_binding_timeout_ms;
    interaction.bindingTimeoutRelation = kBindingTimeoutRelation;
    interaction.transitions            = {
        PatternTransition("grasp_from_primitives", kArmedState, kAwaitReleaseState,
                                     ShapeTransitionWithStationarity(
                              "Shape Open", "Shape Fist", settings.grab_release_stable_ms,
                              settings.grab_select_stable_ms, settings.grab_transition_max_ms),
                                     bound_if_present, true, false, "Grasp", 95),
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
        PatternTransition("drag_end", kDraggingState, kAwaitNeutralState,
                                     ShapeTransitionWithStationarity(
                              "Shape Fist", "Shape Open", settings.grab_select_stable_ms,
                              settings.grab_release_stable_ms, settings.grab_transition_max_ms),
                                     bound, false, true, "Drag End", 120),
        Transition("timeout_dragging", kDraggingState, kAwaitNeutralState, kBindingTimeoutRelation,
                              0, bound, false, true, "Drag Cancelled", 110),
        Transition("rearm", kAwaitNeutralState, kArmedState, "Direction Neutral",
                              settings.neutral_rearm_ms, bound_if_present, false, true, {}, 100),
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

    return spec;
}

} // namespace kfcore::hand_interaction
