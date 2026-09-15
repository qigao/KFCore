#include "kfcore/hand_interaction/hand_interaction.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace kfcore::hand_interaction
{
namespace
{

constexpr char kOkPoseGroup[] = "ok_pose";
constexpr char kHandDirectionGroup[] = "hand_direction";
constexpr char kHandShapeGroup[] = "hand_shape";
constexpr char kIndexPressGroup[] = "index_press";

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
        {"first", thig::PatternOperator::Atom, {}, std::move(first), dwell_ms},
        {"second", thig::PatternOperator::Atom, {}, std::move(second), dwell_ms},
        {"both", thig::PatternOperator::Both, {0, 1}},
    };
    pattern.root = 2U;
    pattern.nodes[pattern.root].minOverlapMs = dwell_ms;
    return pattern;
}

thig::PatternGraph DistinctSourceBothRelations(std::string first, std::string second,
                                               int dwell_ms, int onset_window_ms)
{
    thig::PatternGraph pattern = BothRelations(std::move(first), std::move(second), dwell_ms);
    auto& root = pattern.nodes[pattern.root];
    root.windowMs = onset_window_ms;
    root.sourceJoin = thig::PatternSourceJoin::Distinct;
    return pattern;
}

} // namespace

thig::EngineSpec build_hand_interaction_graph(const HandInteractionSettings& settings)
{
    validate_non_negative(settings.ok_dwell_ms, "OK dwell");
    validate_non_negative(settings.observation_max_gap_ms, "observation max gap");
    validate_non_negative(settings.single_hand_v_dwell_ms, "single-hand V dwell");
    validate_non_negative(settings.dual_hand_dwell_ms, "dual-hand dwell");
    validate_non_negative(settings.dual_hand_onset_window_ms, "dual-hand onset window");
    validate_non_negative(settings.spatial_dwell_ms, "spatial dwell");
    validate_non_negative(settings.shape_window_ms, "shape window");
    validate_non_negative(settings.history_ms, "history");
    validate_non_negative(settings.rotation_cooldown_ms, "rotation cooldown");
    validate_ratio(settings.shape_minimum_support_ratio, "shape minimum support ratio");
    validate_ratio(settings.shape_switch_margin, "shape switch margin");
    if (settings.shape_minimum_supporting_observations <= 0 ||
        settings.shape_maximum_samples_per_source == 0U ||
        settings.max_observations_per_frame == 0U ||
        settings.max_relation_events == 0U ||
        settings.max_observation_window_states == 0U ||
        settings.max_action_states == 0U)
    {
        throw std::invalid_argument("hand interaction capacities must be positive");
    }

    thig::EngineSpec spec;
    spec.version = kHandInteractionSpecVersion;
    spec.historyMs = std::max({settings.history_ms,
                               settings.rotation_cooldown_ms,
                               settings.dual_hand_onset_window_ms,
                               settings.shape_window_ms});
    spec.observationMaxGapMs = settings.observation_max_gap_ms;
    spec.maxObservationsPerFrame = settings.max_observations_per_frame;
    spec.maxRelationEvents = settings.max_relation_events;
    spec.maxObservationWindowStates = settings.max_observation_window_states;
    spec.maxActionStates = settings.max_action_states;

    // The graph now owns only static/spatial semantic actions. Dynamic physical
    // gestures (wave/swipe/grab/release/click) are learned by gesture.temporal-gru.
    spec.relations = {
        {"Pose OK", kOkPoseGroup},
        {"Pose Not OK", kOkPoseGroup},
        {"Direction Right", kHandDirectionGroup},
        {"Direction Left", kHandDirectionGroup},
        {"Direction Up", kHandDirectionGroup},
        {"Direction Down", kHandDirectionGroup},
        {"Direction Neutral", kHandDirectionGroup},
        {"Direction Unclassified", kHandDirectionGroup},
        {"Shape Open", kHandShapeGroup},
        {"Shape Fist", kHandShapeGroup},
        {"Shape Pointer", kHandShapeGroup},
        {"Shape V", kHandShapeGroup},
        {"Shape Unclassified", kHandShapeGroup},
        {"Index Extended", kIndexPressGroup},
        {"Index Pressed", kIndexPressGroup},
        {"Index Intermediate", kIndexPressGroup},
        {"Index Unclassified", kIndexPressGroup},
        {"Motion Stationary", "motion_stationarity"},
        {"Motion Unstable", "motion_stationarity"},
        {"Scale Increasing", "palm_scale"},
        {"Scale Decreasing", "palm_scale"},
        {"Scale Stable", "palm_scale"},
        {"Scale Unclassified", "palm_scale"},
        {"Rotation Clockwise", "palm_rotation"},
        {"Rotation CounterClockwise", "palm_rotation"},
        {"Rotation Stable", "palm_rotation"},
        {"Rotation Unclassified", "palm_rotation"},
        {"Palm Axis Horizontal", "palm_axis_orientation"},
        {"Palm Axis Diagonal", "palm_axis_orientation"},
        {"Palm Axis Vertical", "palm_axis_orientation"},
        {"Hands Distance Expanding", "hands_distance"},
        {"Hands Distance Contracting", "hands_distance"},
        {"Hands Distance Stable", "hands_distance"},
        {"Hands Distance Unclassified", "hands_distance"},
    };

    thig::ObservationWindowSpec shape_window;
    shape_window.exclusiveGroup = kHandShapeGroup;
    shape_window.windowMs = settings.shape_window_ms;
    shape_window.minimumSupportingObservations =
        settings.shape_minimum_supporting_observations;
    shape_window.minimumSupportRatio = settings.shape_minimum_support_ratio;
    shape_window.switchMargin = settings.shape_switch_margin;
    shape_window.maxSamplesPerSource = settings.shape_maximum_samples_per_source;
    shape_window.ignoredRelations = {"Shape Unclassified"};
    spec.observationWindows.push_back(std::move(shape_window));

    thig::ActionSpec ok;
    ok.action = "OK";
    ok.pattern = BothRelations("Pose OK", "Motion Stationary", settings.ok_dwell_ms);
    ok.priority = 100;
    spec.actions.push_back(std::move(ok));

    thig::ActionSpec single_hand_v;
    single_hand_v.action = "Single Hand V";
    single_hand_v.pattern =
        BothRelations("Shape V", "Motion Stationary", settings.single_hand_v_dwell_ms);
    single_hand_v.priority = 75;
    spec.actions.push_back(std::move(single_hand_v));

    const auto add_spatial_action =
        [&](std::string action, thig::PatternGraph pattern,
            std::string exclusive_group, int cooldown_ms)
    {
        thig::ActionSpec item;
        item.action = std::move(action);
        item.pattern = std::move(pattern);
        item.exclusiveGroup = std::move(exclusive_group);
        item.priority = 70;
        item.cooldownMs = cooldown_ms;
        spec.actions.push_back(std::move(item));
    };

    add_spatial_action(
        "Zoom In",
        thig::PatternGraph::Atom("Hands Distance Expanding", settings.spatial_dwell_ms),
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
    dual_hand_v.action = "Two Hand V";
    dual_hand_v.pattern = DistinctSourceBothRelations(
        "Shape V", "Shape V", settings.dual_hand_dwell_ms,
        settings.dual_hand_onset_window_ms);
    dual_hand_v.exclusiveGroup = "dual_hand_shape";
    dual_hand_v.priority = 80;
    spec.actions.push_back(std::move(dual_hand_v));

    return spec;
}

} // namespace kfcore::hand_interaction
