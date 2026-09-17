#include "kfcore/hand_interaction/hand_interaction.hpp"
#include "tinytest.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using kfcore::hand_interaction::GestureFrameContext;
using kfcore::hand_interaction::HandInteractionOptions;
using kfcore::hand_interaction::HandInteractionPipeline;
using kfcore::hand_interaction::HandInteractionSettings;
using kfcore::hand_models::Gesture;
using kfcore::hand_models::HandFrame;
using kfcore::hand_models::HandResult;
using kfcore::thig::ActionEvent;
using kfcore::thig::Observation;
using kfcore::thig::TemporalGraphEngine;

HandInteractionSettings immediate_settings()
{
    HandInteractionSettings settings;
    settings.direction_dwell_ms                        = 0;
    settings.direction_window_ms                       = 1;
    settings.direction_minimum_supporting_observations = 1;
    settings.direction_minimum_support_ratio           = 1.0F;
    settings.direction_switch_margin                   = 0.0F;
    settings.direction_maximum_samples_per_source      = 1;
    settings.grab_select_stable_ms                     = 0;
    settings.grab_release_stable_ms                    = 0;
    settings.grab_transition_max_ms                    = 500;
    settings.ok_dwell_ms                               = 0;
    settings.observation_max_gap_ms                    = 350;
    settings.neutral_rearm_ms                          = 0;
    settings.single_hand_v_dwell_ms                    = 0;
    settings.dual_hand_dwell_ms                        = 0;
    settings.dual_hand_onset_window_ms                 = 0;
    settings.spatial_dwell_ms                          = 0;
    settings.shape_window_ms                           = 1;
    settings.shape_minimum_supporting_observations     = 1;
    settings.shape_minimum_support_ratio               = 1.0F;
    settings.shape_switch_margin                       = 0.0F;
    settings.shape_maximum_samples_per_source          = 1;
    settings.history_ms                                = 1000;
    settings.rotation_cooldown_ms                      = 0;
    return settings;
}

HandInteractionSettings shared_default_direction_settings()
{
    HandInteractionSettings settings = immediate_settings();
    HandInteractionSettings defaults;
    settings.direction_dwell_ms                        = defaults.direction_dwell_ms;
    settings.direction_window_ms                       = defaults.direction_window_ms;
    settings.direction_minimum_supporting_observations =
        defaults.direction_minimum_supporting_observations;
    settings.direction_minimum_support_ratio      = defaults.direction_minimum_support_ratio;
    settings.direction_switch_margin              = defaults.direction_switch_margin;
    settings.direction_maximum_samples_per_source =
        defaults.direction_maximum_samples_per_source;
    settings.neutral_rearm_ms = defaults.neutral_rearm_ms;
    return settings;
}

bool has_action(const std::vector<ActionEvent>& actions, const std::string& name)
{
    return std::any_of(actions.begin(), actions.end(),
                       [&](const ActionEvent& action) { return action.action == name; });
}

bool has_action_from(const std::vector<ActionEvent>& actions, const std::string& name,
                     int source_id)
{
    return std::any_of(actions.begin(), actions.end(), [&](const ActionEvent& action)
                       { return action.action == name && action.source.id == source_id; });
}

Observation relation(std::string name, int source_id, std::uint64_t serial, int target_id = 0)
{
    Observation value;
    value.serial = serial;
    value.source = { "hand", source_id };
    if (target_id > 0)
    {
        value.target = kfcore::thig::EntityRef { "hand", target_id };
    }
    value.relation   = std::move(name);
    value.confidence = 0.95F;
    value.producer   = "test";
    return value;
}

std::vector<ActionEvent> feed(TemporalGraphEngine& engine, std::uint64_t serial, int elapsed_ms,
                              std::initializer_list<Observation> observations)
{
    std::vector<Observation> frame(observations);
    for (Observation& item : frame)
    {
        item.serial = serial;
    }
    return engine.ProcessFrame(frame, std::chrono::steady_clock::time_point {} +
                                          std::chrono::milliseconds(elapsed_ms));
}

int first_default_swipe_ms(std::initializer_list<int> elapsed_times_ms)
{
    TemporalGraphEngine engine(kfcore::hand_interaction::build_hand_interaction_graph());
    std::uint64_t       serial = 0;
    for (const int elapsed_ms : elapsed_times_ms)
    {
        ++serial;
        if (has_action(feed(engine, serial, elapsed_ms,
                            { relation("Shape Open", 1, serial),
                              relation("Direction Left", 1, serial) }),
                       "Swipe Left"))
        {
            return elapsed_ms;
        }
    }
    return -1;
}

bool has_active_relation(const TemporalGraphEngine& engine, const std::string& name)
{
    return std::any_of(engine.Relations().begin(), engine.Relations().end(),
                       [&](const auto& item) { return item.active && item.relation == name; });
}

HandResult model_hand(Gesture gesture = Gesture::Open)
{
    HandResult hand;
    hand.track_id                  = 7;
    hand.gesture                   = gesture;
    hand.palm.confidence           = 0.98F;
    hand.landmark_confidence       = 0.98F;
    hand.palm.box                  = { 40.0F, 40.0F, 160.0F, 160.0F };
    hand.palm.roi.rotation_radians = 0.0F;
    for (auto& point : hand.landmarks)
    {
        point = { 100.0F, 180.0F, 0.0F };
    }
    hand.landmarks[0]  = { 100.0F, 220.0F, 0.0F };
    hand.landmarks[5]  = { 75.0F, 170.0F, 0.0F };
    hand.landmarks[6]  = { 85.0F, 150.0F, 0.0F };
    hand.landmarks[8]  = { 95.0F, 110.0F, 0.0F };
    hand.landmarks[9]  = { 100.0F, 165.0F, 0.0F };
    hand.landmarks[10] = { 100.0F, 140.0F, 0.0F };
    hand.landmarks[12] = { 100.0F, 100.0F, 0.0F };
    hand.landmarks[13] = { 125.0F, 170.0F, 0.0F };
    hand.landmarks[14] = { 128.0F, 145.0F, 0.0F };
    hand.landmarks[16] = { 130.0F, 110.0F, 0.0F };
    hand.landmarks[17] = { 145.0F, 180.0F, 0.0F };
    hand.landmarks[18] = { 150.0F, 155.0F, 0.0F };
    hand.landmarks[20] = { 155.0F, 125.0F, 0.0F };
    return hand;
}

GestureFrameContext frame_context(std::uint64_t serial, int elapsed_ms)
{
    return { serial,
             std::chrono::steady_clock::time_point {} + std::chrono::milliseconds(elapsed_ms), 640,
             480 };
}

} // namespace

spec("hand interaction with THIG temporal gestures")
{
    it("recognizes left and right open-hand swipes in THIG")
    {
        TemporalGraphEngine left_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(immediate_settings()));
        check_true(
            has_action(feed(left_engine, 1, 0,
                            { relation("Shape Open", 1, 1), relation("Direction Left", 1, 1) }),
                       "Swipe Left"));

        TemporalGraphEngine right_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(immediate_settings()));
        check_true(
            has_action(feed(right_engine, 1, 0,
                            { relation("Shape Open", 1, 1), relation("Direction Right", 1, 1) }),
                       "Swipe Right"));
    }

    it("requires sustained direction evidence for default swipes")
    {
        TemporalGraphEngine engine(kfcore::hand_interaction::build_hand_interaction_graph());

        check_false(has_action(
            feed(engine, 1, 0, { relation("Shape Open", 1, 1), relation("Direction Left", 1, 1) }),
            "Swipe Left"));
        check_false(has_action(
            feed(engine, 2, 33, { relation("Shape Open", 1, 2), relation("Direction Left", 1, 2) }),
            "Swipe Left"));
        check_false(has_action(
            feed(engine, 3, 66, { relation("Shape Open", 1, 3), relation("Direction Left", 1, 3) }),
            "Swipe Left"));
        check_true(
            has_action(feed(engine, 4, 100,
                            { relation("Shape Open", 1, 4), relation("Direction Left", 1, 4) }),
                       "Swipe Left"));
    }

    it("confirms a sustained default swipe on a jittered 12 FPS stream")
    {
        const int first_swipe_ms = first_default_swipe_ms({ 0, 86 });
        check(first_swipe_ms == 86);
    }

    it("confirms default swipes across common frame rates and one dropped frame")
    {
        const int at_12_fps       = first_default_swipe_ms({ 0, 86 });
        const int at_30_fps       = first_default_swipe_ms({ 0, 31, 69 });
        const int at_30_fps_drop  = first_default_swipe_ms({ 0, 34, 101 });
        const int at_60_fps       = first_default_swipe_ms({ 0, 15, 34, 51, 69 });
        const int at_120_fps      =
            first_default_swipe_ms({ 0, 8, 17, 25, 34, 42, 51, 59, 68 });

        check(at_12_fps == 86);
        check(at_30_fps == 69);
        check(at_30_fps_drop == 101);
        check(at_60_fps == 69);
        check(at_120_fps == 68);
    }

    it("keeps default direction dwell and window boundaries inclusive")
    {
        const int at_dwell_boundary  = first_default_swipe_ms({ 0, 67 });
        const int at_window_boundary = first_default_swipe_ms({ 0, 167 });
        const int outside_window     = first_default_swipe_ms({ 0, 168 });

        check(at_dwell_boundary == 67);
        check(at_window_boundary == 167);
        check(outside_window == -1);
    }

    it("rejects one opposite direction frame and recovers the stable direction")
    {
        TemporalGraphEngine engine(kfcore::hand_interaction::build_hand_interaction_graph());

        check_false(has_action(
            feed(engine, 1, 0,
                 { relation("Shape Open", 1, 1), relation("Direction Left", 1, 1) }),
            "Swipe Left"));
        check_false(has_action(
            feed(engine, 2, 34,
                 { relation("Shape Open", 1, 2), relation("Direction Left", 1, 2) }),
            "Swipe Left"));
        check_true(has_action(
            feed(engine, 3, 68,
                 { relation("Shape Open", 1, 3), relation("Direction Left", 1, 3) }),
            "Swipe Left"));

        const auto conflict =
            feed(engine, 4, 100,
                 { relation("Shape Open", 1, 4), relation("Direction Right", 1, 4) });
        check_false(has_action(conflict, "Swipe Right"));
        check_false(has_active_relation(engine, "Direction Left"));
        check_false(has_active_relation(engine, "Direction Right"));

        const auto recovered =
            feed(engine, 5, 134,
                 { relation("Shape Open", 1, 5), relation("Direction Left", 1, 5) });
        check_true(has_action(recovered, "Swipe Left"));
        check_true(has_active_relation(engine, "Direction Left"));
    }

    it("does not restore Wave semantics")
    {
        TemporalGraphEngine engine(
            kfcore::hand_interaction::build_hand_interaction_graph(immediate_settings()));
        (void)feed(engine, 1, 0,
                   { relation("Shape Open", 1, 1), relation("Direction Left", 1, 1) });
        (void)feed(engine, 2, 100,
                   { relation("Shape Open", 1, 2), relation("Direction Right", 1, 2) });
        const auto result = feed(
            engine, 3, 200, { relation("Shape Open", 1, 3), relation("Direction Left", 1, 3) });

        check_false(has_action(result, "Wave"));
    }

    it("recognizes primitive gestures without learned gesture events")
    {
        HandInteractionOptions options;
        options.primitives.motion.movement_window_samples = 2;
        options.primitives.motion.stationary_samples      = 2;
        options.temporal                                  = immediate_settings();
        HandInteractionPipeline pipeline(options);
        HandFrame               frame;
        frame.hands.push_back(model_hand(Gesture::Open));

        (void)pipeline.process(frame, frame_context(1, 0));
        (void)pipeline.process(frame, frame_context(2, 20));
        frame.hands[0].gesture = Gesture::Closed;
        (void)pipeline.process(frame, frame_context(3, 40));
        const auto result = pipeline.process(frame, frame_context(4, 60));

        check_true(has_action(result.actions, "Grasp"));
    }

    it("keeps Grasp state on the canonical hand when the raw track id changes")
    {
        HandInteractionOptions options;
        options.primitives.motion.movement_window_samples = 2;
        options.primitives.motion.stationary_samples      = 2;
        options.temporal                                  = immediate_settings();
        HandInteractionPipeline pipeline(options);
        HandFrame               frame;
        frame.hands.push_back(model_hand(Gesture::Open));

        const auto initial = pipeline.process(frame, frame_context(1, 0));
        check_size(initial.primitives.hands, 1U);
        check(initial.primitives.hands[0].raw_track_id == 7);
        check_true(initial.primitives.hands[0].canonical_id > 0);
        const int canonical_id = initial.primitives.hands[0].canonical_id;

        const auto stable_open = pipeline.process(frame, frame_context(2, 20));
        check_size(stable_open.primitives.hands, 1U);
        check(stable_open.primitives.hands[0].canonical_id == canonical_id);

        frame.hands[0].track_id = 71;
        frame.hands[0].gesture  = Gesture::Closed;
        const auto changed      = pipeline.process(frame, frame_context(3, 40));
        check_size(changed.primitives.hands, 1U);
        check(changed.primitives.hands[0].raw_track_id == 71);
        check(changed.primitives.hands[0].canonical_id == canonical_id);

        const auto grasp = pipeline.process(frame, frame_context(4, 60));
        check_size(grasp.primitives.hands, 1U);
        check(grasp.primitives.hands[0].canonical_id == canonical_id);
        check_true(has_action_from(grasp.actions, "Grasp", canonical_id));
    }

    it("uses THIG Grasp Swipe Release relations for deterministic drag state")
    {
        TemporalGraphEngine engine(
            kfcore::hand_interaction::build_hand_interaction_graph(immediate_settings()));
        (void)feed(engine, 1, 0,
                   { relation("Shape Open", 1, 1), relation("Motion Stationary", 1, 1) });
        (void)feed(engine, 2, 20,
                   { relation("Shape Fist", 1, 2), relation("Motion Stationary", 1, 2) });
        const auto grasp = feed(
            engine, 3, 22, { relation("Shape Fist", 1, 3), relation("Motion Stationary", 1, 3) });
        check_true(has_action(grasp, "Grasp"));

        const auto drag_start =
            feed(engine, 4, 40, { relation("Shape Fist", 1, 4), relation("Direction Left", 1, 4) });
        check_true(has_action(drag_start, "Drag Start Left"));
        const auto drag = feed(engine, 5, 60,
                               { relation("Shape Fist", 1, 5), relation("Direction Right", 1, 5) });
        check_true(has_action(drag, "Drag Right"));

        (void)feed(engine, 6, 80,
                   { relation("Shape Open", 1, 6), relation("Motion Stationary", 1, 6) });
        const auto release = feed(
            engine, 7, 82, { relation("Shape Open", 1, 7), relation("Motion Stationary", 1, 7) });
        check_true(has_action(release, "Drag End"));
    }

    it("shares default direction stabilization across Drag and neutral rearm")
    {
        const auto settings = shared_default_direction_settings();

        TemporalGraphEngine drag_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(settings));
        (void)feed(drag_engine, 1, 0,
                   { relation("Shape Open", 1, 1), relation("Motion Stationary", 1, 1) });
        (void)feed(drag_engine, 2, 20,
                   { relation("Shape Fist", 1, 2), relation("Motion Stationary", 1, 2) });
        const auto grasp =
            feed(drag_engine, 3, 22,
                 { relation("Shape Fist", 1, 3), relation("Motion Stationary", 1, 3) });
        check_true(has_action(grasp, "Grasp"));
        check_false(has_action(
            feed(drag_engine, 4, 40,
                 { relation("Shape Fist", 1, 4), relation("Direction Left", 1, 4) }),
            "Drag Start Left"));
        check_true(has_action(
            feed(drag_engine, 5, 108,
                 { relation("Shape Fist", 1, 5), relation("Direction Left", 1, 5) }),
            "Drag Start Left"));
        (void)feed(drag_engine, 6, 120,
                   { relation("Shape Open", 1, 6), relation("Motion Stationary", 1, 6) });
        const auto drag_end =
            feed(drag_engine, 7, 122,
                 { relation("Shape Open", 1, 7), relation("Motion Stationary", 1, 7) });
        check_true(has_action(drag_end, "Drag End"));
        check(drag_engine.StateOf("hand_interaction_cycle") == "await_neutral");

        (void)feed(drag_engine, 8, 140,
                   { relation("Shape Open", 1, 8), relation("Direction Neutral", 1, 8) });
        check(drag_engine.StateOf("hand_interaction_cycle") == "await_neutral");
        (void)feed(drag_engine, 9, 208,
                   { relation("Shape Open", 1, 9), relation("Direction Neutral", 1, 9) });
        check(drag_engine.StateOf("hand_interaction_cycle") == "armed");
    }

    it("lets a second hand grasp while another open hand remains visible")
    {
        TemporalGraphEngine engine(
            kfcore::hand_interaction::build_hand_interaction_graph(immediate_settings()));
        (void)feed(engine, 1, 0,
                   { relation("Shape Open", 1, 1), relation("Motion Stationary", 1, 1),
                     relation("Shape Open", 2, 1), relation("Motion Stationary", 2, 1) });
        (void)feed(engine, 2, 20,
                   { relation("Shape Open", 1, 2), relation("Motion Stationary", 1, 2),
                     relation("Shape Fist", 2, 2), relation("Motion Stationary", 2, 2) });
        const auto grasp =
            feed(engine, 3, 22,
                 { relation("Shape Open", 1, 3), relation("Motion Stationary", 1, 3),
                   relation("Shape Fist", 2, 3), relation("Motion Stationary", 2, 3) });

        check_true(has_action_from(grasp, "Grasp", 2));
    }

    it("retains static and spatial THIG semantic actions")
    {
        const auto settings = immediate_settings();

        TemporalGraphEngine ok_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(settings));
        check_true(
            has_action(feed(ok_engine, 1, 0,
                            { relation("Pose OK", 1, 1), relation("Motion Stationary", 1, 1) }),
                       "OK"));

        TemporalGraphEngine zoom_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(settings));
        check_true(has_action(
            feed(zoom_engine, 1, 0, { relation("Hands Distance Expanding", 1, 1, 2) }), "Zoom In"));
    }

    it("recognizes V and Fist held by two distinct hands")
    {
        TemporalGraphEngine engine(
            kfcore::hand_interaction::build_hand_interaction_graph(immediate_settings()));

        const auto actions =
            feed(engine, 1, 0,
                 { relation("Shape V", 1, 1), relation("Motion Stationary", 1, 1),
                   relation("Shape Fist", 2, 1), relation("Motion Stationary", 2, 1) });

        check_true(has_action(actions, "V Fist"));
    }

    it("requires current evidence from both two-hand shape participants")
    {
        TemporalGraphEngine v_fist_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(immediate_settings()));

        check_false(has_action(
            feed(v_fist_engine, 1, 0,
                 { relation("Shape V", 1, 1), relation("Motion Stationary", 1, 1) }),
            "V Fist"));
        check_false(has_action(
            feed(v_fist_engine, 2, 20,
                 { relation("Shape Fist", 2, 2), relation("Motion Stationary", 2, 2) }),
            "V Fist"));

        const auto current_v_fist =
            feed(v_fist_engine, 3, 40,
                 { relation("Shape V", 1, 3), relation("Motion Stationary", 1, 3),
                   relation("Shape Fist", 2, 3), relation("Motion Stationary", 2, 3) });
        check_true(has_action(current_v_fist, "V Fist"));

        TemporalGraphEngine two_v_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(immediate_settings()));
        check_false(has_action(
            feed(two_v_engine, 1, 0,
                 { relation("Shape V", 1, 1), relation("Motion Stationary", 1, 1) }),
            "Two Hand V"));
        check_false(has_action(
            feed(two_v_engine, 2, 20,
                 { relation("Shape V", 2, 2), relation("Motion Stationary", 2, 2) }),
            "Two Hand V"));

        const auto current_two_v =
            feed(two_v_engine, 3, 40,
                 { relation("Shape V", 1, 3), relation("Motion Stationary", 1, 3),
                   relation("Shape V", 2, 3), relation("Motion Stationary", 2, 3) });
        check_true(has_action(current_two_v, "Two Hand V"));
    }

    it("recognizes V held by two stationary distinct hands")
    {
        TemporalGraphEngine engine(
            kfcore::hand_interaction::build_hand_interaction_graph(immediate_settings()));

        const auto actions =
            feed(engine, 1, 0,
                 { relation("Shape V", 1, 1), relation("Motion Stationary", 1, 1),
                   relation("Shape V", 2, 1), relation("Motion Stationary", 2, 1) });

        check_true(has_action(actions, "Two Hand V"));
    }

    it("rejects V and Fist while either participating hand is moving")
    {
        TemporalGraphEngine moving_fist_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(immediate_settings()));

        const auto moving_fist =
            feed(moving_fist_engine, 1, 0,
                 { relation("Shape V", 1, 1), relation("Motion Stationary", 1, 1),
                   relation("Shape Fist", 2, 1), relation("Motion Unstable", 2, 1) });
        check_false(has_action(moving_fist, "V Fist"));

        TemporalGraphEngine moving_v_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(immediate_settings()));
        const auto moving_v =
            feed(moving_v_engine, 1, 0,
                 { relation("Shape V", 1, 1), relation("Motion Unstable", 1, 1),
                   relation("Shape Fist", 2, 1), relation("Motion Stationary", 2, 1) });

        check_false(has_action(moving_v, "V Fist"));
    }

    it("retains validated external Region context without Click semantics")
    {
        HandInteractionOptions options;
        options.temporal = immediate_settings();
        HandInteractionPipeline pipeline(options);
        HandFrame               frame;
        frame.hands.push_back(model_hand());

        const auto identity     = pipeline.process(frame, frame_context(1, 0));
        const int  canonical_id = identity.primitives.hands.front().canonical_id;
        const auto contextual   = pipeline.process(frame, frame_context(2, 20),
                                                   { relation("Region Center", canonical_id, 2) });

        check_true(std::any_of(contextual.primitives.observations.begin(),
                               contextual.primitives.observations.end(), [](const Observation& item)
                               { return item.relation == "Region Center"; }));
        check_false(has_action(contextual.actions, "Click Center"));
        check_throws_as(pipeline.process(frame, frame_context(3, 40),
                                         { relation("Region Center", canonical_id + 1000, 3) }),
                        std::invalid_argument);
    }

    it("counts external Region context against the frame observation capacity")
    {
        HandInteractionOptions options;
        options.temporal                            = immediate_settings();
        options.temporal.max_observations_per_frame = 8;
        HandInteractionPipeline pipeline(options);
        HandFrame               frame;
        frame.hands.push_back(model_hand());

        const auto identity     = pipeline.process(frame, frame_context(1, 0));
        const int  canonical_id = identity.primitives.hands.front().canonical_id;

        check_throws_as(pipeline.process(frame, frame_context(2, 20),
                                         { relation("Region Center", canonical_id, 2) }),
                        std::length_error);
    }

    it("reset clears THIG gesture state")
    {
        HandInteractionOptions options;
        options.primitives.motion.movement_window_samples = 2;
        options.primitives.motion.stationary_samples      = 2;
        options.temporal                                  = immediate_settings();
        HandInteractionPipeline pipeline(options);
        HandFrame               frame;
        frame.hands.push_back(model_hand(Gesture::Open));

        (void)pipeline.process(frame, frame_context(1, 0));
        (void)pipeline.process(frame, frame_context(2, 20));
        frame.hands[0].gesture = Gesture::Closed;
        (void)pipeline.process(frame, frame_context(3, 40));
        const auto grasp = pipeline.process(frame, frame_context(4, 60));
        check_true(has_action(grasp.actions, "Grasp"));
        pipeline.reset();

        const auto after_reset = pipeline.process(frame, frame_context(10, 200));
        check_false(has_action(after_reset.actions, "Grasp"));
    }
}
