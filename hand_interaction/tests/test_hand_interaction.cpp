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
using kfcore::thig::ActionEvent;
using kfcore::thig::Observation;
using kfcore::thig::TemporalGraphEngine;
using kfcore::vision_models::Gesture;
using kfcore::vision_models::HandFrame;
using kfcore::vision_models::HandResult;

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
    settings.neutral_rearm_ms                          = 0;
    settings.single_hand_v_dwell_ms                    = 0;
    settings.dual_hand_dwell_ms                        = 0;
    settings.spatial_dwell_ms                          = 0;
    settings.shape_window_ms                           = 1;
    settings.shape_minimum_supporting_observations     = 1;
    settings.shape_minimum_support_ratio               = 1.0F;
    settings.shape_switch_margin                       = 0.0F;
    settings.shape_maximum_samples_per_source          = 1;
    settings.region_window_ms                          = 1;
    settings.region_minimum_supporting_observations    = 1;
    settings.region_minimum_support_ratio              = 1.0F;
    settings.region_switch_margin                      = 0.0F;
    settings.region_maximum_samples_per_source         = 1;
    return settings;
}

bool has_action(const std::vector<ActionEvent>& actions, const std::string& name)
{
    return std::any_of(actions.begin(), actions.end(),
                       [&](const ActionEvent& action) { return action.action == name; });
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

HandResult model_hand(Gesture gesture)
{
    HandResult hand;
    hand.track_id            = 0;
    hand.gesture             = gesture;
    hand.palm.confidence     = 0.98F;
    hand.landmark_confidence = 0.98F;
    hand.palm.box            = { 40.0F, 40.0F, 160.0F, 160.0F };
    for (auto& point : hand.landmarks)
    {
        point = { 100.0F, 180.0F, 0.0F };
    }
    hand.landmarks[0]  = { 100.0F, 220.0F, 0.0F };
    hand.landmarks[5]  = { 75.0F, 170.0F, 0.0F };
    hand.landmarks[6]  = { 85.0F, 170.0F, 0.0F };
    hand.landmarks[8]  = { 95.0F, 180.0F, 0.0F };
    hand.landmarks[9]  = { 100.0F, 165.0F, 0.0F };
    hand.landmarks[13] = { 125.0F, 170.0F, 0.0F };
    hand.landmarks[17] = { 145.0F, 180.0F, 0.0F };
    return hand;
}

GestureFrameContext frame_context(std::uint64_t serial, int elapsed_ms)
{
    return { serial,
             std::chrono::steady_clock::time_point {} + std::chrono::milliseconds(elapsed_ms), 640,
             480 };
}

} // namespace

spec("hand interaction")
{
    it("recognizes the bounded open-palm three-stroke Wave sequence")
    {
        TemporalGraphEngine engine(
            kfcore::hand_interaction::build_hand_interaction_graph(immediate_settings()));
        check_empty(
            feed(engine, 1, 0, { relation("Shape Open", 1, 1), relation("Direction Left", 1, 1) }));
        check_empty(feed(engine, 2, 100,
                         { relation("Shape Open", 1, 2), relation("Direction Right", 1, 2) }));
        const auto actions = feed(
            engine, 3, 200, { relation("Shape Open", 1, 3), relation("Direction Left", 1, 3) });
        check_true(has_action(actions, "Wave"));
    }

    it("recognizes OK, dual-hand V, Zoom and Rotate semantic actions")
    {
        const auto          settings = immediate_settings();
        TemporalGraphEngine ok_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(settings));
        check_true(
            has_action(feed(ok_engine, 1, 0,
                            { relation("Pose OK", 1, 1), relation("Motion Stationary", 1, 1) }),
                       "OK"));

        TemporalGraphEngine dual_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(settings));
        check_true(has_action(
            feed(dual_engine, 1, 0, { relation("Shape V", 1, 1), relation("Shape V", 2, 1) }),
            "Two Hand V"));

        TemporalGraphEngine zoom_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(settings));
        check_true(has_action(
            feed(zoom_engine, 1, 0, { relation("Hands Distance Expanding", 1, 1, 2) }), "Zoom In"));

        TemporalGraphEngine rotate_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(settings));
        check_true(has_action(
            feed(rotate_engine, 1, 0,
                 { relation("Rotation Clockwise", 1, 1), relation("Motion Stationary", 1, 1) }),
            "Rotate Clockwise"));
    }

    it("recognizes one Grasp and one Release for the same canonical hand")
    {
        TemporalGraphEngine engine(
            kfcore::hand_interaction::build_hand_interaction_graph(immediate_settings()));
        (void)feed(engine, 1, 0,
                   { relation("Shape Open", 1, 1), relation("Motion Stationary", 1, 1) });
        (void)feed(engine, 2, 100,
                   { relation("Shape Fist", 1, 2), relation("Motion Stationary", 1, 2) });
        const auto grasp = feed(
            engine, 3, 102, { relation("Shape Fist", 1, 3), relation("Motion Stationary", 1, 3) });
        check_true(has_action(grasp, "Grasp"));

        (void)feed(engine, 4, 200,
                   { relation("Shape Open", 1, 4), relation("Motion Stationary", 1, 4) });
        const auto release = feed(
            engine, 5, 202, { relation("Shape Open", 1, 5), relation("Motion Stationary", 1, 5) });
        check_true(has_action(release, "Release"));
    }

    it("runs model primitives and THIG through one resettable facade")
    {
        HandInteractionOptions options;
        options.primitives.motion.movement_window_samples = 2;
        options.primitives.motion.stationary_samples      = 2;
        options.temporal                                  = immediate_settings();
        HandInteractionPipeline pipeline(options);
        HandFrame               frame;
        frame.hands.push_back(model_hand(Gesture::Open));

        (void)pipeline.process(frame, frame_context(1, 0));
        (void)pipeline.process(frame, frame_context(2, 33));
        frame.hands[0].gesture = Gesture::Closed;
        (void)pipeline.process(frame, frame_context(3, 66));
        const auto result = pipeline.process(frame, frame_context(4, 99));
        check_true(has_action(result.actions, "Grasp"));

        pipeline.reset();
        const auto after_reset = pipeline.process(frame, frame_context(10, 200));
        check_false(has_action(after_reset.actions, "Grasp"));
    }

    it("copies declared external Region evidence at the facade boundary")
    {
        HandInteractionOptions options;
        options.temporal = immediate_settings();
        HandInteractionPipeline pipeline(options);
        HandFrame               frame;
        frame.hands.push_back(model_hand(Gesture::Open));
        std::vector<Observation> external = { relation("Region Center", 1, 1) };

        const auto result         = pipeline.process(frame, frame_context(1, 0), external);
        external.front().relation = "Region Left";
        check_true(std::any_of(result.primitives.observations.begin(),
                               result.primitives.observations.end(), [](const Observation& item)
                               { return item.relation == "Region Center"; }));
    }

    it("rejects invalid external evidence before advancing primitive state")
    {
        HandInteractionOptions options;
        options.temporal = immediate_settings();
        HandInteractionPipeline pipeline(options);
        HandFrame               frame;
        frame.hands.push_back(model_hand(Gesture::Open));
        std::vector<Observation> invalid = { relation("Region Center", 1, 2) };

        check_throws_as(pipeline.process(frame, frame_context(1, 0), invalid),
                        std::invalid_argument);
        const auto accepted = pipeline.process(frame, frame_context(1, 0));
        check_true(accepted.primitives.hands.size() == 1U);
    }

    it("rejects frames that cannot fit the configured THIG capacity")
    {
        HandInteractionOptions options;
        options.temporal                            = immediate_settings();
        options.temporal.max_observations_per_frame = 6;
        HandInteractionPipeline pipeline(options);
        HandFrame               frame;
        frame.hands.push_back(model_hand(Gesture::Open));

        check_throws_as(pipeline.process(frame, frame_context(1, 0)), std::length_error);
    }
}
