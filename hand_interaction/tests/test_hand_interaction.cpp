#include "kfcore/hand_interaction/hand_interaction.hpp"
#include "tinytest.hpp"

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

namespace
{

using kfcore::hand_gesture::GestureClass;
using kfcore::hand_gesture::GestureEvent;
using kfcore::hand_gesture::GesturePhase;
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
    settings.ok_dwell_ms = 0;
    settings.observation_max_gap_ms = 350;
    settings.single_hand_v_dwell_ms = 0;
    settings.dual_hand_dwell_ms = 0;
    settings.dual_hand_onset_window_ms = 0;
    settings.spatial_dwell_ms = 0;
    settings.shape_window_ms = 1;
    settings.shape_minimum_supporting_observations = 1;
    settings.shape_minimum_support_ratio = 1.0F;
    settings.shape_switch_margin = 0.0F;
    settings.shape_maximum_samples_per_source = 1;
    settings.history_ms = 1000;
    settings.rotation_cooldown_ms = 0;
    return settings;
}

bool has_action(const std::vector<ActionEvent>& actions, const std::string& name)
{
    return std::any_of(actions.begin(), actions.end(),
                       [&](const ActionEvent& action) { return action.action == name; });
}

Observation relation(std::string name, int source_id, std::uint64_t serial,
                     int target_id = 0)
{
    Observation value;
    value.serial = serial;
    value.source = {"hand", source_id};
    if (target_id > 0)
    {
        value.target = kfcore::thig::EntityRef {"hand", target_id};
    }
    value.relation = std::move(name);
    value.confidence = 0.95F;
    value.producer = "test";
    return value;
}

std::vector<ActionEvent> feed(TemporalGraphEngine& engine, std::uint64_t serial,
                              int elapsed_ms,
                              std::initializer_list<Observation> observations)
{
    std::vector<Observation> frame(observations);
    for (Observation& item : frame)
    {
        item.serial = serial;
    }
    return engine.ProcessFrame(
        frame,
        std::chrono::steady_clock::time_point {} + std::chrono::milliseconds(elapsed_ms));
}

HandResult model_hand(Gesture gesture = Gesture::Open)
{
    HandResult hand;
    hand.track_id = 7;
    hand.gesture = gesture;
    hand.palm.confidence = 0.98F;
    hand.landmark_confidence = 0.98F;
    hand.palm.box = {40.0F, 40.0F, 160.0F, 160.0F};
    hand.palm.roi.rotation_radians = 0.0F;
    for (auto& point : hand.landmarks)
    {
        point = {100.0F, 180.0F, 0.0F};
    }
    hand.landmarks[0]  = {100.0F, 220.0F, 0.0F};
    hand.landmarks[5]  = {75.0F, 170.0F, 0.0F};
    hand.landmarks[6]  = {85.0F, 150.0F, 0.0F};
    hand.landmarks[8]  = {95.0F, 110.0F, 0.0F};
    hand.landmarks[9]  = {100.0F, 165.0F, 0.0F};
    hand.landmarks[10] = {100.0F, 140.0F, 0.0F};
    hand.landmarks[12] = {100.0F, 100.0F, 0.0F};
    hand.landmarks[13] = {125.0F, 170.0F, 0.0F};
    hand.landmarks[14] = {128.0F, 145.0F, 0.0F};
    hand.landmarks[16] = {130.0F, 110.0F, 0.0F};
    hand.landmarks[17] = {145.0F, 180.0F, 0.0F};
    hand.landmarks[18] = {150.0F, 155.0F, 0.0F};
    hand.landmarks[20] = {155.0F, 125.0F, 0.0F};
    return hand;
}

GestureFrameContext frame_context(std::uint64_t serial, int elapsed_ms)
{
    return {serial,
            std::chrono::steady_clock::time_point {} + std::chrono::milliseconds(elapsed_ms),
            640, 480};
}

GestureEvent gesture(int track_id, GestureClass gesture_class, GesturePhase phase,
                     std::uint64_t timestamp_ns = 0U)
{
    return {track_id, gesture_class, phase, 0.95F, timestamp_ns};
}

} // namespace

spec("hand interaction with learned temporal gestures")
{
    it("accepts Wave directly from the GRU instead of reconstructing motion reversals")
    {
        HandInteractionOptions options;
        options.semantic = immediate_settings();
        HandInteractionPipeline pipeline(options);
        HandFrame frame;
        frame.hands.push_back(model_hand());

        const auto result = pipeline.process(
            frame, frame_context(1, 0),
            {gesture(7, GestureClass::Wave, GesturePhase::End)});

        check_true(has_action(result.actions, "Wave"));
    }

    it("does not recognize Wave from the remaining THIG primitive graph")
    {
        TemporalGraphEngine engine(
            kfcore::hand_interaction::build_hand_interaction_graph(immediate_settings()));

        check_empty(feed(engine, 1, 0,
                         {relation("Shape Open", 1, 1), relation("Direction Left", 1, 1)}));
        check_empty(feed(engine, 2, 100,
                         {relation("Shape Open", 1, 2), relation("Direction Right", 1, 2)}));
        const auto actions = feed(
            engine, 3, 200,
            {relation("Shape Open", 1, 3), relation("Direction Left", 1, 3)});
        check_false(has_action(actions, "Wave"));
    }

    it("uses learned Grab Swipe Release events for deterministic drag state")
    {
        HandInteractionOptions options;
        options.semantic = immediate_settings();
        HandInteractionPipeline pipeline(options);
        HandFrame frame;
        frame.hands.push_back(model_hand());

        const auto grasp = pipeline.process(
            frame, frame_context(1, 0),
            {gesture(7, GestureClass::Grab, GesturePhase::Start)});
        check_true(has_action(grasp.actions, "Grasp"));

        const auto drag_start = pipeline.process(
            frame, frame_context(2, 20),
            {gesture(7, GestureClass::SwipeLeft, GesturePhase::End)});
        check_true(has_action(drag_start.actions, "Drag Start Left"));

        const auto drag = pipeline.process(
            frame, frame_context(3, 40),
            {gesture(7, GestureClass::SwipeRight, GesturePhase::End)});
        check_true(has_action(drag.actions, "Drag Right"));

        const auto release = pipeline.process(
            frame, frame_context(4, 60),
            {gesture(7, GestureClass::Release, GesturePhase::Start)});
        check_true(has_action(release.actions, "Drag End"));
    }

    it("maps a learned Click end event to the current external Region")
    {
        HandInteractionOptions options;
        options.semantic = immediate_settings();
        HandInteractionPipeline pipeline(options);
        HandFrame frame;
        frame.hands.push_back(model_hand());

        const auto identity = pipeline.process(frame, frame_context(1, 0), {});
        check_true(identity.primitives.hands.size() == 1U);
        const int canonical_id = identity.primitives.hands.front().canonical_id;
        check_true(canonical_id > 0);

        const auto click = pipeline.process(
            frame, frame_context(2, 20),
            {gesture(7, GestureClass::Click, GesturePhase::End)},
            {relation("Region Center", canonical_id, 2)});
        check_true(has_action(click.actions, "Click Center"));
    }

    it("retains static and spatial THIG semantic actions")
    {
        const auto settings = immediate_settings();

        TemporalGraphEngine ok_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(settings));
        check_true(has_action(
            feed(ok_engine, 1, 0,
                 {relation("Pose OK", 1, 1), relation("Motion Stationary", 1, 1)}),
            "OK"));

        TemporalGraphEngine zoom_engine(
            kfcore::hand_interaction::build_hand_interaction_graph(settings));
        check_true(has_action(
            feed(zoom_engine, 1, 0,
                 {relation("Hands Distance Expanding", 1, 1, 2)}),
            "Zoom In"));
    }

    it("reset clears learned drag state")
    {
        HandInteractionOptions options;
        options.semantic = immediate_settings();
        HandInteractionPipeline pipeline(options);
        HandFrame frame;
        frame.hands.push_back(model_hand());

        (void)pipeline.process(
            frame, frame_context(1, 0),
            {gesture(7, GestureClass::Grab, GesturePhase::Start)});
        pipeline.reset();

        const auto swipe = pipeline.process(
            frame, frame_context(2, 20),
            {gesture(7, GestureClass::SwipeLeft, GesturePhase::End)});
        check_true(has_action(swipe.actions, "Swipe Left"));
        check_false(has_action(swipe.actions, "Drag Start Left"));
    }
}
