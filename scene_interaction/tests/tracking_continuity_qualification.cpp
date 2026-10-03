#include "scene_graph_detail.hpp"

#include "kfcore/scene_interaction/interaction.hpp"
#include "kfcore/yolo/tracking.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using kfcore::pipelines::SceneGraphFrame;
using kfcore::scene_interaction::SceneBehaviorEvent;
using kfcore::scene_interaction::SceneBehaviorEventKind;
using kfcore::scene_interaction::SceneBehaviorEventReason;
using kfcore::scene_interaction::SceneBehaviorModel;
using kfcore::scene_interaction::SceneInteraction;
using kfcore::scene_interaction::SceneInteractionOptions;
using kfcore::yolo::ByteTrackOptions;
using kfcore::yolo::ByteTrackSession;
using kfcore::yolo::Detection;
using kfcore::yolo::DetectionFrame;
using kfcore::yolo::TrackFrame;

void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

DetectionFrame detection_frame()
{
    DetectionFrame frame;
    frame.image_width = 100;
    frame.image_height = 100;
    frame.detections = {
        {{5.0F, 5.0F, 25.0F, 25.0F}, 0.95F, 0},
        {{60.0F, 5.0F, 80.0F, 25.0F}, 0.95F, 1},
    };
    return frame;
}

ByteTrackOptions tracking_options()
{
    ByteTrackOptions options;
    options.minimum_consecutive_frames = 1;
    options.track_activation_threshold = 0.5F;
    options.high_conf_det_threshold = 0.5F;
    options.minimum_iou_threshold = 0.1F;
    return options;
}

SceneBehaviorModel model(std::uint64_t vocabulary_version)
{
    SceneBehaviorModel value;
    value.predicate_count = 1U;
    value.vocabulary_version = vocabulary_version;
    value.reservoir_size = 1;
    value.leak_rate = 1.0F;
    value.neutral_index = 0U;
    value.labels = {"none", "interacting"};
    value.input_weights.assign(
        value.predicate_count +
            kfcore::scene_interaction::kSceneGeometryFeatureCount,
        0.0F);
    value.input_weights[0] = 1.0F;
    value.recurrent_weights = {0.0F};
    value.reservoir_bias = {0.0F};
    value.output_weights = {0.0F, 1.0F};
    value.output_bias = {0.0F, 0.0F};
    return value;
}

SceneInteractionOptions interaction_options()
{
    SceneInteractionOptions options;
    options.minimum_score = 0.5F;
    options.minimum_margin = 0.2F;
    options.confirmation_seconds = 0.10;
    options.end_seconds = 0.10;
    options.maximum_gap_seconds = 0.50;
    options.max_pair_states = 4U;
    return options;
}

SceneGraphFrame scene_from_tracks(
    TrackFrame tracks,
    std::uint64_t vocabulary_version,
    float relation_score = 1.0F)
{
    kfcore::relation::RelationFrame relations;
    relations.image_width = tracks.image_width;
    relations.image_height = tracks.image_height;
    relations.vocabulary_version = vocabulary_version;

    if (tracks.detections.size() >= 2U &&
        tracks.detections[0].track_id &&
        tracks.detections[1].track_id)
    {
        relations.edges.push_back({
            0U,
            1U,
            0U,
            relation_score,
            tracks.detections[0].track_id,
            tracks.detections[1].track_id,
        });
    }

    return kfcore::pipelines::detail::assemble_scene_graph(
        std::move(tracks),
        std::move(relations));
}

SceneGraphFrame confirmed_scene(
    ByteTrackSession& session,
    std::uint64_t vocabulary_version)
{
    (void)session.update(detection_frame());
    return scene_from_tracks(
        session.update(detection_frame()),
        vocabulary_version);
}

void require_event(
    const std::vector<SceneBehaviorEvent>& events,
    SceneBehaviorEventKind kind,
    SceneBehaviorEventReason reason,
    const char* message)
{
    require(events.size() == 1U, message);
    require(events[0].kind == kind, message);
    require(events[0].reason == reason, message);
}

void write_report(
    const std::filesystem::path& path,
    std::uint64_t old_epoch,
    std::uint64_t new_epoch,
    std::uint64_t old_subject,
    std::uint64_t old_object,
    std::uint64_t new_subject,
    std::uint64_t new_object)
{
    if (path.empty() || std::filesystem::exists(path))
    {
        throw std::runtime_error("invalid qualification report path");
    }

    std::ofstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("cannot create qualification report");
    }

    stream
        << "{\n"
        << "  \"schema\": \"kfcore.tracked-relation-continuity/1\",\n"
        << "  \"passed\": true,\n"
        << "  \"tracking_backend\": \"ByteTrack\",\n"
        << "  \"tracking_epoch_before_reset\": " << old_epoch << ",\n"
        << "  \"tracking_epoch_after_reset\": " << new_epoch << ",\n"
        << "  \"numeric_track_ids_reused\": "
        << ((old_subject == new_subject && old_object == new_object)
                ? "true"
                : "false")
        << ",\n"
        << "  \"old_pair\": {"
        << "\"subject\":" << old_subject
        << ",\"object\":" << old_object
        << ",\"epoch\":" << old_epoch << "},\n"
        << "  \"new_pair\": {"
        << "\"subject\":" << new_subject
        << ",\"object\":" << new_object
        << ",\"epoch\":" << new_epoch << "},\n"
        << "  \"tracking_reset_reason\": \"tracking reset\",\n"
        << "  \"vocabulary_change_reason\": \"vocabulary changed\",\n"
        << "  \"pair_loss_reason\": \"pair lost\",\n"
        << "  \"frame_gap_reason\": \"frame gap\"\n"
        << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 2)
        {
            std::cerr
                << "usage: tracking_continuity_qualification <report.json>\n";
            return 2;
        }

        ByteTrackSession tracker(tracking_options());
        require(
            tracker.tracking_epoch() == 1U,
            "ByteTrack initial epoch must be 1");

        SceneInteraction interaction(interaction_options());
        (void)interaction.configure_model(model(7U), 0.0);

        SceneGraphFrame old_scene =
            confirmed_scene(tracker, 7U);
        require(
            old_scene.objects.tracking_epoch == 1U,
            "confirmed scene lost initial tracking epoch");
        require(
            old_scene.relations.edges.size() == 1U,
            "confirmed scene must have one tracked relation");

        const auto old_subject =
            *old_scene.objects.detections[0].track_id;
        const auto old_object =
            *old_scene.objects.detections[1].track_id;

        require(
            interaction.process(old_scene, 0.0).empty(),
            "first old-epoch observation must only create candidate state");
        const auto started =
            interaction.process(
                scene_from_tracks(
                    tracker.update(detection_frame()),
                    7U),
                0.20);
        require_event(
            started,
            SceneBehaviorEventKind::BehaviorStarted,
            SceneBehaviorEventReason::Recognized,
            "old-epoch behavior did not start");
        require(
            interaction.pair_state_count() == 1U,
            "old-epoch pair state missing");

        const std::uint64_t new_epoch =
            tracker.reset();
        require(
            new_epoch == 2U,
            "ByteTrack reset did not advance identity epoch");

        // The first post-reset frame may still be tentative, but the frame
        // itself carries the new identity epoch and must invalidate old state.
        SceneGraphFrame reset_scene =
            scene_from_tracks(
                tracker.update(detection_frame()),
                7U);
        const auto reset_events =
            interaction.process(reset_scene, 0.30);
        require_event(
            reset_events,
            SceneBehaviorEventKind::BehaviorCancelled,
            SceneBehaviorEventReason::TrackingReset,
            "tracking reset did not cancel active old-epoch behavior");
        require(
            reset_events[0].pair.tracking_epoch == 1U,
            "tracking reset cancellation lost old pair epoch");
        require(
            interaction.pair_state_count() == 0U,
            "old pair state survived tracking reset");

        SceneGraphFrame new_scene =
            scene_from_tracks(
                tracker.update(detection_frame()),
                7U);
        require(
            new_scene.objects.tracking_epoch == 2U,
            "new scene lost advanced tracking epoch");
        require(
            new_scene.relations.edges.size() == 1U,
            "new epoch failed to regain tracked relation");

        const auto new_subject =
            *new_scene.objects.detections[0].track_id;
        const auto new_object =
            *new_scene.objects.detections[1].track_id;
        require(
            old_subject == new_subject &&
                old_object == new_object,
            "qualification requires numeric track-ID reuse after reset");

        require(
            interaction.process(new_scene, 0.40).empty(),
            "new-epoch first observation must start fresh state");
        require(
            interaction.pair_state_count() == 1U,
            "new epoch did not create a fresh pair state");

        const auto restarted =
            interaction.process(
                scene_from_tracks(
                    tracker.update(detection_frame()),
                    7U),
                0.55);
        require_event(
            restarted,
            SceneBehaviorEventKind::BehaviorStarted,
            SceneBehaviorEventReason::Recognized,
            "new-epoch behavior did not start");
        require(
            restarted[0].pair.tracking_epoch == 2U,
            "new behavior did not bind the new tracking epoch");

        // Vocabulary change cancels active state and the mismatched frame is
        // not consumed.
        const auto vocabulary_changed =
            interaction.process(
                scene_from_tracks(
                    tracker.update(detection_frame()),
                    8U),
                0.65);
        require_event(
            vocabulary_changed,
            SceneBehaviorEventKind::BehaviorCancelled,
            SceneBehaviorEventReason::VocabularyChanged,
            "vocabulary change did not cancel active state");
        require(
            interaction.pair_state_count() == 0U,
            "vocabulary-mismatched frame was consumed");

        require(
            interaction.process(
                scene_from_tracks(
                    tracker.update(detection_frame()),
                    8U),
                0.75).empty(),
            "repeated mismatched vocabulary must remain ignored");
        require(
            interaction.pair_state_count() == 0U,
            "mismatched vocabulary recreated pair state");

        (void)interaction.configure_model(model(8U), 0.75);
        require(
            interaction.process(
                scene_from_tracks(
                    tracker.update(detection_frame()),
                    8U),
                0.85).empty(),
            "reconfigured vocabulary first frame must create fresh state");
        const auto version8_started =
            interaction.process(
                scene_from_tracks(
                    tracker.update(detection_frame()),
                    8U),
                1.00);
        require_event(
            version8_started,
            SceneBehaviorEventKind::BehaviorStarted,
            SceneBehaviorEventReason::Recognized,
            "temporal processing did not resume after vocabulary reconfigure");

        const auto pair_lost =
            interaction.advance(1.60);
        require_event(
            pair_lost,
            SceneBehaviorEventKind::BehaviorCancelled,
            SceneBehaviorEventReason::PairLost,
            "pair loss reason drifted");

        require(
            interaction.process(
                scene_from_tracks(
                    tracker.update(detection_frame()),
                    8U),
                1.70).empty(),
            "pair-loss recovery first frame must create fresh state");
        (void)interaction.process(
            scene_from_tracks(
                tracker.update(detection_frame()),
                8U),
            1.90);

        const auto frame_gap =
            interaction.process(
                scene_from_tracks(
                    tracker.update(detection_frame()),
                    8U),
                2.50);
        require_event(
            frame_gap,
            SceneBehaviorEventKind::BehaviorCancelled,
            SceneBehaviorEventReason::FrameGap,
            "frame gap reason drifted");

        write_report(
            argv[1],
            1U,
            new_epoch,
            old_subject,
            old_object,
            new_subject,
            new_object);

        std::cout
            << "tracked relation continuity qualification: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "tracked relation continuity qualification: FAIL: "
            << error.what() << '\n';
        return 1;
    }
}
