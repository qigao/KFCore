#include <kfcore/gesture_interaction/interaction.hpp>
#include <tinytest.hpp>
#include <algorithm>
#include <limits>

namespace {
using namespace kfcore::gesture_interaction;
using Gesture = kfcore::mediapipe::CannedGesture;
kfcore::mediapipe::GestureFrame frame(Gesture gesture = Gesture::OpenPalm, float right = 0.95F) {
    kfcore::mediapipe::GestureFrame result;
    result.landmarks.hands.resize(1); result.gestures.resize(1);
    result.landmarks.hands[0].right_hand_probability = right;
    result.landmarks.hands[0].landmarks[0].x = 320;
    result.landmarks.hands[0].landmarks[0].y = 240;
    result.gestures[0].label = gesture;
    result.gestures[0].scores[std::size_t(gesture)] = 1;
    return result;
}
bool has(const std::vector<Event>& events, EventKind kind) {
    return std::any_of(events.begin(), events.end(), [kind](const Event& event) { return event.kind == kind; });
}
void start_basic(GestureInteraction& runtime, Gesture gesture = Gesture::OpenPalm) {
    (void)runtime.process(frame(gesture), {0,640,480});
    (void)runtime.process(frame(gesture), {0.1,640,480});
    check_true(has(runtime.process(frame(gesture), {0.2,640,480}), EventKind::GestureStarted));
}
// Fixed finite weights isolate event lifecycle tests from classifier accuracy.
CompositionEsn fixed_model(CompositionTask task, Composition label, std::size_t capacity = 32) {
    CompositionOptions options;
    options.task = task;
    options.duration_seconds = 0.2; options.live_stride_seconds = 0.05;
    options.minimum_clips_per_class = 1; options.maximum_samples = capacity;
    CompositionEsn model(options);
    CompositionWeights weights; weights.bias.fill(1);
    weights.readout[composition_index(options.task, label)] = 2;
    model.restore_weights(weights);
    CompositionClip samples;
    constexpr int kSamples = 5;
    for (int i = 0; i < kSamples; ++i) samples.push_back(*composition_sample(frame(), i*0.05,640,480));
    if (capacity < samples.size()) { samples = {samples.front(), samples.back()}; }
    for (auto item : composition_labels(options.task)) model.add_training(samples, item);
    model.restore_trained_readout(weights);
    return model;
}
void start_grasp(GestureInteraction& runtime) {
    auto model = fixed_model(CompositionTask::Interaction, Composition::Grasp);
    (void)runtime.configure_sequences({&model}, 0);
    int grasps = 0;
    for (int i = 0; i <= 12; ++i)
        for (const auto& event : runtime.process(frame(Gesture::ClosedFist), {i*0.05,640,480}))
            if (event.kind == EventKind::Grasp) ++grasps;
    check_true(grasps == 1); check_true(runtime.grasping());
}
}
spec("Gesture interaction events") {
    it("recognizes all seven basic gestures without ESN and deduplicates held poses") {
        for (std::size_t i = 1; i < kfcore::mediapipe::kGestureCount; ++i) {
            GestureInteraction runtime;
            start_basic(runtime, Gesture(i));
            check_true(runtime.process(frame(Gesture(i)), {0.3,640,480}).empty());
            const auto events = runtime.reset(0.3);
            check_true(events.size() == 1);
            check_true(events[0].kind == EventKind::GestureCancelled && events[0].gesture == Gesture(i));
        }
    }
    it("requires continuous confidence and score margin instead of trusting the label") {
        GestureInteraction runtime;
        auto ambiguous = frame(); ambiguous.gestures[0].scores[std::size_t(Gesture::ClosedFist)] = 0.9F;
        for (int i = 0; i < 6; ++i) check_true(runtime.process(ambiguous, {i*0.1,640,480}).empty());
        auto inconsistent = frame(Gesture::Victory); inconsistent.gestures[0].label = Gesture::ClosedFist;
        check_true(runtime.process(inconsistent, {0.6,640,480}).empty());
        check_true(runtime.process(frame(Gesture::None), {0.7,640,480}).empty());
        check_true(runtime.process(inconsistent, {0.8,640,480}).empty());
        const auto events = runtime.process(inconsistent, {1.0,640,480});
        check_true(events.size() == 1 && events[0].gesture == Gesture::Victory);
    }
    it("ends unrecognized poses but never publishes None as a basic gesture") {
        GestureInteraction runtime; start_basic(runtime);
        check_true(runtime.process(frame(Gesture::None), {0.3,640,480}).empty());
        const auto events = runtime.process(frame(Gesture::None), {0.41,640,480});
        check_true(events.size() == 1 && events[0].kind == EventKind::GestureEnded);
        check_true(events[0].reason == EventReason::Unrecognized);
        check_true(runtime.process(frame(Gesture::None), {0.5,640,480}).empty());
    }
    it("ends the previous pose before starting the newly confirmed one") {
        GestureInteraction runtime; start_basic(runtime);
        (void)runtime.process(frame(Gesture::Victory), {0.25,640,480});
        const auto events = runtime.process(frame(Gesture::Victory), {0.41,640,480});
        check_true(events.size() == 2);
        check_true(events[0].kind == EventKind::GestureEnded);
        check_true(events[1].kind == EventKind::GestureStarted && events[1].gesture == Gesture::Victory);
    }
    it("tolerates short missing input without duplicate starts and cancels on watchdog timeout") {
        GestureInteraction runtime; start_basic(runtime);
        check_true(runtime.process({}, {0.25,640,480}).empty());
        check_true(runtime.process(frame(), {0.3,640,480}).empty());
        check_true(runtime.advance(0.5).empty());
        const auto events = runtime.advance(0.56);
        check_true(events.size() == 1 && events[0].kind == EventKind::GestureCancelled);
        check_true(events[0].reason == EventReason::HandLost);
        check_true(runtime.advance(0.7).empty());
    }
    it("does not accumulate confirmation time across a missing frame") {
        GestureInteraction runtime;
        (void)runtime.process(frame(), {0,640,480});
        (void)runtime.process({}, {0.1,640,480});
        check_true(runtime.process(frame(), {0.2,640,480}).empty());
        check_true(runtime.process(frame(), {0.3,640,480}).empty());
        check_true(has(runtime.process(frame(), {0.4,640,480}), EventKind::GestureStarted));
    }
    it("preserves 64 bit identities and cancels before accepting another source") {
        GestureInteraction runtime;
        constexpr std::uint64_t kLargeId = (std::uint64_t{1} << 32) + 1;
        (void)runtime.process(frame(), {0,640,480,kLargeId});
        const auto started = runtime.process(frame(), {0.2,640,480,kLargeId});
        check_true(started[0].source_id == kLargeId);
        const auto changed = runtime.process(frame(), {0.3,640,480,1});
        check_true(changed.size() == 1 && changed[0].reason == EventReason::SourceChanged);
        check_true(changed[0].source_id == kLargeId);
        const auto next = runtime.process(frame(), {0.5,640,480,1});
        check_true(next[0].continuity_id > started[0].continuity_id);
    }
    it("tolerates a brief handedness probability flip but rejects sustained source changes") {
        GestureInteraction runtime; start_basic(runtime);
        check_true(runtime.process(frame(Gesture::OpenPalm,0.05F), {0.25,640,480}).empty());
        check_true(runtime.process(frame(), {0.3,640,480}).empty());
        (void)runtime.process(frame(Gesture::OpenPalm,0.05F), {0.35,640,480});
        const auto events = runtime.process(frame(Gesture::OpenPalm,0.05F), {0.51,640,480});
        check_true(has(events, EventKind::GestureCancelled));
        check_true(events[0].reason == EventReason::SourceChanged);
    }
    it("cancels active grasp once when hands vanish even without further camera frames") {
        GestureInteraction runtime; start_grasp(runtime);
        const auto events = runtime.advance(0.9);
        check_true(has(events, EventKind::GraspCancelled)); check_false(runtime.grasping());
        check_false(has(events, EventKind::Release));
        check_true(runtime.advance(1.0).empty());
    }
    it("cancels grasp on reset multi hand input frame gaps and model replacement") {
        for (int scenario = 0; scenario < 4; ++scenario) {
            GestureInteraction runtime; start_grasp(runtime);
            std::vector<Event> events;
            if (scenario == 0) events = runtime.reset(0.7);
            else if (scenario == 1) {
                auto multiple = frame(); multiple.gestures.push_back(multiple.gestures[0]);
                multiple.landmarks.hands.push_back(multiple.landmarks.hands[0]);
                events = runtime.process(multiple, {0.7,640,480});
            } else if (scenario == 2) events = runtime.process(frame(), {1.0,640,480});
            else events = runtime.configure_sequences({}, 0.7);
            check_true(has(events, EventKind::GraspCancelled)); check_false(runtime.grasping());
        }
    }
    it("rejects invalid input and model configuration without changing an active grasp") {
        GestureInteraction runtime; start_grasp(runtime);
        auto bad = frame(); bad.gestures[0].scores[0] = std::numeric_limits<float>::quiet_NaN();
        check_throws(runtime.process(bad, {0.7,640,480}));
        check_throws(runtime.process(frame(), {0.5,640,480}));
        check_throws(runtime.process(frame(), {0.7,0,480}));
        check_throws(runtime.process(frame(), {0.7,640,480,0}));
        CompositionOptions options; options.task = CompositionTask::Motion;
        CompositionEsn untrained(options);
        check_throws(runtime.configure_sequences({nullptr,&untrained}, 0.7));
        check_true(runtime.grasping());
        check_true(runtime.process(frame(), {0.7,640,480}).empty());
    }
    it("releases after a short fist without a prior grasp and keeps wave independent") {
        auto release = fixed_model(CompositionTask::Interaction, Composition::Release);
        auto wave = fixed_model(CompositionTask::Motion, Composition::Wave);
        GestureInteraction runtime;
        (void)runtime.configure_sequences({&release,&wave}, 0);
        int releases = 0, waves = 0, basics = 0;
        for (int i = 0; i <= 16; ++i) for (const auto& event : runtime.process(
                frame(i <= 4 ? Gesture::ClosedFist : Gesture::OpenPalm), {i*0.05,640,480})) {
            if (event.kind == EventKind::Release) ++releases;
            if (event.kind == EventKind::Wave) ++waves;
            if (event.kind == EventKind::GestureStarted) ++basics;
        }
        check_true(releases == 1 && waves == 1 && basics == 2); check_false(runtime.grasping());
        check_false(has(runtime.reset(0.8), EventKind::GraspCancelled));
    }
    it("requires a trained interaction model and confirmed held fist before grasp") {
        GestureInteraction runtime;
        for (int i = 0; i <= 12; ++i)
            check_false(has(runtime.process(frame(Gesture::ClosedFist), {i*0.05,640,480}), EventKind::Grasp));
        check_false(runtime.grasping());
        auto model = fixed_model(CompositionTask::Interaction, Composition::Grasp);
        (void)runtime.configure_sequences({&model}, 0.65);
        int grasps = 0;
        for (int i = 13; i <= 30; ++i)
            for (const auto& event : runtime.process(frame(Gesture::ClosedFist), {i*0.05,640,480}))
                if (event.kind == EventKind::Grasp) ++grasps;
        check_true(grasps == 1 && runtime.grasping());
    }
    it("does not turn a constant release classifier into an isolated palm event") {
        auto model = fixed_model(CompositionTask::Interaction, Composition::Release);
        GestureInteraction runtime;
        (void)runtime.configure_sequences({&model}, 0);
        for (int i = 0; i <= 20; ++i)
            check_false(has(runtime.process(frame(Gesture::OpenPalm), {i*0.05,640,480}), EventKind::Release));
    }
    it("does not continue fist hold through an opening transition") {
        GestureInteraction runtime;
        auto model = fixed_model(CompositionTask::Interaction, Composition::Grasp);
        (void)runtime.configure_sequences({&model}, 0);
        for (int i = 0; i <= 4; ++i)
            check_false(has(runtime.process(frame(Gesture::ClosedFist), {i*0.05,640,480}), EventKind::Grasp));
        for (int i = 5; i <= 8; ++i)
            check_false(has(runtime.process(frame(Gesture::OpenPalm), {i*0.05,640,480}), EventKind::Grasp));
        check_false(runtime.grasping());
    }
    it("cancels a held grasp when opening evidence expires") {
        GestureInteraction runtime;
        start_grasp(runtime);
        int cancellations = 0, releases = 0;
        for (int i = 13; i <= 40; ++i)
            for (const auto& event : runtime.process(frame(Gesture::Victory), {i*0.05,640,480})) {
                if (event.kind == EventKind::GraspCancelled) ++cancellations;
                if (event.kind == EventKind::Release) ++releases;
            }
        for (int i = 41; i <= 45; ++i)
            for (const auto& event : runtime.process(frame(Gesture::OpenPalm), {i*0.05,640,480})) {
                if (event.kind == EventKind::GraspCancelled) ++cancellations;
                if (event.kind == EventKind::Release) ++releases;
            }
        check_true(cancellations == 1 && releases == 0);
        check_false(runtime.grasping());
    }
    it("cancels a held grasp if opening follows a missing hand frame") {
        GestureInteraction runtime;
        start_grasp(runtime);
        (void)runtime.process({}, {0.65,640,480});
        int cancellations = 0, releases = 0;
        for (int i = 14; i <= 18; ++i)
            for (const auto& event : runtime.process(frame(Gesture::OpenPalm), {i*0.05,640,480})) {
                if (event.kind == EventKind::GraspCancelled) ++cancellations;
                if (event.kind == EventKind::Release) ++releases;
            }
        check_true(cancellations == 1 && releases == 0);
        check_false(runtime.grasping());
    }
    it("owns Wave model snapshots and bounds history without continuing partial evidence") {
        auto model = fixed_model(CompositionTask::Motion, Composition::Wave, 2);
        GestureInteraction runtime;
        (void)runtime.configure_sequences({nullptr,&model}, 0);
        model = CompositionEsn{};
        (void)runtime.process(frame(), {0,640,480});
        (void)runtime.process(frame(), {0.05,640,480});
        check_true(runtime.process(frame(), {0.1,640,480}).empty());
        check_false(runtime.grasping());
        check_true(runtime.reset(0.1).empty());
    }
}
