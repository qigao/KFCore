#include "../composition_session.hpp"
#include <kfcore/gesture_interaction/interaction.hpp>
#include <tinytest.hpp>
#include <cmath>
#include <limits>

namespace {
using Gesture = kfcore::mediapipe::CannedGesture;
kfcore::gesture_interaction::CompositionClip sequence(int label, int disturbance = -1) {
    constexpr int kSamples = 61, kPhaseSamples = 20;
    constexpr double kPeriod = 0.05;
    kfcore::gesture_interaction::CompositionClip result;
    for (int i = 0; i < kSamples; ++i) {
        const bool middle = i >= kPhaseSamples && i < 2*kPhaseSamples;
        Gesture gesture = Gesture::OpenPalm;
        if (label == 1) gesture = middle ? Gesture::ClosedFist : Gesture::OpenPalm;
        if (label == 2) gesture = middle ? Gesture::OpenPalm : Gesture::ClosedFist;
        if (i == disturbance) gesture = Gesture::PointingUp;
        kfcore::gesture_interaction::CompositionSample sample{i*kPeriod};
        sample.values[std::size_t(gesture)] = 1;
        sample.values[kfcore::gesture_interaction::kCompositionInputs-2] = 0.9F;
        sample.values.back() = 1;
        result.push_back(sample);
    }
    return result;
}
kfcore::mediapipe::GestureFrame frame(const kfcore::gesture_interaction::CompositionSample& sample) {
    kfcore::mediapipe::GestureFrame result;
    result.landmarks.hands.resize(1); result.gestures.resize(1);
    auto& hand = result.landmarks.hands.front();
    hand.handedness = kfcore::hand_models::Handedness::Right;
    hand.right_hand_probability = sample.values[kfcore::gesture_interaction::kCompositionInputs-2];
    std::copy_n(sample.values.begin(), kfcore::mediapipe::kGestureCount, result.gestures[0].scores.begin());
    result.gestures[0].label = Gesture(std::max_element(result.gestures[0].scores.begin(),
        result.gestures[0].scores.end()) - result.gestures[0].scores.begin());
    return result;
}
kfcore::gesture_interaction::CompositionClip with_transitions(int label, int radius) {
    auto result = sequence(label);
    constexpr std::array<int, 2> kBoundaries{20, 40};
    constexpr float kNoneScore = 0.8F, kPoseScore = 1.0F-kNoneScore;
    for (int boundary : kBoundaries) {
        for (int i = boundary-radius; i <= boundary+radius; ++i) {
            auto& values = result.at(std::size_t(i)).values;
            for (std::size_t gesture = 0; gesture < kfcore::mediapipe::kGestureCount; ++gesture)
                values[gesture] *= kPoseScore;
            values[std::size_t(Gesture::None)] = kNoneScore;
        }
    }
    return result;
}
void record(preview::CompositionSession& session, int key, double start) {
    session.key(key);
    for (const auto& sample : sequence((key-'0')%kfcore::gesture_interaction::kCompositionClasses))
        session.update(frame(sample), start+sample.seconds);
}
kfcore::gesture_interaction::CompositionClip action_sequence(kfcore::gesture_interaction::Composition label, float variation = 0, float direction = 1) {
    auto result = sequence(0);
    constexpr float kPi = 3.14159265358979323846F;
    for (std::size_t i = 0; i < result.size(); ++i) {
        auto& s = result[i];
        const float phase = float(s.seconds/3.0);
        if (label == kfcore::gesture_interaction::Composition::Grasp || label == kfcore::gesture_interaction::Composition::Release) {
            const bool after = phase > 0.5F+variation;
            const bool closed = label == kfcore::gesture_interaction::Composition::Grasp ? after : !after;
            s.values[std::size_t(Gesture::OpenPalm)] = closed ? 0 : 1;
            s.values[std::size_t(Gesture::ClosedFist)] = closed ? 1 : 0;
        }
        float x = 0.5F, y = 0.5F;
        if (label == kfcore::gesture_interaction::Composition::Wave) x += direction*(0.18F+variation)*std::sin(4*kPi*phase);
        if (label == kfcore::gesture_interaction::Composition::None) y += variation*phase;
        s.wrist = std::array<float,2>{x,y};
    }
    return result;
}
kfcore::gesture_interaction::CompositionClip single_sweep(float direction = 1, float amplitude = 0.4F) {
    auto clip = action_sequence(kfcore::gesture_interaction::Composition::None);
    for (auto& sample : clip)
        (*sample.wrist)[0] += direction*amplitude*(float(sample.seconds/3.0)-0.5F);
    return clip;
}
}
spec("ESN gesture composition") {
    it("requires training and all composition classes") {
        kfcore::gesture_interaction::CompositionEsn model;
        check_throws_with(model.predict(sequence(1)), "not trained");
        model.add_training(sequence(0), kfcore::gesture_interaction::Composition::None);
        check_throws_with(model.train_readout(), "training clips per class");
    }
    it("uses history when final gestures are identical") {
        kfcore::gesture_interaction::CompositionEsn model;
        const auto constant = model.encode(sequence(0));
        const auto combination = model.encode(sequence(1));
        check_false(constant == combination);
        check_true(combination == model.encode(sequence(1)));
    }
    it("learns complete combinations including noisy observations") {
        kfcore::gesture_interaction::CompositionEsn model;
        constexpr std::array<int, 3> training_errors{-1, 10, 30};
        for (int label = 0; label < kfcore::gesture_interaction::kCompositionClasses; ++label)
            for (int error : training_errors)
                model.add_training(sequence(label,error), kfcore::gesture_interaction::Composition(label));
        model.train_readout();
        for (int label = 0; label < kfcore::gesture_interaction::kCompositionClasses; ++label)
            check_true(model.predict(sequence(label, 15)).label == kfcore::gesture_interaction::Composition(label));
        model.add_training(sequence(0), kfcore::gesture_interaction::Composition::None);
        check_false(model.trained());
    }
    it("rejects invalid timestamps scores and capacity") {
        kfcore::gesture_interaction::CompositionOptions options;
        options.minimum_clips_per_class = 1; options.maximum_clips = 3;
        kfcore::gesture_interaction::CompositionEsn model(options);
        auto invalid = sequence(1);
        invalid[1].seconds = invalid[0].seconds;
        check_throws_as(model.encode(invalid), std::invalid_argument);
        invalid = sequence(1); invalid[2].values[0] = std::numeric_limits<float>::quiet_NaN();
        check_throws_as(model.encode(invalid), std::invalid_argument);
        for (int i = 0; i < 3; ++i) model.add_training(sequence(0), kfcore::gesture_interaction::Composition::None);
        check_throws_with(model.add_training(sequence(0), kfcore::gesture_interaction::Composition::None), "capacity");
    }
    it("distinguishes missing observations from the official None category") {
        const auto missing = kfcore::gesture_interaction::composition_sample({}, 0);
        check_true(missing.has_value() && missing->values.back() == 0);
        auto none_frame = frame(sequence(0)[0]);
        none_frame.gestures[0].scores.fill(0); none_frame.gestures[0].scores[0] = 1;
        const auto none = kfcore::gesture_interaction::composition_sample(none_frame, 0);
        check_true(none->values[0] == 1 && none->values.back() == 1);
        none_frame.landmarks.hands.resize(2); none_frame.gestures.resize(2);
        check_false(kfcore::gesture_interaction::composition_sample(none_frame, 0).has_value());
    }
    it("keeps actionable cancellation feedback after detection recovers") {
        preview::CompositionSession session({});
        session.key('1');
        session.update(frame(sequence(1)[0]), 0);
        session.update({}, 0.1);
        check_true(session.status.find("short gap") != std::string::npos);
        session.update({}, 0.2);
        session.update({}, 0.3);
        check_true(session.feedback.find("Not saved: hand lost") == 0);
        check_true(session.feedback.find("press 1 to retry") != std::string::npos);
        const auto message = session.feedback;
        session.update(frame(sequence(1)[0]), 0.4);
        check_true(session.status.find("Open_Palm") != std::string::npos);
        check_true(session.feedback == message);
    }
    it("displays unrecognized gestures separately from missing hands and no combination") {
        preview::CompositionSession session({});
        auto observation = frame(sequence(0).front());
        observation.gestures.front().label = Gesture::None;
        observation.gestures.front().scores.fill(0);
        observation.gestures.front().scores[std::size_t(Gesture::None)] = 1;
        session.update(observation, 0);
        check_true(session.status.find("1 hand Right | Unrecognized (None)") == 0);
        check_true(session.status.find(preview::kNoHandStatus) == std::string::npos);
        session.update({}, 0.1);
        check_true(session.status.find(preview::kNoHandStatus) == 0);
        check_true(session.status.find("Unrecognized") == std::string::npos);
        check_true(std::string(kfcore::gesture_interaction::composition_name(kfcore::gesture_interaction::Composition::None)) == "No-combo");
        check_true(std::string(kfcore::mediapipe::gesture_name(Gesture::None)) == "None");
        check_true(std::string(preview::gesture_display_name(Gesture::OpenPalm)) == "Open_Palm");
    }
    it("preserves the complete score vector for a detected None transition") {
        constexpr int kBoundary = 20, kRadius = 4;
        const auto sample = with_transitions(1, kRadius)[kBoundary];
        const auto observation = frame(sample);
        check_true(observation.gestures.front().label == Gesture::None);
        const auto encoded = kfcore::gesture_interaction::composition_sample(observation, sample.seconds);
        check_true(encoded.has_value());
        check_true(encoded->values == sample.values);
        check_true(encoded->seconds == sample.seconds);
    }
    it("records None transitions longer than the lost-hand timeout without cancellation") {
        preview::CompositionSession session({});
        session.key('1');
        constexpr int kRadius = 4;
        // Nine detected None frames span 400ms, longer than the 250ms loss limit.
        for (const auto& sample : with_transitions(1, kRadius)) {
            const auto observation = frame(sample);
            session.update(observation, sample.seconds);
            if (observation.gestures.front().label == Gesture::None) {
                check_true(session.status.find("None") != std::string::npos);
                check_true(session.status.find("REC") != std::string::npos);
            }
            check_true(session.feedback.find("Not saved") == std::string::npos);
        }
        check_true(session.summary().find("No-combo/OCO/COC 0/1/0") != std::string::npos);
        check_true(session.feedback.find("Training clip saved") == 0);
    }
    it("learns combinations with None transitions at an unseen transition duration") {
        kfcore::gesture_interaction::CompositionEsn model;
        constexpr std::array<int, 3> kTrainingRadii{2, 3, 5};
        constexpr int kHeldoutRadius = 4;
        for (int label = 0; label < kfcore::gesture_interaction::kCompositionClasses; ++label)
            for (int radius : kTrainingRadii)
                model.add_training(with_transitions(label, radius), kfcore::gesture_interaction::Composition(label));
        model.train_readout();
        for (int label = 0; label < kfcore::gesture_interaction::kCompositionClasses; ++label)
            check_true(model.predict(with_transitions(label, kHeldoutRadius)).label == kfcore::gesture_interaction::Composition(label));
    }
    it("accepts upper case keys and derives counts from recorded sequences") {
        kfcore::gesture_interaction::CompositionOptions options; options.minimum_clips_per_class = 1;
        preview::CompositionSession session(options);
        session.key('T');
        check_true(session.feedback.find("training clips per class") != std::string::npos);
        for (int label = 0; label < kfcore::gesture_interaction::kCompositionClasses; ++label) record(session,'0'+label,label*4.0);
        check_true(session.summary().find("No-combo/OCO/COC 1/1/1") != std::string::npos);
        session.key('T');
        check_true(session.summary().find("TRAINED |") == 0);
        for (int label = 0; label < kfcore::gesture_interaction::kCompositionClasses; ++label) record(session,'3'+label,16+label*4.0);
        session.key('V');
        check_true(session.feedback.find("Held-out correct=3/3") == 0);
    }
}

spec("Parallel gesture actions") {
    it("waits for countdown then records a complete action without counting preparation") {
        kfcore::gesture_interaction::CompositionOptions options; options.task = kfcore::gesture_interaction::CompositionTask::Interaction;
        preview::CompositionSession session(options);
        session.key('1');
        const auto clip = action_sequence(kfcore::gesture_interaction::Composition::Grasp);
        session.update(frame(clip.front()), 0);
        session.update(frame(clip.front()), 1);
        check_true(session.status.find("READY Grasp") != std::string::npos);
        check_true(session.summary().find("0/0/0") != std::string::npos);
        for (const auto& s : clip) session.update(frame(s), s.seconds+2);
        check_true(session.summary().find("Neutral/Grasp/Release 0/1/0") != std::string::npos);
    }
    it("produces one live grasp event and does not repeat while holding the fist") {
        kfcore::gesture_interaction::CompositionOptions options; options.task = kfcore::gesture_interaction::CompositionTask::Interaction;
        options.record_countdown_seconds = 0;
        preview::CompositionSession session(options);
        double start = 0;
        for (auto label : kfcore::gesture_interaction::composition_labels(options.task))
            for (float variation : {-0.1F, 0.0F, 0.1F}) {
                session.key('0'+kfcore::gesture_interaction::composition_index(options.task, label));
                for (const auto& s : action_sequence(label, variation)) session.update(frame(s), start+s.seconds);
                start += 4;
            }
        const auto closed = action_sequence(kfcore::gesture_interaction::Composition::Grasp).back();
        session.key('0');
        for (const auto& s : sequence(0)) session.update(frame(closed), start+s.seconds);
        session.key('t');
        start += 4;
        kfcore::gesture_interaction::GestureInteraction runtime;
        (void)runtime.configure_sequences({&session.model()}, start);
        std::size_t grasp_events = 0;
        const auto observe = [&](const auto& sample, double seconds) {
            for (const auto& event : runtime.process(frame(sample), {seconds, 640, 480}))
                if (event.kind == kfcore::gesture_interaction::EventKind::Grasp) ++grasp_events;
        };
        for (const auto& s : action_sequence(kfcore::gesture_interaction::Composition::Grasp, 0.04F)) observe(s, start+s.seconds);
        for (int i = 1; i <= 120; ++i) observe(closed, start+3+i*0.05);
        check_true(grasp_events == 1);
        check_true(runtime.grasping());
        const auto cancelled = runtime.advance(start+10);
        check_true(std::any_of(cancelled.begin(), cancelled.end(), [](const auto& event) {
            return event.kind == kfcore::gesture_interaction::EventKind::GraspCancelled;
        }));
        check_false(runtime.grasping());
    }
    it("routes a trained open fist open sequence into matched grasp and release events") {
        using namespace kfcore::gesture_interaction;
        CompositionOptions options; options.task = CompositionTask::Interaction;
        CompositionEsn model(options);
        for (auto label : composition_labels(options.task))
            for (float variation : {-0.1F, 0.0F, 0.1F}) model.add_training(action_sequence(label, variation), label);
        auto fist = action_sequence(Composition::Grasp);
        for (auto& sample : fist) {
            sample.values[std::size_t(Gesture::OpenPalm)] = 0;
            sample.values[std::size_t(Gesture::ClosedFist)] = 1;
        }
        model.add_training(fist, Composition::None); model.train_readout();
        GestureInteraction runtime; (void)runtime.configure_sequences({&model}, 0);
        std::vector<EventKind> actions;
        const auto feed = [&](const CompositionSample& sample, double time) {
            for (const auto& event : runtime.process(frame(sample), {time,640,480}))
                if (event.kind == EventKind::Grasp || event.kind == EventKind::Release || event.kind == EventKind::GraspCancelled)
                    actions.push_back(event.kind);
        };
        for (const auto& sample : action_sequence(Composition::Grasp)) feed(sample, sample.seconds);
        for (int i = 1; i <= 40; ++i) feed(fist.back(), 3+i*0.05);
        for (const auto& sample : action_sequence(Composition::Release)) feed(sample, 5.05+sample.seconds);
        const auto open = action_sequence(Composition::Release).back();
        for (int i = 1; i <= 40; ++i) feed(open, 8.05+i*0.05);
        check_true(actions.size() == 2);
        check_true(actions.front() == EventKind::Grasp && actions.back() == EventKind::Release);
        check_false(runtime.grasping());
    }
    it("learns grasp and release separately from static poses") {
        kfcore::gesture_interaction::CompositionOptions options; options.task = kfcore::gesture_interaction::CompositionTask::Interaction;
        kfcore::gesture_interaction::CompositionEsn model(options);
        for (auto label : kfcore::gesture_interaction::composition_labels(options.task))
            for (float variation : {-0.1F, 0.0F, 0.1F})
                model.add_training(action_sequence(label, variation), label);
        auto fist = action_sequence(kfcore::gesture_interaction::Composition::None);
        for (auto& s : fist) {
            s.values[std::size_t(Gesture::OpenPalm)] = 0;
            s.values[std::size_t(Gesture::ClosedFist)] = 1;
        }
        model.add_training(fist, kfcore::gesture_interaction::Composition::None);
        model.train_readout();
        check_true(model.predict(fist).label == kfcore::gesture_interaction::Composition::None);
        for (auto label : kfcore::gesture_interaction::composition_labels(options.task))
            check_true(model.predict(action_sequence(label, 0.04F)).label == label);
        check_throws_with(model.add_training(action_sequence(kfcore::gesture_interaction::Composition::Wave), kfcore::gesture_interaction::Composition::Wave), "different task");
    }
    it("learns binary wave recognition and treats single sweeps as neutral") {
        kfcore::gesture_interaction::CompositionOptions options; options.task = kfcore::gesture_interaction::CompositionTask::Motion;
        kfcore::gesture_interaction::CompositionEsn model(options);
        for (auto label : kfcore::gesture_interaction::composition_labels(options.task))
            for (float variation : {-0.05F, 0.0F, 0.05F})
                for (float direction : {-1.0F, 1.0F})
                    model.add_training(action_sequence(label, variation, direction), label);
        for (float amplitude : {0.3F, 0.4F, 0.5F})
            for (float direction : {-1.0F, 1.0F})
                model.add_training(single_sweep(direction, amplitude), kfcore::gesture_interaction::Composition::None);
        model.train_readout();
        check_true(model.class_count() == 2);
        for (float direction : {-1.0F, 1.0F}) {
            const auto prediction = model.predict(single_sweep(direction, 0.35F));
            check_true(prediction.class_count == 2);
            check_true(prediction.label == kfcore::gesture_interaction::Composition::None);
        }
        for (auto label : kfcore::gesture_interaction::composition_labels(options.task))
            for (float direction : {-1.0F, 1.0F}) {
                const auto predicted = model.predict(action_sequence(label, 0.025F, direction));
                info("expected=%s actual=%s direction=%f", kfcore::gesture_interaction::composition_name(label), kfcore::gesture_interaction::composition_name(predicted.label), direction);
                check_true(predicted.label == label);
            }
    }
    it("normalizes wrist coordinates and rejects missing or nonfinite motion geometry") {
        auto observation = frame(sequence(0).front());
        observation.landmarks.hands[0].landmarks[0].x = 320;
        observation.landmarks.hands[0].landmarks[0].y = 240;
        const auto small = kfcore::gesture_interaction::composition_sample(observation, 0, 640, 480);
        observation.landmarks.hands[0].landmarks[0].x *= 2;
        observation.landmarks.hands[0].landmarks[0].y *= 2;
        const auto large = kfcore::gesture_interaction::composition_sample(observation, 0, 1280, 960);
        check_true(small->wrist == large->wrist);
        kfcore::gesture_interaction::CompositionOptions options; options.task = kfcore::gesture_interaction::CompositionTask::Motion;
        kfcore::gesture_interaction::CompositionEsn model(options);
        check_throws_with(model.encode(sequence(0)), "wrist");
        auto clip = single_sweep();
        (*clip[3].wrist)[0] = std::numeric_limits<float>::quiet_NaN();
        check_throws_with(model.encode(clip), "non-finite");
        check_throws_with(kfcore::gesture_interaction::composition_sample(observation, 0, 640, 0), "geometry");
    }
    it("keeps motion features translation invariant and independent of prior clips") {
        kfcore::gesture_interaction::CompositionOptions options; options.task = kfcore::gesture_interaction::CompositionTask::Motion;
        kfcore::gesture_interaction::CompositionEsn model(options);
        auto clip = action_sequence(kfcore::gesture_interaction::Composition::Wave);
        const auto original = model.encode(clip);
        for (auto& sample : clip) { (*sample.wrist)[0] += 0.125F; (*sample.wrist)[1] -= 0.125F; }
        const auto shifted = model.encode(clip);
        for (std::size_t i = 0; i < original.size(); ++i)
            check_true(std::abs(original[i]-shifted[i]) < 0.00001F);
        (void)model.encode(single_sweep());
        check_true(model.encode(action_sequence(kfcore::gesture_interaction::Composition::Wave)) == original);
    }
    it("confirms events once and rearms only after sustained neutral evidence") {
        kfcore::gesture_interaction::ActionGate gate;
        const kfcore::gesture_interaction::CompositionPrediction action{kfcore::gesture_interaction::Composition::Grasp, {0,1,0}};
        const kfcore::gesture_interaction::CompositionPrediction neutral{kfcore::gesture_interaction::Composition::None, {1,0,0}};
        const kfcore::gesture_interaction::CompositionPrediction uncertain{kfcore::gesture_interaction::Composition::None, {0.4F,0.3F,0.3F}};
        check_false(gate.update(action, 0).has_value());
        check_true(gate.update(action, 0.2) == kfcore::gesture_interaction::Composition::Grasp);
        check_false(gate.update(action, 0.4).has_value());
        gate.interrupt();
        check_false(gate.update(uncertain, 0.5).has_value());
        check_false(gate.update(action, 0.6).has_value());
        check_false(gate.update(action, 0.8).has_value());
        check_false(gate.update(neutral, 1).has_value());
        check_false(gate.update(neutral, 1.4).has_value());
        check_false(gate.update(action, 1.5).has_value());
        check_true(gate.update(action, 1.7) == kfcore::gesture_interaction::Composition::Grasp);
        gate.reset();
        check_false(gate.update(action, 2).has_value());
        check_throws_with(gate.update(action, 2), "timestamp");
    }
    it("keeps recording intact on repeated keys and isolates branch samples") {
        kfcore::gesture_interaction::CompositionOptions options; options.task = kfcore::gesture_interaction::CompositionTask::Interaction;
        options.record_countdown_seconds = 0;
        preview::CompositionSession interaction(options);
        options.task = kfcore::gesture_interaction::CompositionTask::Motion;
        preview::CompositionSession motion(options);
        interaction.key('1');
        const auto clip = action_sequence(kfcore::gesture_interaction::Composition::Grasp);
        for (std::size_t i = 0; i < clip.size(); ++i) {
            if (i == 20) interaction.key('1');
            interaction.update(frame(clip[i]), clip[i].seconds, 640, 480);
        }
        check_true(interaction.summary().find("Neutral/Grasp/Release 0/1/0") != std::string::npos);
        check_true(motion.summary().find("Neutral/Wave 0/0 (need 3 each) | test 0/0") != std::string::npos);
        check_false(interaction.recording());
    }
    it("trains and evaluates wave with only two classes and rejects removed keys") {
        kfcore::gesture_interaction::CompositionOptions options; options.task = kfcore::gesture_interaction::CompositionTask::Motion;
        options.record_countdown_seconds = 0;
        options.minimum_clips_per_class = 1;
        options.maximum_clips = 2;
        preview::CompositionSession session(options);
        session.key('2');
        check_false(session.recording());
        check_true(session.feedback.find("2/5 unused") != std::string::npos);
        session.key('5');
        check_false(session.recording());
        double start = 0;
        for (int key : {'0', '1', '3', '4'}) {
            const int index = (key-'0')%kfcore::gesture_interaction::kCompositionClasses;
            session.key(key);
            for (const auto& sample : action_sequence(kfcore::gesture_interaction::composition_labels(options.task)[index])) {
                auto observation = frame(sample);
                observation.landmarks.hands[0].landmarks[0].x = (*sample.wrist)[0]*640;
                observation.landmarks.hands[0].landmarks[0].y = (*sample.wrist)[1]*480;
                session.update(observation, start+sample.seconds, 640, 480);
            }
            start += 4;
            if (key == '1') session.key('t');
        }
        check_true(session.summary().find("TRAINED | train Neutral/Wave 1/1 (need 1 each) | test 1/1") == 0);
        session.key('v');
        check_true(session.feedback.find("Held-out correct=2/2") == 0);
    }
    it("ignores inactive score storage when gating binary wave predictions") {
        kfcore::gesture_interaction::ActionGate gate;
        const kfcore::gesture_interaction::CompositionPrediction wave{kfcore::gesture_interaction::Composition::Wave, {0,1,100}, 2};
        check_false(gate.update(wave, 0).has_value());
        check_true(gate.update(wave, 0.2) == kfcore::gesture_interaction::Composition::Wave);
        check_false(gate.update(wave, 0.4).has_value());
        auto invalid = wave; invalid.class_count = 1;
        check_throws_with(gate.update(invalid, 0.6), "class count");
    }
}
