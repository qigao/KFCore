#include "../composition_session.hpp"
#include <tinytest.hpp>
#include <salts/crypto.h>
#include <filesystem>
#include <limits>

namespace {
constexpr char kPipeline[] = "test-only-pipeline-identity";
kfcore::gesture_interaction::CompositionClip clip(kfcore::gesture_interaction::Composition label) {
    kfcore::gesture_interaction::CompositionClip result;
    constexpr int kSamples = 61;
    constexpr float kPi = 3.14159265358979323846F;
    using Gesture = kfcore::mediapipe::CannedGesture;
    for (int i = 0; i < kSamples; ++i) {
        kfcore::gesture_interaction::CompositionSample sample{i*0.05};
        const bool closed = label == kfcore::gesture_interaction::Composition::Grasp ? i > 30 :
            label == kfcore::gesture_interaction::Composition::Release && i <= 30;
        sample.values[std::size_t(closed ? Gesture::ClosedFist : Gesture::OpenPalm)] = 1;
        sample.values[kfcore::gesture_interaction::kCompositionInputs-2] = 0.9F; sample.values.back() = 1;
        sample.wrist = std::array<float,2>{0.5F+(label == kfcore::gesture_interaction::Composition::Wave ? 0.2F*std::sin(4*kPi*i/60) : 0), 0.5F};
        result.push_back(sample);
    }
    return result;
}
kfcore::gesture_interaction::SessionArchive session(kfcore::gesture_interaction::CompositionTask task, bool trained) {
    kfcore::gesture_interaction::SessionArchive result;
    result.options.task = task; result.options.minimum_clips_per_class = 1;
    result.options.record_countdown_seconds = 0;
    kfcore::gesture_interaction::CompositionEsn model(result.options);
    for (auto label : kfcore::gesture_interaction::composition_labels(task)) {
        const auto samples = clip(label);
        model.add_training(samples, label);
        result.training.push_back({samples,label,123,456});
        result.heldout.push_back({samples,label,789,987});
    }
    if (trained) model.train_readout();
    result.weights = model.weights(); result.trained = trained;
    return result;
}
kfcore::gesture_interaction::ExperimentArchive experiment(bool trained = true) {
    return {kPipeline,{session(kfcore::gesture_interaction::CompositionTask::Interaction,trained), session(kfcore::gesture_interaction::CompositionTask::Motion,trained)}};
}
void rehash(std::vector<std::uint8_t>& bytes) {
    constexpr auto kHashBytes = SALTS_CRYPTO_SHA256_DIGEST_SIZE;
    if (salts_crypto_sha256(bytes.data(),bytes.size()-kHashBytes,bytes.data()+bytes.size()-kHashBytes) != SALTS_CRYPTO_OK)
        throw std::runtime_error("test SHA-256 failed");
}
class TempDirectory {
public:
    char* name = tt_make_temp_dir("kfhand-experiment");
    TempDirectory() { if (!name) throw std::runtime_error("test temp directory failed"); }
    ~TempDirectory() { tt_remove_tree(name); free(name); }
    std::filesystem::path file() const { return std::filesystem::path(name)/L"experiment.kfesn"; }
};
}
spec("ESN experiment persistence") {
    it("archives newly recorded training and test clips with their original observations") {
        kfcore::gesture_interaction::CompositionOptions options;
        options.task = kfcore::gesture_interaction::CompositionTask::Interaction;
        options.record_countdown_seconds = 0;
        preview::CompositionSession current(options);
        const auto samples = clip(kfcore::gesture_interaction::Composition::Grasp);
        for (int key : {'1', '4'}) {
            current.key(key);
            for (const auto& sample : samples) {
                kfcore::mediapipe::GestureFrame frame;
                frame.landmarks.hands.resize(1); frame.gestures.resize(1);
                frame.landmarks.hands[0].handedness = kfcore::hand_models::Handedness::Right;
                frame.landmarks.hands[0].right_hand_probability = sample.values[kfcore::gesture_interaction::kCompositionInputs-2];
                std::copy_n(sample.values.begin(), kfcore::mediapipe::kGestureCount, frame.gestures[0].scores.begin());
                current.update(frame, sample.seconds);
            }
            check_false(current.recording());
        }
        const auto saved = current.archive();
        check_true(current.dirty());
        check_true(saved.training.size() == 1 && saved.heldout.size() == 1);
        check_true(saved.training[0].label == kfcore::gesture_interaction::Composition::Grasp);
        check_true(saved.training[0].batch == saved.heldout[0].batch);
        check_true(saved.training[0].recorded_at_ms > 0);
        check_true(saved.training[0].samples.size() == samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) {
            check_true(saved.training[0].samples[i].seconds == samples[i].seconds);
            check_true(saved.training[0].samples[i].values == samples[i].values);
        }
        auto archive = experiment(false); archive.sessions[0] = saved;
        const auto loaded = kfcore::gesture_interaction::decode_experiment(kfcore::gesture_interaction::encode_experiment(archive), kPipeline);
        check_true(loaded.sessions[0].training[0].samples.size() == samples.size());
        check_false(loaded.sessions[0].trained);
    }
    it("round trips raw training heldout metadata options and exact weights") {
        auto original = experiment();
        original.sessions[0].gate.minimum_margin = 0.23F;
        const auto bytes = kfcore::gesture_interaction::encode_experiment(original);
        const auto loaded = kfcore::gesture_interaction::decode_experiment(bytes,kPipeline);
        check_true(kfcore::gesture_interaction::encode_experiment(loaded) == bytes);
        check_true(loaded.sessions.size() == 2);
        for (std::size_t i = 0; i < loaded.sessions.size(); ++i) {
            const auto& s = loaded.sessions[i];
            check_true(s.trained);
            check_true(s.training[0].batch == 123);
            check_true(s.heldout[0].batch == 789);
            check_true(s.training[0].samples[5].values == original.sessions[i].training[0].samples[5].values);
            check_true(s.training[0].samples[5].wrist == original.sessions[i].training[0].samples[5].wrist);
            check_true(s.weights.readout == original.sessions[i].weights.readout);
        }
    }
    it("restores trained predictions without fitting and starts with clean event history") {
        const auto original = experiment();
        const auto loaded = kfcore::gesture_interaction::decode_experiment(kfcore::gesture_interaction::encode_experiment(original),kPipeline);
        for (const auto& saved : loaded.sessions) {
            auto restored = preview::CompositionSession::restore(saved);
            check_true(restored->summary().find("TRAINED |") == 0);
            check_false(restored->dirty()); check_false(restored->recording());
            check_true(restored->prediction == "ESN: collecting history");
            kfcore::gesture_interaction::CompositionEsn model(saved.options);
            model.restore_weights(saved.weights);
            for (const auto& row : saved.training) model.add_training(row.samples,row.label);
            model.restore_trained_readout(saved.weights);
            for (auto label : kfcore::gesture_interaction::composition_labels(saved.options.task))
                check_true(model.predict(clip(label)).label == label);
        }
    }
    it("preserves untrained status while allowing further training after loading") {
        auto original = experiment(false);
        original.sessions[0].heldout.clear();
        auto restored = preview::CompositionSession::restore(kfcore::gesture_interaction::decode_experiment(kfcore::gesture_interaction::encode_experiment(original),kPipeline).sessions[0]);
        check_true(restored->summary().find("UNTRAINED |") == 0);
        restored->key('t');
        check_true(restored->summary().find("TRAINED |") == 0);
        check_true(restored->dirty());
        restored->mark_saved(); check_false(restored->dirty());
    }
    it("rejects damaged truncated incompatible and extra data") {
        const auto bytes = kfcore::gesture_interaction::encode_experiment(experiment());
        auto bad = bytes; bad[100] ^= 1;
        check_throws_with(kfcore::gesture_interaction::decode_experiment(bad,kPipeline),"checksum");
        bad = bytes; bad.pop_back();
        check_throws_with(kfcore::gesture_interaction::decode_experiment(bad,kPipeline),"checksum");
        bad = bytes; bad[8] = 99; rehash(bad);
        check_throws_with(kfcore::gesture_interaction::decode_experiment(bad,kPipeline),"version");
        bad = bytes; bad[24] = 99; rehash(bad);
        check_throws_with(kfcore::gesture_interaction::decode_experiment(bad,kPipeline),"dimensions");
        bad = bytes; bad.insert(bad.end()-SALTS_CRYPTO_SHA256_DIGEST_SIZE,0); rehash(bad);
        check_throws_with(kfcore::gesture_interaction::decode_experiment(bad,kPipeline),"trailing");
        check_throws_with(kfcore::gesture_interaction::decode_experiment(bytes,"another model"),"identity");
    }
    it("rejects invalid options labels weights and samples before saving") {
        auto bad = experiment(); bad.sessions[0].options.maximum_clips = 0;
        check_throws_as(kfcore::gesture_interaction::encode_experiment(bad),std::invalid_argument);
        bad = experiment(); bad.sessions[0].options.record_countdown_seconds = (std::numeric_limits<double>::max)();
        check_throws_as(kfcore::gesture_interaction::encode_experiment(bad),std::invalid_argument);
        bad = experiment(); bad.sessions[1].training[0].label = kfcore::gesture_interaction::Composition::Release;
        check_throws_with(kfcore::gesture_interaction::encode_experiment(bad),"different task");
        bad = experiment(); bad.sessions[0].weights.readout[0] = std::numeric_limits<float>::quiet_NaN();
        check_throws_with(kfcore::gesture_interaction::encode_experiment(bad),"non-finite");
        bad = experiment(); bad.sessions[0].training[0].samples[2].seconds = 0;
        check_throws_with(kfcore::gesture_interaction::encode_experiment(bad),"timestamp");
        bad = experiment(); bad.sessions[1].training.clear();
        check_throws_with(kfcore::gesture_interaction::encode_experiment(bad),"lacks training samples");
    }
    it("saves exclusively and refuses overwriting an existing experiment") {
        TempDirectory directory;
        const auto data = experiment();
        kfcore::gesture_interaction::save_experiment(directory.file(),data);
        check_true(kfcore::gesture_interaction::encode_experiment(kfcore::gesture_interaction::load_experiment(directory.file(),kPipeline)) == kfcore::gesture_interaction::encode_experiment(data));
        check_throws_with(kfcore::gesture_interaction::save_experiment(directory.file(),experiment(false)),"already exists");
        check_true(kfcore::gesture_interaction::load_experiment(directory.file(),kPipeline).sessions[0].trained);
        check_false(std::filesystem::exists(directory.file().wstring()+L".partial"));
    }
    it("leaves another writers partial file untouched and preserves live session on load failure") {
        TempDirectory directory;
        const auto partial = directory.file().string()+".partial";
        check_true(tt_write_file(partial.c_str(),"owned elsewhere",15) == 0);
        auto current = preview::CompositionSession::restore(session(kfcore::gesture_interaction::CompositionTask::Motion,true));
        const auto before = current->summary();
        check_throws_with(kfcore::gesture_interaction::save_experiment(directory.file(),experiment()),"open");
        check_false(std::filesystem::exists(directory.file()));
        check_true(std::filesystem::file_size(partial) == 15);
        check_throws(kfcore::gesture_interaction::load_experiment(directory.file(),kPipeline));
        check_true(current->summary() == before);
    }
    it("does not archive an unfinished recording") {
        auto current = preview::CompositionSession::restore(session(kfcore::gesture_interaction::CompositionTask::Motion,false));
        current->key('1');
        check_throws_with(current->archive(),"recording");
        current->key('c');
        check_nothrow(current->archive());
    }
}
