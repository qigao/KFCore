#include <kfcore/gesture_interaction/composition_esn.hpp>
#include <esn.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace kfcore::gesture_interaction {
namespace {
constexpr int kPowerIterations = 200;
constexpr float kDensity = 0.3F;
void require_status(kfcore_esn_status status) {
    if (status != KFCORE_ESN_OK) throw std::runtime_error("ESN operation failed: " + std::to_string(status));
}
}
const char* composition_name(Composition label) {
    switch (label) {
    case Composition::None: return "No-combo";
    case Composition::OpenCloseOpen: return "Open-Close-Open";
    case Composition::CloseOpenClose: return "Close-Open-Close";
    case Composition::Grasp: return "Grasp";
    case Composition::Release: return "Release";
    case Composition::Wave: return "Wave";
    }
    throw std::invalid_argument("invalid composition label");
}
const std::vector<Composition>& composition_labels(CompositionTask task) {
    static const std::vector<Composition> legacy{Composition::None, Composition::OpenCloseOpen, Composition::CloseOpenClose};
    static const std::vector<Composition> interaction{Composition::None, Composition::Grasp, Composition::Release};
    static const std::vector<Composition> motion{Composition::None, Composition::Wave};
    switch (task) {
    case CompositionTask::Legacy: return legacy;
    case CompositionTask::Interaction: return interaction;
    case CompositionTask::Motion: return motion;
    }
    throw std::invalid_argument("invalid composition task");
}
int composition_index(CompositionTask task, Composition label) {
    const auto& labels = composition_labels(task);
    const auto found = std::find(labels.begin(), labels.end(), label);
    if (found == labels.end()) throw std::invalid_argument("label belongs to a different task");
    return int(found-labels.begin());
}
CompositionEsn::CompositionEsn(CompositionOptions options) : options_(options) {
    (void)composition_labels(options.task);
    if (!std::isfinite(options.duration_seconds) || options.duration_seconds <= 0 ||
        !std::isfinite(options.live_stride_seconds) || options.live_stride_seconds <= 0 ||
        options.live_stride_seconds > options.duration_seconds ||
        !std::isfinite(options.record_countdown_seconds) || options.record_countdown_seconds < 0 ||
        options.record_countdown_seconds > std::numeric_limits<int>::max() ||
        !std::isfinite(options.maximum_gap_seconds) || options.maximum_gap_seconds <= 0 ||
        options.maximum_samples < 2 || options.maximum_samples > 4096 ||
        options.maximum_clips < std::size_t(class_count()) || options.maximum_clips > 4096 ||
        options.minimum_clips_per_class < 1 ||
        std::size_t(options.minimum_clips_per_class) > options.maximum_clips / class_count() ||
        !std::isfinite(options.ridge) || options.ridge <= 0 ||
        !std::isfinite(options.leak) || options.leak <= 0 || options.leak > 1 ||
        !std::isfinite(options.input_scale) || options.input_scale <= 0 ||
        !std::isfinite(options.displacement_scale) || options.displacement_scale <= 0 ||
        !std::isfinite(options.velocity_scale) || options.velocity_scale <= 0)
        throw std::invalid_argument("invalid ESN composition options");
    require_status(kfcore_esn_init_reservoir_weights(recurrent_.data(), kCompositionNeurons, options.seed, kDensity));
    std::array<float, kCompositionNeurons * 2> workspace{};
    require_status(kfcore_esn_scale_spectral_radius(recurrent_.data(), kCompositionNeurons,
        options.spectral_radius, kPowerIterations, workspace.data()));
    // Reuse the core deterministic initializer; take a rectangular prefix.
    std::array<float, kCompositionNeurons * kCompositionNeurons> generated{};
    require_status(kfcore_esn_init_reservoir_weights(generated.data(), kCompositionNeurons, options.seed + 1, 1.0F));
    const int weights = kCompositionNeurons * input_count();
    for (int i = 0; i < weights; ++i) input_[i] = generated[i] * options.input_scale;
    for (int i = 0; i < kCompositionNeurons; ++i) bias_[i] = generated[weights + i] * options.input_scale;
    states_.reserve(options.maximum_clips);
    labels_.reserve(options.maximum_clips);
}
CompositionState CompositionEsn::encode(const CompositionClip& clip) const {
    if (clip.size() < 2 || clip.size() > options_.maximum_samples)
        throw std::invalid_argument("clip sample count outside bounds");
    for (std::size_t i = 0; i < clip.size(); ++i) {
        const auto& s = clip[i];
        if (!std::isfinite(s.seconds) || (i && (s.seconds <= clip[i-1].seconds ||
            s.seconds - clip[i-1].seconds > options_.maximum_gap_seconds)))
            throw std::invalid_argument("clip contains invalid sample or timestamp gap");
        for (float value : s.values)
            if (!std::isfinite(value) || value < 0 || value > 1)
                throw std::invalid_argument("gesture input outside [0,1]");
        if (options_.task == CompositionTask::Motion && s.values.back() != 0 && !s.wrist)
            throw std::invalid_argument("motion input requires normalized wrist coordinates");
        if (s.wrist) for (float coordinate : *s.wrist)
            if (!std::isfinite(coordinate)) throw std::invalid_argument("non-finite wrist coordinate");
    }
    if (clip.back().seconds - clip.front().seconds < options_.duration_seconds)
        throw std::invalid_argument("clip is shorter than configured window");
    kfcore_esn_model model{input_count(), kCompositionNeurons, class_count(), options_.leak,
        input_.data(), recurrent_.data(), bias_.data(), nullptr, nullptr};
    std::array<float, kCompositionNeurons> state{}, workspace{};
    CompositionState history{};
    std::size_t right = 1;
    const double step_seconds = options_.duration_seconds / (kCompositionSteps - 1);
    std::optional<std::array<float, 2>> origin;
    // O(samples + steps * neurons^2), bounded; every clip starts at zero state.
    for (int step = 0; step < kCompositionSteps; ++step) {
        const double t = clip.front().seconds + step * step_seconds;
        while (right + 1 < clip.size() && clip[right].seconds < t) ++right;
        const auto& a = clip[right-1]; const auto& b = clip[right];
        const float fraction = float((t - a.seconds) / (b.seconds - a.seconds));
        std::array<float, kMotionInputs> features{};
        for (int i = 0; i < kCompositionInputs; ++i)
            features[i] = a.values[i] + fraction * (b.values[i]-a.values[i]);
        // Do not synthesize motion across missing observations. Position is relative
        // to the clip origin, velocity uses real elapsed seconds, not camera FPS.
        if (options_.task == CompositionTask::Motion && a.wrist && b.wrist &&
            a.values.back() == 1 && b.values.back() == 1) {
            if (!origin) origin = a.wrist;
            for (int axis = 0; axis < 2; ++axis) {
                const float delta = (*b.wrist)[axis]-(*a.wrist)[axis];
                const float position = (*a.wrist)[axis] + fraction*delta;
                features[kCompositionInputs+axis] = std::tanh((position-(*origin)[axis])*options_.displacement_scale);
                features[kCompositionInputs+2+axis] = std::tanh(float(delta/(b.seconds-a.seconds))*options_.velocity_scale);
            }
        }
        require_status(kfcore_esn_step(&model, features.data(), state.data(), workspace.data()));
        // Preserve early/middle/late reservoir evidence instead of only the final pose.
        constexpr int kPartSteps = kCompositionSteps / kHistoryParts;
        if ((step + 1) % kPartSteps == 0)
            std::copy(state.begin(), state.end(), history.begin() + (step / kPartSteps) * kCompositionNeurons);
    }
    // Temporal contrast makes "no change" separable from both opposite changes.
    // A linear sum of early/late pose states alone can confuse static fists with
    // open->fist and fist->open. Legacy inputs retain zero contrast features.
    if (options_.task != CompositionTask::Legacy)
        for (int part = 1; part < kHistoryParts; ++part)
            for (int neuron = 0; neuron < kCompositionNeurons; ++neuron) {
                const float delta = history[part*kCompositionNeurons+neuron] - history[(part-1)*kCompositionNeurons+neuron];
                history[kHistoryFeatures+(part-1)*kCompositionNeurons+neuron] = delta*delta;
            }
    return history;
}
void CompositionEsn::add_training(const CompositionClip& clip, Composition label) {
    const auto index = composition_index(options_.task, label);
    if (states_.size() >= options_.maximum_clips) throw std::runtime_error("training clip capacity reached");
    const auto state = encode(clip);
    states_.push_back(state); labels_.push_back(label); ++counts_[index];
    trained_ = false;
}
void CompositionEsn::train_readout() {
    const int classes = class_count();
    for (int i = 0; i < classes; ++i) if (counts_[i] < std::size_t(options_.minimum_clips_per_class))
        throw std::runtime_error("need at least " + std::to_string(options_.minimum_clips_per_class) + " training clips per class");
    std::vector<float> states(states_.size()*kCompositionFeatures), targets(states_.size()*classes);
    for (std::size_t i = 0; i < states_.size(); ++i) {
        std::copy(states_[i].begin(), states_[i].end(), states.begin()+i*kCompositionFeatures);
        targets[i*classes+composition_index(options_.task, labels_[i])] = 1.0F;
    }
    decltype(readout_) candidate{};
    std::array<float, kCompositionFeatures*kCompositionFeatures> gram{};
    require_status(kfcore_esn_fit_ridge(states.data(), targets.data(), kCompositionFeatures, classes,
        int(states_.size()), options_.ridge, candidate.data(), gram.data()));
    readout_ = candidate; trained_ = true;
}
void CompositionEsn::restore_weights(const CompositionWeights& weights) {
    if (!states_.empty()) throw std::logic_error("restore reservoir before adding samples");
    for (const auto* values : {weights.input.data(), weights.recurrent.data(), weights.bias.data(), weights.readout.data()}) {
        const auto size = values == weights.input.data() ? weights.input.size() :
            values == weights.recurrent.data() ? weights.recurrent.size() :
            values == weights.bias.data() ? weights.bias.size() : weights.readout.size();
        for (std::size_t i = 0; i < size; ++i)
            if (!std::isfinite(values[i])) throw std::invalid_argument("non-finite saved ESN weight");
    }
    input_ = weights.input; recurrent_ = weights.recurrent; bias_ = weights.bias;
    readout_ = weights.readout; trained_ = false;
}
void CompositionEsn::restore_trained_readout(const CompositionWeights& weights) {
    for (int i = 0; i < class_count(); ++i)
        if (counts_[i] < std::size_t(options_.minimum_clips_per_class))
            throw std::invalid_argument("saved trained model lacks training samples");
    if (input_ != weights.input || recurrent_ != weights.recurrent || bias_ != weights.bias)
        throw std::invalid_argument("saved reservoir mismatch");
    for (float value : weights.readout)
        if (!std::isfinite(value)) throw std::invalid_argument("non-finite saved readout");
    readout_ = weights.readout; trained_ = true;
}
CompositionPrediction CompositionEsn::predict(const CompositionClip& clip) const {
    if (!trained_) throw std::runtime_error("ESN readout is not trained");
    const auto state = encode(clip);
    CompositionPrediction result{};
    result.class_count = class_count();
    const std::array<float, kCompositionClasses> bias{};
    kfcore_esn_model readout{};
    readout.reservoir_size = kCompositionFeatures;
    readout.output_size = result.class_count;
    readout.output_weights = readout_.data();
    readout.output_bias = bias.data();
    require_status(kfcore_esn_predict(&readout, state.data(), result.scores.data()));
    result.label = composition_labels(options_.task)[std::max_element(result.scores.begin(), result.scores.begin()+result.class_count) - result.scores.begin()];
    return result;
}
std::optional<CompositionSample> composition_sample(const kfcore::mediapipe::GestureFrame& frame, double seconds, int width, int height) {
    if (width < 0 || height < 0 || (width == 0) != (height == 0))
        throw std::invalid_argument("invalid image geometry");
    if (!std::isfinite(seconds) || frame.landmarks.hands.size() != frame.gestures.size())
        throw std::invalid_argument("invalid gesture frame metadata");
    if (frame.gestures.size() > 1) return std::nullopt;
    CompositionSample result{seconds};
    if (frame.gestures.empty()) return result;
    constexpr auto kPresence = kCompositionInputs - 1;
    constexpr auto kHandedness = kCompositionInputs - 2;
    const auto& hand = frame.landmarks.hands.front();
    if (!hand.right_hand_probability) throw std::invalid_argument("missing handedness probability");
    std::copy(frame.gestures.front().scores.begin(), frame.gestures.front().scores.end(), result.values.begin());
    result.values[kHandedness] = *hand.right_hand_probability;
    result.values[kPresence] = 1;
    if (width > 0) {
        const auto& wrist = hand.landmarks[0];
        if (!std::isfinite(wrist.x) || !std::isfinite(wrist.y))
            throw std::invalid_argument("non-finite wrist coordinate");
        result.wrist = std::array<float, 2>{wrist.x/width, wrist.y/height};
    }
    return result;
}
} // namespace kfcore::gesture_interaction
