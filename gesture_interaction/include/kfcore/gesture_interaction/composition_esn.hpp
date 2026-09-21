#pragma once
#include <kfcore/mediapipe/gesture_recognizer.hpp>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace kfcore::gesture_interaction {
enum class Composition { None, OpenCloseOpen, CloseOpenClose, Grasp, Release, Wave };
enum class CompositionTask { Legacy, Interaction, Motion };
const std::vector<Composition>& composition_labels(CompositionTask task);
int composition_index(CompositionTask task, Composition label);
inline constexpr int kCompositionClasses = 3;
inline constexpr int kCompositionInputs = kfcore::mediapipe::kGestureCount + 2;
inline constexpr int kCompositionNeurons = 48;
inline constexpr int kCompositionSteps = 60;
inline constexpr int kHistoryParts = 3;
inline constexpr int kHistoryFeatures = kCompositionNeurons * kHistoryParts;
inline constexpr int kCompositionFeatures = kHistoryFeatures + kCompositionNeurons * (kHistoryParts-1);
using CompositionState = std::array<float, kCompositionFeatures>;
inline constexpr int kMotionInputs = kCompositionInputs + 4;
struct CompositionSample {
    double seconds;
    std::array<float, kCompositionInputs> values{};
    std::optional<std::array<float, 2>> wrist; // Source-image fractions, before display mirroring.
};
using CompositionClip = std::vector<CompositionSample>;
struct CompositionOptions {
    CompositionTask task = CompositionTask::Legacy;
    float displacement_scale = 4.0F;
    float velocity_scale = 1.0F;
    double live_stride_seconds = 0.1;
    double record_countdown_seconds = 2.0;
    double duration_seconds = 3.0;
    double maximum_gap_seconds = 0.25;
    std::size_t maximum_samples = 256;
    std::size_t maximum_clips = 120;
    int minimum_clips_per_class = 3;
    float ridge = 0.1F;
    float leak = 0.5F;
    float spectral_radius = 0.8F;
    float input_scale = 0.5F;
    std::uint64_t seed = 42;
};
struct CompositionPrediction {
    Composition label;
    std::array<float, kCompositionClasses> scores; // Linear readout, NOT probabilities.
    int class_count = kCompositionClasses; // Only this prefix of scores is active.
};
struct CompositionWeights {
    std::array<float, kCompositionNeurons * kMotionInputs> input{};
    std::array<float, kCompositionNeurons * kCompositionNeurons> recurrent{};
    std::array<float, kCompositionNeurons> bias{};
    std::array<float, kCompositionClasses * kCompositionFeatures> readout{};
};
// Experimental, single-threaded, in-memory clip classifier, not an event detector.
// Trained and held-out clips are disjoint by caller collection session/recording.
class CompositionEsn final {
public:
    explicit CompositionEsn(CompositionOptions options = {});
    CompositionState encode(const CompositionClip& clip) const;
    void add_training(const CompositionClip& clip, Composition label);
    void train_readout();
    CompositionPrediction predict(const CompositionClip& clip) const;
    bool trained() const noexcept { return trained_; }
    const auto& counts() const noexcept { return counts_; }
    const CompositionOptions& options() const noexcept { return options_; }
    int class_count() const { return int(composition_labels(options_.task).size()); }
    CompositionWeights weights() const { return {input_, recurrent_, bias_, readout_}; }
    void restore_weights(const CompositionWeights& weights);
    void restore_trained_readout(const CompositionWeights& weights);
private:
    CompositionOptions options_;
    int input_count() const noexcept { return options_.task == CompositionTask::Motion ? kMotionInputs : kCompositionInputs; }
    std::array<float, kCompositionNeurons * kMotionInputs> input_{};
    std::array<float, kCompositionNeurons * kCompositionNeurons> recurrent_{};
    std::array<float, kCompositionNeurons> bias_{};
    std::array<float, kCompositionClasses * kCompositionFeatures> readout_{};
    std::array<std::size_t, kCompositionClasses> counts_{};
    std::vector<CompositionState> states_;
    std::vector<Composition> labels_;
    bool trained_ = false;
};
std::optional<CompositionSample> composition_sample(const kfcore::mediapipe::GestureFrame& frame, double seconds,
    int width = 0, int height = 0);
const char* composition_name(Composition label);
} // namespace kfcore::gesture_interaction
