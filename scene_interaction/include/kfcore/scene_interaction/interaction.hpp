#pragma once

#include "kfcore/pipelines/scene_graph.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kfcore::scene_interaction
{

inline constexpr std::size_t kSceneGeometryFeatureCount = 6U;

struct PairKey
{
    std::uint64_t subject_track_id = 0U;
    std::uint64_t object_track_id = 0U;

    friend bool operator==(const PairKey& left, const PairKey& right) noexcept
    {
        return left.subject_track_id == right.subject_track_id &&
               left.object_track_id == right.object_track_id;
    }

    friend bool operator<(const PairKey& left, const PairKey& right) noexcept
    {
        return left.subject_track_id < right.subject_track_id ||
               (left.subject_track_id == right.subject_track_id &&
                left.object_track_id < right.object_track_id);
    }
};

struct SceneBehaviorModel
{
    std::size_t predicate_count = 0U;
    int reservoir_size = 0;
    float leak_rate = 0.5F;
    std::size_t neutral_index = 0U;
    std::vector<std::string> labels;

    // Column-major matrices, matching kfcore_esn_model.
    std::vector<float> input_weights;
    std::vector<float> recurrent_weights;
    std::vector<float> reservoir_bias;
    std::vector<float> output_weights;
    std::vector<float> output_bias;
};

struct SceneInteractionOptions
{
    float minimum_score = 0.5F;
    float minimum_margin = 0.15F;
    double confirmation_seconds = 0.20;
    double end_seconds = 0.30;
    double maximum_gap_seconds = 0.50;
    std::size_t max_pair_states = 128U;
};

enum class SceneBehaviorEventKind
{
    BehaviorStarted,
    BehaviorEnded,
    BehaviorCancelled,
};

enum class SceneBehaviorEventReason
{
    Recognized,
    Unrecognized,
    PairLost,
    FrameGap,
    Reset,
    ModelChanged,
};

struct SceneBehaviorEvent
{
    SceneBehaviorEventKind kind = SceneBehaviorEventKind::BehaviorStarted;
    SceneBehaviorEventReason reason = SceneBehaviorEventReason::Recognized;
    double seconds = 0.0;
    PairKey pair;
    std::int32_t subject_class_id = -1;
    std::int32_t object_class_id = -1;
    std::size_t behavior_index = 0U;
    float score = 0.0F;
};

class SceneInteraction final
{
public:
    explicit SceneInteraction(SceneInteractionOptions options = {});
    ~SceneInteraction();

    SceneInteraction(const SceneInteraction&) = delete;
    SceneInteraction& operator=(const SceneInteraction&) = delete;

    [[nodiscard]] std::vector<SceneBehaviorEvent>
    configure_model(const SceneBehaviorModel& model, double seconds);

    [[nodiscard]] std::vector<SceneBehaviorEvent>
    process(const pipelines::SceneGraphFrame& frame, double seconds);

    [[nodiscard]] std::vector<SceneBehaviorEvent> advance(double seconds);
    [[nodiscard]] std::vector<SceneBehaviorEvent> reset(double seconds);

    [[nodiscard]] bool configured() const noexcept;
    [[nodiscard]] std::size_t pair_state_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] const char* event_kind_name(SceneBehaviorEventKind kind);
[[nodiscard]] const char* event_reason_name(SceneBehaviorEventReason reason);

} // namespace kfcore::scene_interaction
