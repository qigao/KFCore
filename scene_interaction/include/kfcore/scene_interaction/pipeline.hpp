#pragma once

#include "kfcore/image_processor/types.hpp"
#include "kfcore/pipelines/scene_graph.hpp"
#include "kfcore/scene_interaction/interaction.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kfcore::scene_interaction
{

struct SceneBehaviorFrame
{
    pipelines::SceneGraphFrame scene;
    std::vector<SceneBehaviorEvent> events;
};

class SceneBehaviorPipeline final
{
public:
    ~SceneBehaviorPipeline();

    SceneBehaviorPipeline(const SceneBehaviorPipeline&) = delete;
    SceneBehaviorPipeline& operator=(const SceneBehaviorPipeline&) = delete;

    [[nodiscard]] static std::unique_ptr<SceneBehaviorPipeline>
    create(std::unique_ptr<pipelines::SceneGraphPipeline> scene_graph,
           const SceneInteractionOptions& interaction_options = {});

    [[nodiscard]] std::vector<SceneBehaviorEvent>
    configure_model(const SceneBehaviorModel& model, double seconds);

    [[nodiscard]] SceneBehaviorFrame
    process(const image::ImageView& image, double seconds);

    [[nodiscard]] std::vector<SceneBehaviorEvent>
    advance(double seconds);

    [[nodiscard]] std::vector<SceneBehaviorEvent>
    reset_temporal(double seconds);

    void reset_tracking() noexcept;

    [[nodiscard]] bool supports_dynamic_vocabulary() const noexcept;
    [[nodiscard]] bool supports_live_predicates() const noexcept;

    void set_vocabulary(relation::PredicateVocabulary vocabulary);
    void set_predicates(const std::vector<std::string>& predicates);

    [[nodiscard]] std::uint64_t vocabulary_version() const noexcept;
    [[nodiscard]] const std::vector<std::string>& predicates() const noexcept;

    [[nodiscard]] bool temporal_configured() const noexcept;
    [[nodiscard]] std::size_t pair_state_count() const noexcept;

private:
    struct Impl;
    explicit SceneBehaviorPipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::scene_interaction
