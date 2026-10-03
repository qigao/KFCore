#pragma once

#include "kfcore/pipelines/scene_graph.hpp"

#include <memory>
#include <string>
#include <vector>

namespace kfcore::pipelines::detail
{

class DetectionRunner
{
public:
    virtual ~DetectionRunner() = default;

    DetectionRunner(const DetectionRunner&) = delete;
    DetectionRunner& operator=(const DetectionRunner&) = delete;

    [[nodiscard]] virtual yolo::DetectionFrame
    detect(const image::ImageView& image) = 0;

protected:
    DetectionRunner() = default;
};

class RelationRunner
{
public:
    virtual ~RelationRunner() = default;

    RelationRunner(const RelationRunner&) = delete;
    RelationRunner& operator=(const RelationRunner&) = delete;

    [[nodiscard]] virtual std::size_t max_boxes() const noexcept = 0;
    [[nodiscard]] virtual bool supports_dynamic_vocabulary() const noexcept = 0;
    [[nodiscard]] virtual bool supports_live_predicates() const noexcept = 0;

    virtual void set_vocabulary(
        relation::PredicateVocabulary vocabulary) = 0;
    virtual void set_predicates(
        const std::vector<std::string>& predicates) = 0;

    [[nodiscard]] virtual std::uint64_t
    vocabulary_version() const noexcept = 0;

    [[nodiscard]] virtual const std::vector<std::string>&
    predicates() const noexcept = 0;

    [[nodiscard]] virtual relation::RelationFrame
    infer(const image::ImageView& image,
          const std::vector<relation::Region>& regions) = 0;

protected:
    RelationRunner() = default;
};

class SceneGraphEngine final
{
public:
    SceneGraphEngine(
        std::unique_ptr<DetectionRunner> detector,
        std::unique_ptr<RelationRunner> relation_model,
        const SceneGraphPipelineOptions& options = {});

    SceneGraphEngine(SceneGraphEngine&&) noexcept = default;
    SceneGraphEngine& operator=(SceneGraphEngine&&) noexcept = default;
    SceneGraphEngine(const SceneGraphEngine&) = delete;
    SceneGraphEngine& operator=(const SceneGraphEngine&) = delete;

    [[nodiscard]] SceneGraphFrame
    process(const image::ImageView& image);

    [[nodiscard]] TimedSceneGraphFrame
    process_timed(const image::ImageView& image);

    [[nodiscard]] bool
    supports_dynamic_vocabulary() const noexcept;

    [[nodiscard]] bool
    supports_live_predicates() const noexcept;

    void set_vocabulary(
        relation::PredicateVocabulary vocabulary);

    void set_predicates(
        const std::vector<std::string>& predicates);

    [[nodiscard]] std::uint64_t
    vocabulary_version() const noexcept;

    [[nodiscard]] const std::vector<std::string>&
    predicates() const noexcept;

    [[nodiscard]] std::uint64_t tracking_epoch() const noexcept;
    [[nodiscard]] std::uint64_t reset_tracking();

private:
    std::unique_ptr<DetectionRunner> detector_;
    std::unique_ptr<RelationRunner> relation_model_;
    yolo::ByteTrackSession tracking_;
};

[[nodiscard]] std::vector<relation::Region>
regions_from_tracks(const yolo::TrackFrame& tracks);

[[nodiscard]] SceneGraphFrame
assemble_scene_graph(yolo::TrackFrame tracks,
                     relation::RelationFrame relations);

} // namespace kfcore::pipelines::detail
