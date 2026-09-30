#pragma once

#include "kfcore/image_processor/types.hpp"
#include "kfcore/relation/open_vocabulary_relation.hpp"
#include "kfcore/relation/relate_anything.hpp"
#include "kfcore/yolo/detector.hpp"
#include "kfcore/yolo/tracking.hpp"

#include <cstdint>
#include <memory>

namespace kfcore::pipelines
{

struct SceneGraphFrame
{
    yolo::TrackFrame objects;
    relation::RelationFrame relations;
};

struct SceneGraphPipelineOptions
{
    yolo::ByteTrackOptions tracking;
};

class SceneGraphPipeline final
{
public:
    ~SceneGraphPipeline();

    SceneGraphPipeline(const SceneGraphPipeline&) = delete;
    SceneGraphPipeline& operator=(const SceneGraphPipeline&) = delete;

    [[nodiscard]] static std::unique_ptr<SceneGraphPipeline>
    create(std::unique_ptr<yolo::YoloDetector> detector,
           std::unique_ptr<relation::RelateAnything> relation_model,
           const SceneGraphPipelineOptions& options = {});

    [[nodiscard]] static std::unique_ptr<SceneGraphPipeline>
    create(std::unique_ptr<yolo::YoloDetector> detector,
           std::unique_ptr<relation::OpenVocabularyRelation> relation_model,
           const SceneGraphPipelineOptions& options = {});

    [[nodiscard]] SceneGraphFrame process(const image::ImageView& image);

    // Dynamic-vocabulary control is available only when this pipeline was
    // created with OpenVocabularyRelation. Vocabulary changes do not reset
    // ByteTrack identity; RelationFrame::vocabulary_version tells temporal
    // consumers when predicate-index semantics changed.
    [[nodiscard]] bool supports_dynamic_vocabulary() const noexcept;
    void set_vocabulary(relation::PredicateVocabulary vocabulary);
    [[nodiscard]] std::uint64_t vocabulary_version() const noexcept;

    void reset_tracking() noexcept;

private:
    struct Impl;
    explicit SceneGraphPipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::pipelines
