#pragma once

#include "kfcore/image_processor/types.hpp"
#include "kfcore/relation/open_vocabulary_relation.hpp"
#include "kfcore/relation/predicate_routing_gate.hpp"
#include "kfcore/relation/predicate_text_encoder.hpp"
#include "kfcore/relation/relate_anything.hpp"
#include "kfcore/yolo/detector.hpp"
#include "kfcore/yolo/tracking.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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

    [[nodiscard]] static std::unique_ptr<SceneGraphPipeline>
    create(std::unique_ptr<yolo::YoloDetector> detector,
           std::unique_ptr<relation::OpenVocabularyRelation> relation_model,
           std::unique_ptr<relation::PredicateTextEncoder> text_encoder,
           std::unique_ptr<relation::PredicateRoutingGate> routing_gate,
           const SceneGraphPipelineOptions& options = {});

    void set_vocabulary(relation::PredicateVocabulary vocabulary);
    void set_predicates(const std::vector<std::string>& predicates);

    [[nodiscard]] std::uint64_t vocabulary_version() const noexcept;
    [[nodiscard]] const std::vector<std::string>& predicates() const noexcept;

    [[nodiscard]] SceneGraphFrame process(const image::ImageView& image);

    void reset_tracking() noexcept;

private:
    struct Impl;
    explicit SceneGraphPipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::pipelines
