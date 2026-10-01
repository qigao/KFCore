#pragma once

#include "kfcore/image_processor/types.hpp"
#include "kfcore/relation/open_vocabulary_relation.hpp"
#include "kfcore/relation/predicate_routing_gate.hpp"
#include "kfcore/relation/predicate_text_encoder.hpp"
#include "kfcore/relation/relate_anything.hpp"
#include "kfcore/yolo/detector.hpp"
#include "kfcore/yolo/tracking.hpp"

#include <cstddef>
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

struct SceneGraphTiming
{
    double detector_ms = 0.0;
    double tracker_ms = 0.0;
    double region_prepare_ms = 0.0;
    double relation_ms = 0.0;
    double assembly_ms = 0.0;
    double total_ms = 0.0;

    std::size_t detection_count = 0U;
    std::size_t tracked_object_count = 0U;
    std::size_t relation_edge_count = 0U;
};

struct TimedSceneGraphFrame
{
    SceneGraphFrame frame;
    SceneGraphTiming timing;
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

    [[nodiscard]] SceneGraphFrame process(const image::ImageView& image);

    [[nodiscard]] TimedSceneGraphFrame
    process_timed(const image::ImageView& image);

    // Dynamic-vocabulary control is available only when this pipeline was
    // created with OpenVocabularyRelation. Vocabulary changes do not reset
    // ByteTrack identity; RelationFrame::vocabulary_version tells temporal
    // consumers when predicate-index semantics changed.
    [[nodiscard]] bool supports_dynamic_vocabulary() const noexcept;
    [[nodiscard]] bool supports_live_predicates() const noexcept;

    void set_vocabulary(relation::PredicateVocabulary vocabulary);
    void set_predicates(const std::vector<std::string>& predicates);

    [[nodiscard]] std::uint64_t vocabulary_version() const noexcept;
    [[nodiscard]] const std::vector<std::string>& predicates() const noexcept;

    [[nodiscard]] std::uint64_t tracking_epoch() const noexcept;
    [[nodiscard]] std::uint64_t reset_tracking();

private:
    struct Impl;
    explicit SceneGraphPipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::pipelines
