#pragma once

#include "kfcore/image_processor/types.hpp"
#include "kfcore/relation/types.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/resolver.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace kfcore::relation
{

inline constexpr std::string_view kOpenVocabularyRelationModelType =
    "relation.open-vocabulary-encoder";

struct PredicateVocabulary
{
    std::vector<std::string> predicates;
    std::vector<float> embeddings;
    std::vector<float> spatial_weights;
    std::size_t embedding_dim = 0U;
};

[[nodiscard]] PredicateVocabulary normalize_predicate_vocabulary(
    PredicateVocabulary vocabulary,
    std::size_t expected_embedding_dim,
    std::size_t max_bytes);

struct OpenVocabularyRelationOptions
{
    std::int32_t input_size = 0;
    std::size_t max_boxes = 32U;
    std::size_t max_pairs = 128U;
    std::size_t query_dim = 512U;

    float logit_scale = 14.285714F;
    float logit_bias = 0.0F;
    float threshold = 0.40F;
    float pair_weight = 1.0F;
    float calibration_a = 1.0F;
    float calibration_b = 0.0F;

    std::size_t top_k = 20U;
    bool weight_ranking_by_detector_score = true;

    std::size_t max_source_bytes = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes = 64U * 1024U * 1024U;
    std::size_t max_output_bytes = 64U * 1024U * 1024U;
    std::size_t max_vocabulary_bytes = 64U * 1024U * 1024U;
};

class OpenVocabularyRelation final
{
public:
    ~OpenVocabularyRelation();

    OpenVocabularyRelation(const OpenVocabularyRelation&) = delete;
    OpenVocabularyRelation& operator=(const OpenVocabularyRelation&) = delete;

    [[nodiscard]] static std::unique_ptr<OpenVocabularyRelation>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& package,
         const runtime::ExecutionPolicy& policy,
         const OpenVocabularyRelationOptions& options = {});

    void set_vocabulary(PredicateVocabulary vocabulary);

    [[nodiscard]] RelationFrame infer(const image::ImageView& image,
                                      const std::vector<Region>& regions);

    [[nodiscard]] std::int32_t input_size() const noexcept;
    [[nodiscard]] std::size_t max_boxes() const noexcept;
    [[nodiscard]] std::size_t max_pairs() const noexcept;
    [[nodiscard]] std::size_t query_dim() const noexcept;
    [[nodiscard]] std::size_t predicate_count() const noexcept;
    [[nodiscard]] const std::vector<std::string>& predicates() const noexcept;
    [[nodiscard]] const runtime::ExecutionRoute& execution_route() const noexcept;

private:
    struct Impl;
    explicit OpenVocabularyRelation(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::relation
