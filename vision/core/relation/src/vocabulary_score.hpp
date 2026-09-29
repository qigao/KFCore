#pragma once

#include <cstddef>
#include <cstdint>

namespace kfcore::relation::detail
{

struct RawOpenVocabularyQueries
{
    const float* semantic_query = nullptr;
    const float* spatial_query = nullptr;
    const std::uint8_t* valid_mask = nullptr;
    std::size_t pair_count = 0U;
    std::size_t query_dim = 0U;
};

struct NormalizedVocabularyView
{
    const float* embeddings = nullptr;
    const float* spatial_weights = nullptr;
    std::size_t predicate_count = 0U;
    std::size_t embedding_dim = 0U;
};

void score_open_vocabulary_queries(
    const RawOpenVocabularyQueries& queries,
    const NormalizedVocabularyView& vocabulary,
    float logit_scale,
    float logit_bias,
    float* output_logits);

} // namespace kfcore::relation::detail
