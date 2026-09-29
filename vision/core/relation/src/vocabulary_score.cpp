#include "vocabulary_score.hpp"

#include "kfcore/relation/error.hpp"

#include <cmath>
#include <cstddef>
#include <string>

namespace kfcore::relation::detail
{
namespace
{

[[noreturn]] void throw_score_error(const char* detail)
{
    throw RelationError(RelationErrorCode::RuntimeFailure,
                        std::string("open-vocabulary scoring: ") + detail);
}

float inverse_norm(const float* values, std::size_t count)
{
    double sum = 0.0;
    for (std::size_t index = 0U; index < count; ++index)
    {
        const float value = values[index];
        if (!std::isfinite(value))
        {
            throw_score_error("query contains a non-finite value");
        }
        sum += static_cast<double>(value) * static_cast<double>(value);
    }
    if (!(sum > 0.0) || !std::isfinite(sum))
    {
        return 0.0F;
    }
    return static_cast<float>(1.0 / std::sqrt(sum));
}

float cosine(const float* query, float inv_query_norm,
             const float* normalized_embedding, std::size_t count)
{
    if (inv_query_norm == 0.0F)
    {
        return 0.0F;
    }

    double dot = 0.0;
    for (std::size_t index = 0U; index < count; ++index)
    {
        const float embedding = normalized_embedding[index];
        if (!std::isfinite(embedding))
        {
            throw_score_error("vocabulary contains a non-finite value");
        }
        dot += static_cast<double>(query[index]) *
               static_cast<double>(embedding);
    }
    return static_cast<float>(dot) * inv_query_norm;
}

} // namespace

void score_open_vocabulary_queries(
    const RawOpenVocabularyQueries& queries,
    const NormalizedVocabularyView& vocabulary,
    float logit_scale,
    float logit_bias,
    float* output_logits)
{
    if (queries.semantic_query == nullptr ||
        queries.spatial_query == nullptr ||
        queries.valid_mask == nullptr ||
        vocabulary.embeddings == nullptr ||
        output_logits == nullptr)
    {
        throw_score_error("null scoring buffer");
    }
    if (queries.query_dim == 0U ||
        queries.query_dim != vocabulary.embedding_dim ||
        vocabulary.predicate_count == 0U)
    {
        throw_score_error("query/vocabulary shape mismatch");
    }
    if (!std::isfinite(logit_scale) || logit_scale <= 0.0F ||
        !std::isfinite(logit_bias))
    {
        throw_score_error("invalid logit scale/bias");
    }

    for (std::size_t pair = 0U; pair < queries.pair_count; ++pair)
    {
        float* row =
            output_logits + pair * vocabulary.predicate_count;
        if (queries.valid_mask[pair] == 0U)
        {
            for (std::size_t predicate = 0U;
                 predicate < vocabulary.predicate_count; ++predicate)
            {
                row[predicate] = 0.0F;
            }
            continue;
        }

        const float* semantic =
            queries.semantic_query + pair * queries.query_dim;
        const float* spatial =
            queries.spatial_query + pair * queries.query_dim;
        const float semantic_inv_norm =
            inverse_norm(semantic, queries.query_dim);
        const float spatial_inv_norm =
            inverse_norm(spatial, queries.query_dim);

        for (std::size_t predicate = 0U;
             predicate < vocabulary.predicate_count; ++predicate)
        {
            const float* embedding =
                vocabulary.embeddings +
                predicate * vocabulary.embedding_dim;
            const float alpha =
                vocabulary.spatial_weights == nullptr
                    ? 0.0F
                    : vocabulary.spatial_weights[predicate];
            if (!std::isfinite(alpha) || alpha < 0.0F || alpha > 1.0F)
            {
                throw_score_error("spatial routing weight is outside [0,1]");
            }

            const float semantic_score =
                cosine(semantic, semantic_inv_norm, embedding,
                       queries.query_dim);
            const float spatial_score =
                cosine(spatial, spatial_inv_norm, embedding,
                       queries.query_dim);
            const float mixed =
                (1.0F - alpha) * semantic_score +
                alpha * spatial_score;
            row[predicate] = logit_scale * mixed + logit_bias;
        }
    }
}

} // namespace kfcore::relation::detail
