#include "vocabulary_score.hpp"

#include "kfcore/relation/error.hpp"
#include "kfcore/relation/open_vocabulary_relation.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>

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


namespace kfcore::relation
{
namespace
{

[[noreturn]] void throw_vocabulary_invalid(const std::string& detail)
{
    throw RelationError(RelationErrorCode::InvalidArgument,
                        "PredicateVocabulary: " + detail);
}

[[noreturn]] void throw_vocabulary_resource(const std::string& detail)
{
    throw RelationError(RelationErrorCode::ResourceLimitExceeded,
                        "PredicateVocabulary: " + detail);
}

std::size_t checked_vocabulary_multiply(
    std::size_t left, std::size_t right)
{
    if (left != 0U &&
        right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_vocabulary_resource("byte count overflow");
    }
    return left * right;
}

std::size_t checked_vocabulary_add(
    std::size_t left, std::size_t right)
{
    if (right > (std::numeric_limits<std::size_t>::max)() - left)
    {
        throw_vocabulary_resource("byte count overflow");
    }
    return left + right;
}

} // namespace

PredicateVocabulary normalize_predicate_vocabulary(
    PredicateVocabulary vocabulary,
    std::size_t expected_embedding_dim,
    std::size_t max_bytes)
{
    if (expected_embedding_dim == 0U || max_bytes == 0U)
    {
        throw_vocabulary_invalid(
            "limits and expected embedding dimension must be positive");
    }
    if (vocabulary.predicates.empty())
    {
        throw_vocabulary_invalid("predicate vocabulary must not be empty");
    }
    if (vocabulary.embedding_dim != expected_embedding_dim)
    {
        throw_vocabulary_invalid(
            "embedding dimension does not match relation encoder");
    }

    const std::size_t expected_values =
        checked_vocabulary_multiply(
            vocabulary.predicates.size(),
            vocabulary.embedding_dim);
    if (vocabulary.embeddings.size() != expected_values)
    {
        throw_vocabulary_invalid(
            "embedding matrix has the wrong number of values");
    }
    if (!vocabulary.spatial_weights.empty() &&
        vocabulary.spatial_weights.size() !=
            vocabulary.predicates.size())
    {
        throw_vocabulary_invalid(
            "spatial_weights must be empty or contain V values");
    }

    std::size_t bytes = checked_vocabulary_multiply(
        vocabulary.embeddings.size(), sizeof(float));
    bytes = checked_vocabulary_add(
        bytes,
        checked_vocabulary_multiply(
            vocabulary.spatial_weights.empty()
                ? vocabulary.predicates.size()
                : vocabulary.spatial_weights.size(),
            sizeof(float)));
    for (const auto& name : vocabulary.predicates)
    {
        bytes = checked_vocabulary_add(bytes, name.size());
    }
    if (bytes > max_bytes)
    {
        throw_vocabulary_resource(
            "vocabulary exceeds configured byte limit");
    }

    std::unordered_set<std::string> names;
    names.reserve(vocabulary.predicates.size());
    for (const auto& name : vocabulary.predicates)
    {
        if (name.empty())
        {
            throw_vocabulary_invalid(
                "predicate names must not be empty");
        }
        if (!names.insert(name).second)
        {
            throw_vocabulary_invalid(
                "predicate names must be unique");
        }
    }

    for (std::size_t row = 0U;
         row < vocabulary.predicates.size(); ++row)
    {
        double norm2 = 0.0;
        float* values =
            vocabulary.embeddings.data() +
            row * vocabulary.embedding_dim;
        for (std::size_t column = 0U;
             column < vocabulary.embedding_dim; ++column)
        {
            const float value = values[column];
            if (!std::isfinite(value))
            {
                throw_vocabulary_invalid(
                    "embedding values must be finite");
            }
            norm2 += static_cast<double>(value) *
                     static_cast<double>(value);
        }
        if (!(norm2 > 0.0) || !std::isfinite(norm2))
        {
            throw_vocabulary_invalid(
                "embedding rows must have non-zero norm");
        }
        const float inv_norm =
            static_cast<float>(1.0 / std::sqrt(norm2));
        for (std::size_t column = 0U;
             column < vocabulary.embedding_dim; ++column)
        {
            values[column] *= inv_norm;
        }
    }

    if (vocabulary.spatial_weights.empty())
    {
        vocabulary.spatial_weights.assign(
            vocabulary.predicates.size(), 0.0F);
    }
    else
    {
        for (float value : vocabulary.spatial_weights)
        {
            if (!std::isfinite(value) ||
                value < 0.0F || value > 1.0F)
            {
                throw_vocabulary_invalid(
                    "spatial weights must be finite within [0,1]");
            }
        }
    }

    return vocabulary;
}

} // namespace kfcore::relation
