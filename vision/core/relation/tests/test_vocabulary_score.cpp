#include "vocabulary_score.hpp"

#include "kfcore/relation/error.hpp"
#include "kfcore/relation/open_vocabulary_relation.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

using namespace kfcore::relation;

spec("open vocabulary relation scoring")
{
    it("keeps generic default and exposes explicit Apache 40-box options")
    {
        const OpenVocabularyRelationOptions legacy;
        const auto apache =
            apache_released_open_vocabulary_relation_options();

        check(legacy.max_boxes == kLegacyRelationMaxBoxes);
        check(apache.max_boxes == kApacheReleasedMaxBoxes);
    }

    it("scores multiple vocabulary sizes without changing query shape")
    {
        const float semantic[] = {1.0F, 0.0F};
        const float spatial[] = {0.0F, 1.0F};
        const std::uint8_t valid[] = {1U};
        const detail::RawOpenVocabularyQueries queries {
            semantic, spatial, valid, 1U, 2U
        };

        const float one_embedding[] = {1.0F, 0.0F};
        const float one_alpha[] = {0.0F};
        float one_output[] = {0.0F};
        detail::score_open_vocabulary_queries(
            queries,
            {one_embedding, one_alpha, 1U, 2U},
            2.0F, 0.5F, one_output);
        check(std::fabs(one_output[0] - 2.5F) < 1.0e-6F);

        const float three_embeddings[] = {
            1.0F, 0.0F,
            0.0F, 1.0F,
            -1.0F, 0.0F,
        };
        const float three_alpha[] = {0.0F, 1.0F, 0.0F};
        float three_output[] = {0.0F, 0.0F, 0.0F};
        detail::score_open_vocabulary_queries(
            queries,
            {three_embeddings, three_alpha, 3U, 2U},
            1.0F, 0.0F, three_output);

        check(std::fabs(three_output[0] - 1.0F) < 1.0e-6F);
        check(std::fabs(three_output[1] - 1.0F) < 1.0e-6F);
        check(std::fabs(three_output[2] + 1.0F) < 1.0e-6F);
    }

    it("normalizes vocabulary rows and defaults routing to semantic")
    {
        PredicateVocabulary vocabulary;
        vocabulary.predicates = {"riding", "beside"};
        vocabulary.embedding_dim = 2U;
        vocabulary.embeddings = {
            3.0F, 4.0F,
            0.0F, 2.0F,
        };

        const auto normalized =
            normalize_predicate_vocabulary(
                std::move(vocabulary), 2U, 1024U);

        check(normalized.spatial_weights.size() == std::size_t{2U});
        check(normalized.spatial_weights[0] == 0.0F);
        check(normalized.spatial_weights[1] == 0.0F);
        check(std::fabs(normalized.embeddings[0] - 0.6F) < 1.0e-6F);
        check(std::fabs(normalized.embeddings[1] - 0.8F) < 1.0e-6F);
        check(std::fabs(normalized.embeddings[2] - 0.0F) < 1.0e-6F);
        check(std::fabs(normalized.embeddings[3] - 1.0F) < 1.0e-6F);
    }

    it("rejects invalid vocabulary contracts")
    {
        PredicateVocabulary duplicate;
        duplicate.predicates = {"on", "on"};
        duplicate.embedding_dim = 2U;
        duplicate.embeddings = {1.0F, 0.0F, 0.0F, 1.0F};
        check_throws_as(
            normalize_predicate_vocabulary(
                duplicate, 2U, 1024U),
            RelationError);

        PredicateVocabulary wrong_dim;
        wrong_dim.predicates = {"on"};
        wrong_dim.embedding_dim = 3U;
        wrong_dim.embeddings = {1.0F, 0.0F, 0.0F};
        check_throws_as(
            normalize_predicate_vocabulary(
                wrong_dim, 2U, 1024U),
            RelationError);

        PredicateVocabulary nonfinite;
        nonfinite.predicates = {"on"};
        nonfinite.embedding_dim = 2U;
        nonfinite.embeddings = {
            std::numeric_limits<float>::infinity(), 0.0F
        };
        check_throws_as(
            normalize_predicate_vocabulary(
                nonfinite, 2U, 1024U),
            RelationError);

        PredicateVocabulary bad_alpha;
        bad_alpha.predicates = {"on"};
        bad_alpha.embedding_dim = 2U;
        bad_alpha.embeddings = {1.0F, 0.0F};
        bad_alpha.spatial_weights = {1.5F};
        check_throws_as(
            normalize_predicate_vocabulary(
                bad_alpha, 2U, 1024U),
            RelationError);

        PredicateVocabulary too_large;
        too_large.predicates = {"on"};
        too_large.embedding_dim = 2U;
        too_large.embeddings = {1.0F, 0.0F};
        check_throws_as(
            normalize_predicate_vocabulary(
                too_large, 2U, 1U),
            RelationError);
    }

    it("counts default spatial weights against the vocabulary byte limit")
    {
        PredicateVocabulary vocabulary;
        vocabulary.predicates = {"x"};
        vocabulary.embedding_dim = 1U;
        vocabulary.embeddings = {1.0F};

        check_throws_as(
            normalize_predicate_vocabulary(vocabulary, 1U, 8U),
            RelationError);
        const auto normalized =
            normalize_predicate_vocabulary(std::move(vocabulary), 1U, 9U);
        check(normalized.spatial_weights.size() == std::size_t{1U});
    }

    it("zeros invalid pair logits before decoding")
    {
        const float semantic[] = {1.0F, 0.0F};
        const float spatial[] = {0.0F, 1.0F};
        const float embedding[] = {1.0F, 0.0F};
        const float alpha[] = {0.0F};
        const std::uint8_t valid[] = {0U};
        float output[] = {99.0F};

        detail::score_open_vocabulary_queries(
            {semantic, spatial, valid, 1U, 2U},
            {embedding, alpha, 1U, 2U},
            1.0F, 0.0F, output);

        check(output[0] == 0.0F);
    }
}
