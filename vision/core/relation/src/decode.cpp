#include "decode.hpp"

#include "kfcore/relation/error.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::relation::detail
{
namespace
{

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw RelationError(RelationErrorCode::InvalidArgument,
                        "relation decode: " + detail);
}

float sigmoid(float value) noexcept
{
    if (value >= 0.0F)
    {
        return 1.0F / (1.0F + std::exp(-value));
    }
    const float exp_value = std::exp(value);
    return exp_value / (1.0F + exp_value);
}

float relation_score(float predicate_logit, float pair_logit,
                     const RelateAnythingOptions& options)
{
    if (!std::isfinite(predicate_logit) || !std::isfinite(pair_logit))
    {
        throw_invalid("valid pair logits must be finite");
    }
    const float fused = options.calibration_a *
                            (predicate_logit + options.pair_weight * pair_logit) +
                        options.calibration_b;
    if (!std::isfinite(fused))
    {
        throw_invalid("fused relation logit is not finite");
    }
    return sigmoid(fused);
}

struct Candidate
{
    RelationEdge edge;
    float rank = 0.0F;
};

} // namespace

std::vector<RelationEdge>
decode_relation_outputs(const RawRelationOutputs& outputs,
                        const std::vector<Region>& regions,
                        const RelateAnythingOptions& options)
{
    if (outputs.pair_count == 0U || outputs.predicate_count == 0U)
    {
        return {};
    }
    if (outputs.pred_logits == nullptr || outputs.pair_logits == nullptr ||
        outputs.subject_indices == nullptr || outputs.object_indices == nullptr ||
        outputs.valid_mask == nullptr)
    {
        throw_invalid("output buffers must not be null");
    }
    if (outputs.predicate_count != options.predicates.size())
    {
        throw_invalid("predicate output width does not match configured vocabulary");
    }

    std::vector<Candidate> candidates;
    candidates.reserve((std::min)(outputs.pair_count, options.top_k));

    for (std::size_t pair = 0U; pair < outputs.pair_count; ++pair)
    {
        if (outputs.valid_mask[pair] == 0U)
        {
            continue;
        }

        const std::int64_t subject_raw = outputs.subject_indices[pair];
        const std::int64_t object_raw = outputs.object_indices[pair];
        if (subject_raw < 0 || object_raw < 0)
        {
            continue;
        }
        const auto subject = static_cast<std::size_t>(subject_raw);
        const auto object = static_cast<std::size_t>(object_raw);
        if (subject >= regions.size() || object >= regions.size() ||
            subject == object)
        {
            continue;
        }

        std::size_t best_predicate = 0U;
        float best_score = -1.0F;
        for (std::size_t predicate = 0U;
             predicate < outputs.predicate_count; ++predicate)
        {
            const float score = relation_score(
                outputs.pred_logits[pair * outputs.predicate_count + predicate],
                outputs.pair_logits[pair], options);
            if (score > best_score)
            {
                best_score = score;
                best_predicate = predicate;
            }
        }

        if (best_score < options.threshold)
        {
            continue;
        }

        float rank = best_score;
        if (options.weight_ranking_by_detector_score)
        {
            rank *= regions[subject].detector_score *
                    regions[object].detector_score;
        }

        RelationEdge edge;
        edge.subject_index = subject;
        edge.object_index = object;
        edge.predicate_index = best_predicate;
        edge.score = best_score;
        edge.subject_track_id = regions[subject].track_id;
        edge.object_track_id = regions[object].track_id;
        candidates.push_back({std::move(edge), rank});
    }

    std::stable_sort(candidates.begin(), candidates.end(),
        [](const Candidate& left, const Candidate& right)
        {
            return left.rank > right.rank;
        });

    if (candidates.size() > options.top_k)
    {
        candidates.resize(options.top_k);
    }

    std::vector<RelationEdge> edges;
    edges.reserve(candidates.size());
    for (Candidate& candidate : candidates)
    {
        edges.push_back(std::move(candidate.edge));
    }
    return edges;
}

} // namespace kfcore::relation::detail
