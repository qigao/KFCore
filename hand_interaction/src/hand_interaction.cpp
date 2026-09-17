#include "kfcore/hand_interaction/hand_interaction.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kfcore::hand_interaction
{
namespace
{

    constexpr char kHandEntityKind[] = "hand";

    bool is_declared_region(const std::string& relation) noexcept
    {
        return relation == "Region Center" || relation == "Region Left" ||
               relation == "Region Right" || relation == "Region Top" ||
               relation == "Region Bottom" || relation == "Region Unclassified";
    }

    void validate_external_observations(const std::vector<thig::Observation>& observations,
                                        const GestureFrameContext&            context)
    {
        std::unordered_set<std::int64_t> region_sources;
        region_sources.reserve(observations.size());
        for (const thig::Observation& observation : observations)
        {
            if (observation.serial != context.serial)
            {
                throw std::invalid_argument(
                    "external observation serial must match the frame context");
            }
            if (observation.source.kind != kHandEntityKind || observation.source.id <= 0 ||
                observation.target.has_value())
            {
                throw std::invalid_argument(
                    "external Region observations require one canonical hand source");
            }
            if (!is_declared_region(observation.relation))
            {
                throw std::invalid_argument(
                    "external observations must use a declared Region relation");
            }
            if (!std::isfinite(observation.confidence) || observation.confidence < 0.0F ||
                observation.confidence > 1.0F)
            {
                throw std::invalid_argument("external observation confidence must be within [0,1]");
            }
            if (!region_sources.insert(observation.source.id).second)
            {
                throw std::invalid_argument(
                    "external observations contain multiple Regions for one hand");
            }
        }
    }

    std::size_t maximum_primitive_observations(std::size_t hand_count)
    {
        constexpr std::size_t kObservationsPerHand = 8U;
        const std::size_t pair_count = hand_count > 1U ? hand_count * (hand_count - 1U) / 2U : 0U;
        return hand_count * kObservationsPerHand + pair_count;
    }

    std::unordered_map<int, int> raw_to_canonical(const PrimitiveFrame& frame)
    {
        std::unordered_map<int, int> result;
        result.reserve(frame.hands.size());
        for (const CanonicalHand& hand : frame.hands)
        {
            if (hand.raw_track_id < 0 || hand.canonical_id <= 0)
            {
                continue;
            }
            result.emplace(hand.raw_track_id, hand.canonical_id);
        }
        return result;
    }

    void validate_region_sources(const std::vector<thig::Observation>& external_observations,
                                 const std::unordered_set<int>&        canonical_ids)
    {
        for (const auto& observation : external_observations)
        {
            const int canonical_id = static_cast<int>(observation.source.id);
            if (canonical_ids.find(canonical_id) == canonical_ids.end())
            {
                throw std::invalid_argument(
                    "external Region observation refers to a hand absent from this frame");
            }
        }
    }

} // namespace

struct HandInteractionPipeline::Impl
{
    explicit Impl(const HandInteractionOptions& options)
        : primitives(options.primitives)
        , engine(build_hand_interaction_graph(options.temporal))
        , max_hands(options.primitives.max_hands)
    {
    }

    HandPrimitiveExtractor    primitives;
    thig::TemporalGraphEngine engine;
    std::size_t               max_hands;
};

HandInteractionPipeline::HandInteractionPipeline(const HandInteractionOptions& options)
    : impl_(std::make_unique<Impl>(options))
{
}

HandInteractionPipeline::~HandInteractionPipeline()                                  = default;
HandInteractionPipeline::HandInteractionPipeline(HandInteractionPipeline&&) noexcept = default;
HandInteractionPipeline&
HandInteractionPipeline::operator=(HandInteractionPipeline&&) noexcept = default;

HandInteractionFrame
HandInteractionPipeline::process(const hand_models::HandFrame&         frame,
                                 const GestureFrameContext&            context,
                                 const std::vector<thig::Observation>& external_observations)
{
    validate_external_observations(external_observations, context);
    if (frame.hands.size() > impl_->max_hands)
    {
        throw std::length_error("hand interaction frame exceeds max_hands");
    }
    const std::size_t maximum_observations = maximum_primitive_observations(frame.hands.size());
    if (maximum_observations > impl_->engine.Spec().maxObservationsPerFrame ||
        external_observations.size() >
            impl_->engine.Spec().maxObservationsPerFrame - maximum_observations)
    {
        throw std::length_error("hand interaction frame exceeds the observation capacity");
    }

    HandPrimitiveExtractor staged_primitives = impl_->primitives.clone();
    HandInteractionFrame   result;
    result.primitives = staged_primitives.process(frame, context);

    const auto              canonical_map = raw_to_canonical(result.primitives);
    std::unordered_set<int> canonical_ids;
    canonical_ids.reserve(canonical_map.size());
    for (const auto& entry : canonical_map)
    {
        canonical_ids.insert(entry.second);
    }
    validate_region_sources(external_observations, canonical_ids);

    result.actions =
        impl_->engine.ProcessFrame(result.primitives.observations, context.observed_at);
    result.primitives.observations.insert(result.primitives.observations.end(),
                                          external_observations.begin(),
                                          external_observations.end());

    impl_->primitives = std::move(staged_primitives);
    return result;
}

void HandInteractionPipeline::reset()
{
    impl_->primitives.reset();
    impl_->engine.Reset();
}

const thig::EngineSpec& HandInteractionPipeline::graph_spec() const
{
    return impl_->engine.Spec();
}

std::string HandInteractionPipeline::graph_state(const std::string& graph_id) const
{
    return impl_->engine.StateOf(graph_id);
}

} // namespace kfcore::hand_interaction
