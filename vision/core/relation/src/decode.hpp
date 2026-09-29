#pragma once

#include "kfcore/relation/relate_anything.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kfcore::relation::detail
{

struct RawRelationOutputs
{
    const float* pred_logits = nullptr;
    const float* pair_logits = nullptr;
    const std::int64_t* subject_indices = nullptr;
    const std::int64_t* object_indices = nullptr;
    const std::uint8_t* valid_mask = nullptr;
    std::size_t pair_count = 0U;
    std::size_t predicate_count = 0U;
};

[[nodiscard]] std::vector<RelationEdge>
decode_relation_outputs(const RawRelationOutputs& outputs,
                        const std::vector<Region>& regions,
                        const RelateAnythingOptions& options);

} // namespace kfcore::relation::detail
