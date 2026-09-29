#pragma once

#include "kfcore/scene_interaction/interaction.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kfcore::scene_interaction::detail
{

struct PairObservation
{
    PairKey pair;
    std::int32_t subject_class_id = -1;
    std::int32_t object_class_id = -1;
    std::vector<float> values;
};

[[nodiscard]] std::size_t scene_input_size(std::size_t predicate_count);

[[nodiscard]] std::vector<PairObservation>
encode_pair_observations(const pipelines::SceneGraphFrame& frame,
                         std::size_t predicate_count);

} // namespace kfcore::scene_interaction::detail
