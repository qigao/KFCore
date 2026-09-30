#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace kfcore::relation
{

struct Region
{
    float left = 0.0F;
    float top = 0.0F;
    float right = 0.0F;
    float bottom = 0.0F;
    float detector_score = 1.0F;
    std::optional<std::uint64_t> track_id;
};

struct RelationEdge
{
    std::size_t subject_index = 0U;
    std::size_t object_index = 0U;
    std::size_t predicate_index = 0U;
    float score = 0.0F;
    std::optional<std::uint64_t> subject_track_id;
    std::optional<std::uint64_t> object_track_id;
};

struct RelationFrame
{
    std::int32_t image_width = 0;
    std::int32_t image_height = 0;
    std::vector<RelationEdge> edges;

    // Immutable for all predicate_index values in this frame.
    // Zero means unspecified/legacy external frame; runtime relation models
    // emitted by KFCore use positive monotonically-scoped versions.
    std::uint64_t vocabulary_version = 0U;
};

} // namespace kfcore::relation
