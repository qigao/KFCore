#pragma once

#include "kfcore/yolo/types.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace kfcore::yolo::demo
{

enum class DomainKind
{
    Drone,
    Football,
    Parking,
};

struct DomainProfile
{
    DomainKind               kind = DomainKind::Drone;
    std::string              name;
    std::vector<std::string> class_labels;
};

struct DomainSummary
{
    std::vector<std::size_t> class_counts;
    std::size_t              confirmed_tracks = 0U;
    std::optional<double>    parking_occupancy_percent;
};

[[nodiscard]] DomainKind parse_domain_kind(std::string_view value);
[[nodiscard]] const DomainProfile& domain_profile(DomainKind kind);
[[nodiscard]] const std::string& class_label(const DomainProfile& profile,
                                              std::int32_t class_id);
[[nodiscard]] DetectionFrame filter_detections(const DetectionFrame& frame,
                                                const DomainProfile& profile,
                                                float score_threshold);
[[nodiscard]] DomainSummary summarize_tracks(const TrackFrame& frame,
                                             const DomainProfile& profile);
[[nodiscard]] std::string format_summary(const DomainProfile& profile,
                                         const DomainSummary& summary);

} // namespace kfcore::yolo::demo
