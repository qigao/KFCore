#include "yolo_domain_profile.hpp"

#include "kfcore/yolo/error.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>

namespace kfcore::yolo::demo
{
namespace
{

[[noreturn]] void throw_invalid(const std::string& message)
{
    throw YoloError(YoloErrorCode::InvalidArgument, "domain application: " + message);
}

[[noreturn]] void throw_contract(const std::string& message)
{
    throw YoloError(YoloErrorCode::EngineContractMismatch,
                    "domain model contract: " + message);
}

void validate_profile(const DomainProfile& profile)
{
    if (profile.name.empty() || profile.class_labels.empty())
    {
        throw_invalid("profile name and class labels must not be empty");
    }
}

void validate_frame_dimensions(std::int32_t width, std::int32_t height)
{
    if (width <= 0 || height <= 0)
    {
        throw_invalid("frame dimensions must be positive");
    }
}

} // namespace

DomainKind parse_domain_kind(std::string_view value)
{
    if (value == "drone")
    {
        return DomainKind::Drone;
    }
    if (value == "football")
    {
        return DomainKind::Football;
    }
    if (value == "parking")
    {
        return DomainKind::Parking;
    }
    throw_invalid("--application must be drone, football, or parking");
}

const DomainProfile& domain_profile(DomainKind kind)
{
    static const DomainProfile drone {
        DomainKind::Drone, "drone", { "class-0", "class-1" }
    };
    static const DomainProfile football {
        DomainKind::Football, "football", { "ball", "goalkeeper", "player", "referee" }
    };
    static const DomainProfile parking {
        DomainKind::Parking, "parking", { "space-empty", "space-occupied" }
    };
    switch (kind)
    {
    case DomainKind::Drone:
        return drone;
    case DomainKind::Football:
        return football;
    case DomainKind::Parking:
        return parking;
    }
    throw_invalid("application kind is invalid");
}

const std::string& class_label(const DomainProfile& profile, std::int32_t class_id)
{
    validate_profile(profile);
    if (class_id < 0 || static_cast<std::size_t>(class_id) >= profile.class_labels.size())
    {
        throw_contract("class identifier is outside the selected application profile");
    }
    return profile.class_labels[static_cast<std::size_t>(class_id)];
}

DetectionFrame filter_detections(const DetectionFrame& frame,
                                 const DomainProfile& profile,
                                 float score_threshold)
{
    validate_frame_dimensions(frame.image_width, frame.image_height);
    validate_profile(profile);
    if (!std::isfinite(score_threshold) || score_threshold < 0.0F ||
        score_threshold > 1.0F)
    {
        throw_invalid("score threshold must be finite within [0,1]");
    }

    DetectionFrame result { frame.image_width, frame.image_height, {} };
    result.detections.reserve(frame.detections.size());
    for (const Detection& detection : frame.detections)
    {
        (void)class_label(profile, detection.class_id);
        if (!std::isfinite(detection.score) || detection.score < 0.0F ||
            detection.score > 1.0F)
        {
            throw_contract("detection score is outside [0,1]");
        }
        if (detection.score >= score_threshold)
        {
            result.detections.push_back(detection);
        }
    }
    return result;
}

DomainSummary summarize_tracks(const TrackFrame& frame, const DomainProfile& profile)
{
    validate_frame_dimensions(frame.image_width, frame.image_height);
    validate_profile(profile);
    DomainSummary result;
    result.class_counts.assign(profile.class_labels.size(), 0U);
    for (const TrackedDetection& tracked : frame.detections)
    {
        const std::int32_t class_id = tracked.detection.class_id;
        (void)class_label(profile, class_id);
        ++result.class_counts[static_cast<std::size_t>(class_id)];
        if (tracked.track_id.has_value())
        {
            ++result.confirmed_tracks;
        }
    }
    if (profile.kind == DomainKind::Parking)
    {
        const std::size_t total = result.class_counts[0] + result.class_counts[1];
        if (total != 0U)
        {
            result.parking_occupancy_percent =
                100.0 * static_cast<double>(result.class_counts[1]) /
                static_cast<double>(total);
        }
    }
    return result;
}

std::string format_summary(const DomainProfile& profile, const DomainSummary& summary)
{
    validate_profile(profile);
    if (summary.class_counts.size() != profile.class_labels.size())
    {
        throw_invalid("summary class count does not match the selected profile");
    }
    std::ostringstream output;
    for (std::size_t index = 0; index < profile.class_labels.size(); ++index)
    {
        if (index != 0U)
        {
            output << ' ';
        }
        output << profile.class_labels[index] << '=' << summary.class_counts[index];
    }
    if (summary.parking_occupancy_percent.has_value())
    {
        output << " occupancy=" << std::fixed << std::setprecision(1)
               << *summary.parking_occupancy_percent << '%';
    }
    output << " tracked=" << summary.confirmed_tracks;
    return output.str();
}

} // namespace kfcore::yolo::demo
