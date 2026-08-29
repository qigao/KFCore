#include "yolo_domain_profile.hpp"

#include "kfcore/yolo/error.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <cstdint>
#include <string>

using namespace kfcore::yolo;
using namespace kfcore::yolo::demo;

namespace
{

template <typename Operation>
void check_error(const Operation& operation, YoloErrorCode code,
                 const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const YoloError& error)
    {
        threw = true;
        check(error.code() == code);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

TrackedDetection tracked(std::int32_t class_id, float score,
                         std::optional<std::uint64_t> id)
{
    return { { { 1.0F, 2.0F, 10.0F, 12.0F }, score, class_id }, id };
}

} // namespace

spec("YOLOv8 domain profiles")
{
    it("defines exact drone football and parking class contracts")
    {
        const DomainProfile& drone = domain_profile(parse_domain_kind("drone"));
        check(drone.class_labels.size() == 2U);
        check(drone.class_labels[0] == "class-0");
        check(drone.class_labels[1] == "class-1");

        const DomainProfile& football = domain_profile(parse_domain_kind("football"));
        check(football.class_labels.size() == 4U);
        check(football.class_labels[0] == "ball");
        check(football.class_labels[1] == "goalkeeper");
        check(football.class_labels[2] == "player");
        check(football.class_labels[3] == "referee");

        const DomainProfile& parking = domain_profile(parse_domain_kind("parking"));
        check(parking.class_labels.size() == 2U);
        check(parking.class_labels[0] == "space-empty");
        check(parking.class_labels[1] == "space-occupied");
    }

    it("rejects unknown application names and class identifiers")
    {
        check_error([&] { (void)parse_domain_kind("face"); },
                    YoloErrorCode::InvalidArgument, "application");
        check_error([&] { (void)class_label(domain_profile(DomainKind::Parking), -1); },
                    YoloErrorCode::EngineContractMismatch, "class identifier");
        check_error([&] { (void)class_label(domain_profile(DomainKind::Parking), 2); },
                    YoloErrorCode::EngineContractMismatch, "class identifier");
    }

    it("filters scores while preserving frame geometry and validates model classes")
    {
        DetectionFrame input { 640, 480,
                               { { { 0.0F, 0.0F, 20.0F, 20.0F }, 0.24F, 0 },
                                 { { 2.0F, 3.0F, 30.0F, 40.0F }, 0.25F, 1 } } };
        const DetectionFrame output = filter_detections(
            input, domain_profile(DomainKind::Parking), 0.25F);
        check(output.image_width == 640);
        check(output.image_height == 480);
        check(output.detections.size() == 1U);
        check(output.detections[0].class_id == 1);

        input.detections[0].class_id = 2;
        check_error([&] {
            (void)filter_detections(input, domain_profile(DomainKind::Parking), 0.0F);
        }, YoloErrorCode::EngineContractMismatch, "class identifier");
        check_error([&] {
            (void)filter_detections(input, domain_profile(DomainKind::Parking), 1.1F);
        }, YoloErrorCode::InvalidArgument, "score threshold");
    }

    it("summarizes football tracks and confirmed identifiers")
    {
        TrackFrame frame { 960, 540,
                           { tracked(2, 0.9F, 10U), tracked(0, 0.8F, 11U),
                             tracked(2, 0.7F, std::nullopt), tracked(3, 0.6F, 12U) } };
        const DomainProfile& profile = domain_profile(DomainKind::Football);
        const DomainSummary summary = summarize_tracks(frame, profile);
        check(summary.class_counts.size() == 4U);
        check(summary.class_counts[0] == 1U);
        check(summary.class_counts[1] == 0U);
        check(summary.class_counts[2] == 2U);
        check(summary.class_counts[3] == 1U);
        check(summary.confirmed_tracks == 3U);
        check_false(summary.parking_occupancy_percent.has_value());
        check(format_summary(profile, summary) ==
              "ball=1 goalkeeper=0 player=2 referee=1 tracked=3");
    }

    it("computes parking occupancy from the current detection fact source")
    {
        TrackFrame frame { 640, 480,
                           { tracked(0, 0.9F, 1U), tracked(1, 0.8F, 2U),
                             tracked(1, 0.7F, 3U) } };
        const DomainProfile& profile = domain_profile(DomainKind::Parking);
        const DomainSummary summary = summarize_tracks(frame, profile);
        check(summary.parking_occupancy_percent.has_value());
        check(std::fabs(*summary.parking_occupancy_percent - 200.0 / 3.0) < 1.0e-6);
        check(format_summary(profile, summary) ==
              "space-empty=1 space-occupied=2 occupancy=66.7% tracked=3");

        const DomainSummary empty = summarize_tracks({ 640, 480, {} }, profile);
        check_false(empty.parking_occupancy_percent.has_value());
    }
}
