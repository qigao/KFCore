#include "yolo_domain_frame.hpp"
#include "yolo_domain_ui.hpp"

#include "tinytest.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace kfcore::yolo;
using namespace kfcore::yolo::demo;

spec("YOLOv8 domain frame UI and metrics")
{
    it("converts exact supported capture frames into owning packed BGR")
    {
        CapturedFrame rgb;
        rgb.width = 1;
        rgb.height = 1;
        rgb.format = SALTS_VIDEO_CAPTURE_FORMAT_RGB24;
        rgb.pixels = { 10U, 20U, 30U };
        kfcore::image::BgrImage bgr = to_bgr(rgb, 1024U);
        check(bgr.height == 1);
        check(bgr.width == 1);
        check(bgr.pixels[0] == 30U);
        rgb.pixels[0] = 99U;
        check(bgr.pixels[2] == 10U);

        CapturedFrame nv12;
        nv12.width = 2;
        nv12.height = 2;
        nv12.format = SALTS_VIDEO_CAPTURE_FORMAT_NV12;
        nv12.pixels = { 16U, 16U, 16U, 16U, 128U, 128U };
        const kfcore::image::BgrImage nv12_bgr = to_bgr(nv12, 1024U);
        check(nv12_bgr.height == 2);
        check(nv12_bgr.width == 2);

        nv12.pixels.pop_back();
        check_throws_as(to_bgr(nv12, 1024U), std::invalid_argument);
    }

    it("round trips a bounded PNG without OpenCV")
    {
        kfcore::image::BgrImage original;
        original.width = 2;
        original.height = 2;
        original.pixels = {
            0, 0, 255, 0, 255, 0,
            255, 0, 0, 255, 255, 255,
        };
        char* path = tt_make_temp_file("kfcore-yolo-domain", ".png");
        check_not_null(path);
        save_bgr(path, original, 1024U);
        const kfcore::image::BgrImage decoded = load_bgr(path, 1024U, 1024U);
        check(decoded.width == original.width);
        check(decoded.height == original.height);
        check_eq_container(decoded.pixels, original.pixels);
        check(tt_remove_file(path) == 0);
        std::free(path);
    }

    it("keeps bounded timing samples and computes reproducible percentiles")
    {
        TimingWindow window(4U);
        window.add({ 1, 2, 10, 1, 2, 16 });
        window.add({ 2, 3, 20, 2, 3, 30 });
        window.add({ 3, 4, 30, 3, 4, 44 });
        window.add({ 4, 5, 40, 4, 5, 58 });
        MetricsSnapshot snapshot = window.snapshot();
        check(snapshot.sample_count == 4U);
        check(std::fabs(snapshot.detect.current_ms - 40.0) < 1.0e-9);
        check(std::fabs(snapshot.detect.mean_ms - 25.0) < 1.0e-9);
        check(std::fabs(snapshot.detect.p50_ms - 20.0) < 1.0e-9);
        check(std::fabs(snapshot.detect.p95_ms - 40.0) < 1.0e-9);

        window.add({ 5, 6, 50, 5, 6, 72 });
        snapshot = window.snapshot();
        check(window.size() == 4U);
        check(std::fabs(snapshot.detect.mean_ms - 35.0) < 1.0e-9);
        const std::string formatted = format_metrics(snapshot);
        check(formatted.find("capture=5.00/3.50/3.00/5.00ms") != std::string::npos);
        check(formatted.find("detect=50.00/35.00/30.00/50.00ms") !=
              std::string::npos);
        check(formatted.find("render=6.00/4.50/4.00/6.00ms") != std::string::npos);

        const double previous_mean = snapshot.detect.mean_ms;
        update_current_metrics(snapshot, { 9, 8, 77, 6, 5, 105 }, 5U);
        check(snapshot.sample_count == 5U);
        check(std::fabs(snapshot.detect.current_ms - 77.0) < 1.0e-9);
        check(std::fabs(snapshot.total.current_ms - 105.0) < 1.0e-9);
        check(std::fabs(snapshot.detect.mean_ms - previous_mean) < 1.0e-9);
        check_throws_as(update_current_metrics(
                            snapshot, { 1, 2, 3, 4, 5, 15 }, 0U),
                        std::invalid_argument);
        check_throws_as(TimingWindow(0U), std::invalid_argument);
    }

    it("formats semantic high-contrast labels and decodes controls")
    {
        const TrackedDetection tracked {
            { { 10.0F, 20.0F, 80.0F, 100.0F }, 0.9F, 2 }, 42U
        };
        const DomainProfile& profile = domain_profile(DomainKind::Football);
        check(track_label(profile, tracked) == "player 90.0% track=42");
        check(decode_key('q') == KeyAction::Quit);
        check(decode_key(27) == KeyAction::Quit);
        check(decode_key('R') == KeyAction::Reset);
        check(decode_key(-1) == KeyAction::None);
    }

    it("draws readable overlays without mutating tracking facts")
    {
        kfcore::image::BgrImage image;
        image.width = 640;
        image.height = 240;
        image.pixels.assign(640U * 240U * 3U, 220U);
        const std::vector<std::uint8_t> before = image.pixels;
        TrackFrame tracks {
            640, 240,
            { { { { 100.0F, 80.0F, 240.0F, 200.0F }, 0.75F, 1 }, 9U } }
        };
        const TrackFrame original = tracks;
        const DomainProfile& profile = domain_profile(DomainKind::Parking);
        const DomainSummary summary = summarize_tracks(tracks, profile);
        OverlayState state;
        state.profile = &profile;
        state.tracks = &tracks;
        state.summary = &summary;
        state.backend = "cpu";
        state.model_load_ms = 12.5;
        state.fps = 30.0;
        draw_overlay(image, state);
        check(image.pixels != before);
        check(tracks.detections.size() == original.detections.size());
        check(tracks.detections[0].track_id == original.detections[0].track_id);
        const std::size_t header_offset = (4U * 640U + 4U) * 3U;
        check(image.pixels[header_offset] < 80U);
        check(image.pixels[header_offset + 1U] < 80U);
        check(image.pixels[header_offset + 2U] < 80U);
    }
}
