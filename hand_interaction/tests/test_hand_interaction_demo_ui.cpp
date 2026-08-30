#include "hand_interaction_demo_capture.hpp"
#include "hand_interaction_demo_frame.hpp"
#include "hand_interaction_demo_ui.hpp"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tinytest.hpp"

namespace
{

namespace demo = kfcore::hand_interaction::demo;

demo::CapturedFrame frame(int format, std::initializer_list<std::uint8_t> pixels)
{
    demo::CapturedFrame result;
    result.width  = 2;
    result.height = 2;
    result.format = format;
    result.pixels.assign(pixels);
    return result;
}

} // namespace

spec("hand interaction demo frame and UI")
{
    it("converts packed RGB and BGRA to owning BGR")
    {
        auto rgb = frame(TURBO_VIDEO_CAPTURE_FORMAT_RGB24,
                         { 255U, 0U, 0U, 0U, 255U, 0U,
                           0U, 0U, 255U, 255U, 255U, 255U });
        cv::Mat bgr = demo::to_bgr(rgb);
        check_equal(bgr.type(), CV_8UC3);
        check_equal(bgr.rows, 2);
        check_equal(bgr.cols, 2);
        check_equal(bgr.at<cv::Vec3b>(0, 0)[0], (std::uint8_t)0U);
        check_equal(bgr.at<cv::Vec3b>(0, 0)[2], (std::uint8_t)255U);
        rgb.pixels[0] = 0U;
        check_equal(bgr.at<cv::Vec3b>(0, 0)[2], (std::uint8_t)255U);

        const auto bgra = frame(TURBO_VIDEO_CAPTURE_FORMAT_BGRA,
                                { 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U,
                                  9U, 10U, 11U, 12U, 13U, 14U, 15U, 16U });
        bgr = demo::to_bgr(bgra);
        check_equal(bgr.at<cv::Vec3b>(0, 0)[0], (std::uint8_t)1U);
        check_equal(bgr.at<cv::Vec3b>(0, 0)[2], (std::uint8_t)3U);
    }

    it("converts packed NV12 and I420 black fixtures")
    {
        const auto nv12 = frame(TURBO_VIDEO_CAPTURE_FORMAT_NV12,
                                { 16U, 16U, 16U, 16U, 128U, 128U });
        const auto i420 = frame(TURBO_VIDEO_CAPTURE_FORMAT_I420,
                                { 16U, 16U, 16U, 16U, 128U, 128U });
        const cv::Mat nv12_bgr = demo::to_bgr(nv12);
        const cv::Mat i420_bgr = demo::to_bgr(i420);
        check_equal(cv::countNonZero(nv12_bgr.reshape(1)), 0);
        check_equal(cv::countNonZero(i420_bgr.reshape(1)), 0);
    }

    it("rejects malformed unsupported and empty capture frames")
    {
        auto malformed = frame(TURBO_VIDEO_CAPTURE_FORMAT_RGB24, { 1U, 2U, 3U });
        check_throws_as(demo::to_bgr(malformed), std::invalid_argument);
        malformed.pixels.clear();
        check_throws_as(demo::to_bgr(malformed), std::invalid_argument);
        malformed = frame(TURBO_VIDEO_CAPTURE_FORMAT_MJPEG,
                          { 1U, 2U, 3U, 4U, 5U, 6U });
        check_throws_as(demo::to_bgr(malformed), std::invalid_argument);
        malformed.width = 3;
        malformed.format = TURBO_VIDEO_CAPTURE_FORMAT_NV12;
        check_throws_as(demo::to_bgr(malformed), std::invalid_argument);
    }

    it("maps reset and quit keys deterministically")
    {
        check(demo::action_from_key('r') == demo::DemoAction::Reset);
        check(demo::action_from_key('R') == demo::DemoAction::Reset);
        check(demo::action_from_key('q') == demo::DemoAction::Quit);
        check(demo::action_from_key(27) == demo::DemoAction::Quit);
        check(demo::action_from_key(-1) == demo::DemoAction::None);
    }

    it("formats every processing stage with stable units")
    {
        demo::DemoMetrics metrics;
        metrics.fps                           = 24.5;
        metrics.convert_ms                    = 1.25;
        metrics.model.preprocess_ms            = 2.5;
        metrics.model.palm_inference_ms        = 3.75;
        metrics.model.landmark_inference_ms    = 4.0;
        metrics.model.classifier_inference_ms  = 5.25;
        metrics.model.appearance_ms             = 0.5;
        metrics.model.tracking_ms              = 6.5;
        metrics.model.total_ms                 = 22.0;
        metrics.face.detection_preprocess_ms   = 1.0;
        metrics.face.detection_inference_ms    = 2.0;
        metrics.face.landmark_preprocess_ms    = 3.0;
        metrics.face.landmark_inference_ms     = 4.0;
        metrics.face.total_ms                  = 10.5;
        metrics.thig_ms                        = 0.75;
        metrics.frame_ms                       = 24.0;

        const auto lines = demo::format_timing_lines(metrics);
        check_equal(lines[0],
                    std::string("FPS 24.50 | ms conv 1.25 pre 2.50 palm 3.75 land 4.00"));
        check_equal(lines[1],
                    std::string("ms cls 5.25 reid 0.50 track 6.50 model 22.00 thig 0.75 pipe 24.00"));
        check_equal(lines[2],
                    std::string("ms face-pre 1.00 face-det 2.00 mesh-pre 3.00 mesh 4.00 face 10.50"));
    }

    it("distinguishes raw model gesture from derived hand primitives")
    {
        kfcore::hand_models::HandResult hand;
        hand.track_id = 3;
        hand.gesture  = kfcore::hand_models::Gesture::Open;

        kfcore::hand_interaction::PrimitiveFrame primitives;
        for (const std::string& relation : { "Shape V", "Motion Stationary",
                                             "Direction Right", "Rotation Clockwise",
                                             "Palm Axis Horizontal", "Pose OK" })
        {
            kfcore::thig::Observation observation;
            observation.source   = { "hand", 1 };
            observation.relation = relation;
            primitives.observations.push_back(std::move(observation));
        }

        const demo::HandOverlayText text =
            demo::format_hand_overlay_text(
                { 0U, 3, 1,
                  kfcore::hand_interaction::HandIdentityAssociation::ShapeReacquired },
                hand, primitives);
        check_equal(text.identity,
                    std::string(
                        "Hand ID:1 | Track ID:3 | Match:Shape | Raw:Open"));
        check_equal(text.derived,
                    std::string("Derived:V | Motion:Stationary | Axis:Horizontal"));
        check_equal(text.dynamics,
                    std::string("Direction:Right | Rotation:Clockwise | Pose:OK"));
    }

    it("formats one bounded diagnostic snapshot for identity transitions")
    {
        kfcore::hand_models::HandFrame hands;
        kfcore::hand_models::HandResult hand;
        hand.track_id = 7;
        hand.handedness = kfcore::hand_models::Handedness::Left;
        hand.palm.box = { 10.0F, 20.0F, 40.0F, 60.0F };
        hand.palm.confidence = 0.90F;
        hand.landmark_confidence = 0.80F;
        kfcore::hand_models::HandAppearanceDescriptor appearance;
        appearance.valid_parts =
            kfcore::hand_models::hand_appearance_part_bit(
                kfcore::hand_models::HandAppearancePart::Palm) |
            kfcore::hand_models::hand_appearance_part_bit(
                kfcore::hand_models::HandAppearancePart::Pinky);
        hand.appearance = appearance;
        hands.hands.push_back(hand);

        kfcore::hand_interaction::HandInteractionFrame interaction;
        interaction.primitives.hands.push_back(
            { 0U, 7, 3,
              kfcore::hand_interaction::HandIdentityAssociation::AppearanceReacquired });

        const auto diagnostics =
            demo::make_identity_diagnostics(hands, interaction);
        check_size(diagnostics, 1U);
        check_equal(diagnostics[0].raw_track_id, 7);
        check_equal(diagnostics[0].canonical_id, 3);
        check_equal(diagnostics[0].appearance_parts, (std::uint8_t)0x21U);
        check_equal(
            demo::format_identity_diagnostic_line(42U, hands, interaction),
            std::string(
                "[ID] frame=42 hands=1 | input=0 raw=7 hand=3 match=Appearance "
                "side=Left parts=0x21 center=(30.0,50.0) scale=60.0 "
                "confidence=(0.90,0.80)"));
    }

    it("keeps a semantic action visible for the configured interval")
    {
        demo::RecentActionHistory history;
        kfcore::thig::ActionEvent grasp;
        grasp.action = "Grasp";
        grasp.source = { "hand", 1 };
        const auto started = std::chrono::steady_clock::time_point {};

        auto visible = history.update({ grasp }, started);
        check_size(visible, 1U);
        visible = history.update({}, started + std::chrono::milliseconds(1499));
        check_size(visible, 1U);
        check_equal(visible[0].action, std::string("Grasp"));
        visible = history.update({}, started + std::chrono::milliseconds(1500));
        check_empty(visible);
    }

    it("bounds and resets semantic action presentation history")
    {
        demo::RecentActionHistory history;
        const auto started = std::chrono::steady_clock::time_point {};
        std::vector<kfcore::thig::ActionEvent> actions;
        for (int index = 0; index < 5; ++index)
        {
            kfcore::thig::ActionEvent action;
            action.action = "Action " + std::to_string(index);
            action.source = { "hand", index + 1 };
            actions.push_back(std::move(action));
        }

        const auto visible = history.update(actions, started);
        check_size(visible, 4U);
        check_equal(visible.front().action, std::string("Action 1"));
        check_equal(visible.back().action, std::string("Action 4"));
        history.reset();
        check_empty(history.update({}, started));
    }

    it("rejects unusable semantic action presentation bounds")
    {
        check_throws_as(demo::RecentActionHistory(std::chrono::milliseconds(0), 4U),
                        std::invalid_argument);
        check_throws_as(demo::RecentActionHistory(std::chrono::milliseconds(1500), 0U),
                        std::invalid_argument);
    }

    it("formats THIG graph state and semantic action diagnostics")
    {
        demo::DemoThigStatus status;
        status.hand_state  = "armed";
        status.wave_state  = "armed";
        status.click_state = "ready";
        kfcore::thig::ActionEvent grasp;
        grasp.action = "Grasp";
        grasp.source = { "hand", 1 };

        check_equal(demo::format_thig_state_line(status),
                    std::string("THIG hand=armed | wave=armed | click=ready"));
        check_equal(demo::format_action_line(grasp),
                    std::string("ACTION: Grasp | Hand ID:1"));
    }

    it("mirrors the camera image without modifying its source")
    {
        cv::Mat source(240, 320, CV_8UC3, cv::Scalar(0, 0, 0));
        const cv::Vec3b marker(5U, 17U, 93U);
        source.at<cv::Vec3b>(200, 10) = marker;

        const cv::Mat output = demo::compose_overlay(source, {}, {}, {}, nullptr, {});
        const cv::Vec3b mirrored_marker = output.at<cv::Vec3b>(200, 309);
        const cv::Vec3b original_position = output.at<cv::Vec3b>(200, 10);

        check_equal(mirrored_marker[0], marker[0]);
        check_equal(mirrored_marker[1], marker[1]);
        check_equal(mirrored_marker[2], marker[2]);
        check_equal(original_position[0], (std::uint8_t)0U);
        check_equal(original_position[1], (std::uint8_t)0U);
        check_equal(original_position[2], (std::uint8_t)0U);
        check_equal(source.at<cv::Vec3b>(200, 10)[2], marker[2]);
    }

    it("draws tracked hands actions and metrics without changing the input")
    {
        cv::Mat source(240, 320, CV_8UC3, cv::Scalar(0, 0, 0));
        kfcore::hand_models::HandFrame hands;
        kfcore::hand_models::HandResult hand;
        hand.palm.box            = { 40.0F, 50.0F, 100.0F, 120.0F };
        hand.palm.confidence     = 0.95F;
        hand.landmark_confidence = 0.96F;
        hand.track_id            = 3;
        for (std::size_t index = 0U; index < hand.landmarks.size(); ++index)
        {
            hand.landmarks[index] = { 60.0F + static_cast<float>(index) * 3.0F,
                                      80.0F + static_cast<float>(index), 0.0F };
        }
        hands.hands.push_back(hand);

        kfcore::hand_interaction::HandInteractionFrame interaction;
        interaction.primitives.hands.push_back({ 0U, 3, 1 });
        kfcore::thig::ActionEvent action;
        action.action     = "Wave";
        action.source     = { "hand", 1 };
        action.confidence = 0.92F;
        interaction.actions.push_back(action);

        demo::DemoMetrics metrics;
        metrics.capture.captured_frames  = 12U;
        metrics.capture.coalesced_frames = 2U;
        metrics.model.total_ms            = 8.5;
        metrics.thig_ms                   = 0.4;
        metrics.frame_ms                  = 10.2;

        kfcore::face_models::FaceMeshFrame face;
        face.detection = kfcore::face_models::FaceDetection {
            { 160.0F, 40.0F, 100.0F, 120.0F }, 0.9F
        };
        kfcore::face_models::FaceLandmarkResult landmarks;
        landmarks.confidence = 0.95F;
        for (std::size_t index = 0U; index < landmarks.landmarks.size(); ++index)
        {
            landmarks.landmarks[index] = {
                170.0F + static_cast<float>(index % 20U),
                60.0F + static_cast<float>(index / 20U), 0.0F
            };
        }
        face.landmarks = landmarks;

        demo::DemoThigStatus status;
        status.recent_actions = interaction.actions;
        const cv::Mat output = demo::compose_overlay(
            source, hands, interaction, status, &face, metrics);
        check_equal(output.type(), CV_8UC3);
        check_equal(output.rows, source.rows);
        check_equal(output.cols, source.cols);
        check_equal(cv::countNonZero(source.reshape(1)), 0);
        check_greater(cv::countNonZero(output.reshape(1)), 0);
        const cv::Vec3b face_point = output.at<cv::Vec3b>(60, 149);
        check_equal(face_point[0], (std::uint8_t)255U);
        check_equal(face_point[1], (std::uint8_t)80U);
        check_equal(face_point[2], (std::uint8_t)180U);
    }

    it("labels appearance-based identity reacquisition")
    {
        kfcore::hand_models::HandResult hand;
        hand.track_id = 8;
        const kfcore::hand_interaction::PrimitiveFrame primitives;

        const demo::HandOverlayText text = demo::format_hand_overlay_text(
            { 0U, 8, 3,
              kfcore::hand_interaction::HandIdentityAssociation::AppearanceReacquired },
            hand, primitives);

        check_equal(text.identity,
                    std::string(
                        "Hand ID:3 | Track ID:8 | Match:Appearance | Raw:Unknown"));
    }

    it("keeps hand labels readable on both bright and dark camera frames")
    {
        kfcore::hand_models::HandFrame hands;
        kfcore::hand_models::HandResult hand;
        hand.palm.box            = { 40.0F, 180.0F, 160.0F, 120.0F };
        hand.palm.confidence     = 0.98F;
        hand.landmark_confidence = 0.98F;
        hand.track_id            = 7;
        hands.hands.push_back(hand);

        kfcore::hand_interaction::HandInteractionFrame interaction;
        interaction.primitives.hands.push_back(
            { 0U, 7, 2,
              kfcore::hand_interaction::HandIdentityAssociation::RawTrackContinuity });

        const auto count_pixels = [](const cv::Mat& image, bool bright)
        {
            std::size_t count = 0U;
            const cv::Rect label_region(425, 125, 215, 70);
            for (int y = label_region.y;
                 y < label_region.y + label_region.height; ++y)
            {
                for (int x = label_region.x;
                     x < label_region.x + label_region.width; ++x)
                {
                    const cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
                    const bool matches = bright
                        ? pixel[0] >= 230U && pixel[1] >= 230U && pixel[2] >= 230U
                        : pixel[0] <= 24U && pixel[1] <= 24U && pixel[2] <= 24U;
                    count += matches ? 1U : 0U;
                }
            }
            return count;
        };

        const cv::Mat bright_source(360, 640, CV_8UC3,
                                    cv::Scalar(255, 255, 255));
        const cv::Mat dark_source(360, 640, CV_8UC3, cv::Scalar(0, 0, 0));
        const cv::Mat on_bright = demo::compose_overlay(
            bright_source, hands, interaction, {}, nullptr, {});
        const cv::Mat on_dark = demo::compose_overlay(
            dark_source, hands, interaction, {}, nullptr, {});

        check_greater(count_pixels(on_bright, false), 10U);
        check_greater(count_pixels(on_dark, true), 10U);
    }

    it("rejects non-BGR UI input")
    {
        cv::Mat gray(32, 32, CV_8UC1, cv::Scalar(0));
        check_throws_as(demo::compose_overlay(gray, {}, {}, {}, nullptr, {}),
                        std::invalid_argument);
    }
}
