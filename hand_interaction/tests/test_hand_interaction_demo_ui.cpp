#include "hand_interaction_demo_capture.hpp"
#include "hand_interaction_demo_frame.hpp"
#include "hand_interaction_demo_ui.hpp"

#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

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
                    std::string("ms cls 5.25 track 6.50 model 22.00 thig 0.75 pipe 24.00"));
        check_equal(lines[2],
                    std::string("ms face-pre 1.00 face-det 2.00 mesh-pre 3.00 mesh 4.00 face 10.50"));
    }

    it("formats explicit hand, track, and gesture identities")
    {
        kfcore::vision_models::HandResult hand;
        hand.track_id = 3;
        hand.gesture  = kfcore::vision_models::Gesture::Pointer;

        check_equal(demo::format_hand_label(1, hand),
                    std::string("Hand ID:1 | Track ID:3 | Gesture:Pointer"));
    }

    it("mirrors the camera image without modifying its source")
    {
        cv::Mat source(240, 320, CV_8UC3, cv::Scalar(0, 0, 0));
        const cv::Vec3b marker(5U, 17U, 93U);
        source.at<cv::Vec3b>(200, 10) = marker;

        const cv::Mat output = demo::compose_overlay(source, {}, {}, nullptr, {});
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
        kfcore::vision_models::HandFrame hands;
        kfcore::vision_models::HandResult hand;
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

        kfcore::vision_models::FaceMeshFrame face;
        face.detection = kfcore::vision_models::FaceDetection {
            { 160.0F, 40.0F, 100.0F, 120.0F }, 0.9F
        };
        kfcore::vision_models::FaceLandmarkResult landmarks;
        landmarks.confidence = 0.95F;
        for (std::size_t index = 0U; index < landmarks.landmarks.size(); ++index)
        {
            landmarks.landmarks[index] = {
                170.0F + static_cast<float>(index % 20U),
                60.0F + static_cast<float>(index / 20U), 0.0F
            };
        }
        face.landmarks = landmarks;

        const cv::Mat output = demo::compose_overlay(
            source, hands, interaction, &face, metrics);
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

    it("rejects non-BGR UI input")
    {
        cv::Mat gray(32, 32, CV_8UC1, cv::Scalar(0));
        check_throws_as(demo::compose_overlay(gray, {}, {}, nullptr, {}),
                        std::invalid_argument);
    }
}
