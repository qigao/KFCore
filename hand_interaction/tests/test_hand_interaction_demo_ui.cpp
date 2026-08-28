#include "hand_interaction_demo_capture.hpp"
#include "hand_interaction_demo_frame.hpp"
#include "hand_interaction_demo_ui.hpp"

#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <stdexcept>

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

        const cv::Mat output = demo::compose_overlay(source, hands, interaction, metrics);
        check_equal(output.type(), CV_8UC3);
        check_equal(output.rows, source.rows);
        check_equal(output.cols, source.cols);
        check_equal(cv::countNonZero(source.reshape(1)), 0);
        check_greater(cv::countNonZero(output.reshape(1)), 0);
    }

    it("rejects non-BGR UI input")
    {
        cv::Mat gray(32, 32, CV_8UC1, cv::Scalar(0));
        check_throws_as(demo::compose_overlay(gray, {}, {}, {}), std::invalid_argument);
    }
}
