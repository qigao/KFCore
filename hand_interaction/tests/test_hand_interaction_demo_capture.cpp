#include "hand_interaction_demo_capture.hpp"

#include <chrono>
#include <cstdint>
#include <future>
#include <vector>

#include "tinytest.hpp"

namespace
{

namespace demo = kfcore::hand_interaction::demo;
using namespace std::chrono_literals;

} // namespace

spec("hand interaction demo capture mailbox")
{
    it("copies one borrowed frame and transfers owned storage")
    {
        demo::LatestFrameMailbox mailbox(16U);
        demo::CapturedFrame      output = mailbox.make_consumer_frame();
        const std::uint8_t       input[] { 1U, 2U, 3U, 4U, 5U, 6U };

        check_true(mailbox.publish(input, sizeof(input), 2, 1,
                                   TURBO_VIDEO_CAPTURE_FORMAT_RGB24, 123U));
        check(mailbox.take_latest(output, 0ms) == demo::TakeStatus::Frame);
        check_equal(output.pixels.size(), sizeof(input));
        check_equal(output.pixels[4], (std::uint8_t)5U);
        check_equal(output.serial, (std::uint64_t)1U);
        check_equal(output.timestamp_us, (std::uint64_t)123U);
        check_equal(output.width, 2);
    }

    it("coalesces unread frames and exposes monotonic accepted serials")
    {
        demo::LatestFrameMailbox mailbox(16U);
        demo::CapturedFrame      output = mailbox.make_consumer_frame();
        const std::uint8_t       first[] { 1U, 2U, 3U };
        const std::uint8_t       second[] { 7U, 8U, 9U };

        check_true(mailbox.publish(first, sizeof(first), 1, 1,
                                   TURBO_VIDEO_CAPTURE_FORMAT_RGB24, 1U));
        check_true(mailbox.publish(second, sizeof(second), 1, 1,
                                   TURBO_VIDEO_CAPTURE_FORMAT_RGB24, 2U));
        check(mailbox.take_latest(output, 0ms) == demo::TakeStatus::Frame);
        check_equal(output.pixels[0], (std::uint8_t)7U);
        check_equal(output.serial, (std::uint64_t)2U);

        const auto counters = mailbox.counters();
        check_equal(counters.captured_frames, (std::uint64_t)2U);
        check_equal(counters.consumed_frames, (std::uint64_t)1U);
        check_equal(counters.coalesced_frames, (std::uint64_t)1U);
        check_equal(counters.rejected_frames, (std::uint64_t)0U);
    }

    it("accepts the exact capacity and rejects oversize or malformed frames")
    {
        demo::LatestFrameMailbox mailbox(6U);
        const std::uint8_t       exact[] { 1U, 2U, 3U, 4U, 5U, 6U };
        const std::uint8_t       oversized[] { 1U, 2U, 3U, 4U, 5U, 6U, 7U };

        check_true(mailbox.publish(exact, sizeof(exact), 2, 1,
                                   TURBO_VIDEO_CAPTURE_FORMAT_RGB24, 0U));
        check_false(mailbox.publish(oversized, sizeof(oversized), 2, 1,
                                    TURBO_VIDEO_CAPTURE_FORMAT_RGB24, 0U));
        check_false(mailbox.publish(exact, sizeof(exact) - 1U, 2, 1,
                                    TURBO_VIDEO_CAPTURE_FORMAT_RGB24, 0U));
        check_false(mailbox.publish(exact, sizeof(exact), 2, 1,
                                    TURBO_VIDEO_CAPTURE_FORMAT_MJPEG, 0U));
        check_false(mailbox.publish(nullptr, sizeof(exact), 2, 1,
                                    TURBO_VIDEO_CAPTURE_FORMAT_RGB24, 0U));
        check_equal(mailbox.counters().rejected_frames, (std::uint64_t)4U);
    }

    it("validates packed YUV dimensions and lengths")
    {
        demo::LatestFrameMailbox mailbox(16U);
        const std::uint8_t       yuv[] { 1U, 2U, 3U, 4U, 5U, 6U };
        check_true(mailbox.publish(yuv, sizeof(yuv), 2, 2,
                                   TURBO_VIDEO_CAPTURE_FORMAT_NV12, 0U));
        check_false(mailbox.publish(yuv, sizeof(yuv), 3, 2,
                                    TURBO_VIDEO_CAPTURE_FORMAT_I420, 0U));
    }

    it("returns timeout when no frame is available")
    {
        demo::LatestFrameMailbox mailbox(16U);
        demo::CapturedFrame      output = mailbox.make_consumer_frame();
        check(mailbox.take_latest(output, 1ms) == demo::TakeStatus::Timeout);
    }

    it("wakes a waiting consumer when closed and rejects later publication")
    {
        demo::LatestFrameMailbox mailbox(16U);
        auto waiter = std::async(std::launch::async, [&]
        {
            auto output = mailbox.make_consumer_frame();
            return mailbox.take_latest(output, 5s);
        });
        mailbox.close();
        check(waiter.get() == demo::TakeStatus::Closed);
        const std::uint8_t rgb[] { 1U, 2U, 3U };
        check_false(mailbox.publish(rgb, sizeof(rgb), 1, 1,
                                    TURBO_VIDEO_CAPTURE_FORMAT_RGB24, 0U));
    }

    it("requires consumer storage prepared to the same hard limit")
    {
        demo::LatestFrameMailbox mailbox(16U);
        demo::CapturedFrame      output;
        check_throws_as(mailbox.take_latest(output, 0ms), std::invalid_argument);
    }
}
