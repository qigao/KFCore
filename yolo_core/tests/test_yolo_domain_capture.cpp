#include "yolo_domain_capture.hpp"

#include <chrono>
#include <cstdint>
#include <future>

#include "tinytest.hpp"

using namespace kfcore::yolo::demo;
using namespace std::chrono_literals;

spec("YOLOv8 domain capture mailbox")
{
    it("copies borrowed frames and coalesces unread data")
    {
        LatestFrameMailbox mailbox(16U);
        CapturedFrame output = mailbox.make_consumer_frame();
        const std::uint8_t first[] { 1U, 2U, 3U };
        const std::uint8_t second[] { 7U, 8U, 9U };
        check_true(mailbox.publish(first, sizeof(first), 1, 1,
                                   SALTS_VIDEO_CAPTURE_FORMAT_RGB24, 1U));
        check_true(mailbox.publish(second, sizeof(second), 1, 1,
                                   SALTS_VIDEO_CAPTURE_FORMAT_RGB24, 2U));
        check(mailbox.take_latest(output, 0ms) == TakeStatus::Frame);
        check(output.pixels[0] == 7U);
        check(output.serial == 2U);
        const CaptureCounters counters = mailbox.counters();
        check(counters.captured_frames == 2U);
        check(counters.consumed_frames == 1U);
        check(counters.coalesced_frames == 1U);
    }

    it("validates exact packed bytes formats dimensions and capacity")
    {
        check(packed_frame_bytes(640, 480, SALTS_VIDEO_CAPTURE_FORMAT_NV12) ==
              640U * 480U * 3U / 2U);
        check_false(packed_frame_bytes(641, 480,
                                       SALTS_VIDEO_CAPTURE_FORMAT_NV12).has_value());
        check_false(packed_frame_bytes(640, 480,
                                       SALTS_VIDEO_CAPTURE_FORMAT_MJPEG).has_value());

        LatestFrameMailbox mailbox(6U);
        const std::uint8_t bytes[] { 1U, 2U, 3U, 4U, 5U, 6U };
        check_true(mailbox.publish(bytes, sizeof(bytes), 2, 2,
                                   SALTS_VIDEO_CAPTURE_FORMAT_NV12, 0U));
        check_false(mailbox.publish(bytes, sizeof(bytes) - 1U, 2, 2,
                                    SALTS_VIDEO_CAPTURE_FORMAT_NV12, 0U));
        check_false(mailbox.publish(bytes, sizeof(bytes), 2, 1,
                                    SALTS_VIDEO_CAPTURE_FORMAT_BGRA, 0U));
    }

    it("wakes a waiting consumer on close and rejects later frames")
    {
        LatestFrameMailbox mailbox(16U);
        auto waiter = std::async(std::launch::async, [&]
        {
            CapturedFrame output = mailbox.make_consumer_frame();
            return mailbox.take_latest(output, 5s);
        });
        mailbox.close();
        check(waiter.get() == TakeStatus::Closed);
        const std::uint8_t rgb[] { 1U, 2U, 3U };
        check_false(mailbox.publish(rgb, sizeof(rgb), 1, 1,
                                    SALTS_VIDEO_CAPTURE_FORMAT_RGB24, 0U));
    }
}
