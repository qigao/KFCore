#pragma once

#include <turbo_capture.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace kfcore::yolo::demo
{

inline constexpr std::size_t kMaximumListedCameraModes = 4096U;

struct CapturedFrame
{
    std::vector<std::uint8_t> pixels;
    int                       width = 0;
    int                       height = 0;
    int                       format = TURBO_VIDEO_CAPTURE_FORMAT_NV12;
    std::uint64_t             timestamp_us = 0U;
    std::uint64_t             serial = 0U;
};

struct CaptureCounters
{
    std::uint64_t captured_frames = 0U;
    std::uint64_t consumed_frames = 0U;
    std::uint64_t coalesced_frames = 0U;
    std::uint64_t rejected_frames = 0U;
};

enum class TakeStatus
{
    Frame,
    Timeout,
    Closed,
};

[[nodiscard]] std::optional<std::size_t>
packed_frame_bytes(int width, int height, int format) noexcept;

class LatestFrameMailbox final
{
public:
    explicit LatestFrameMailbox(std::size_t max_frame_bytes);
    LatestFrameMailbox(const LatestFrameMailbox&)            = delete;
    LatestFrameMailbox& operator=(const LatestFrameMailbox&) = delete;

    [[nodiscard]] CapturedFrame make_consumer_frame() const;
    bool publish(const std::uint8_t* data, std::size_t size, int width, int height,
                 int format, std::uint64_t timestamp_us) noexcept;
    TakeStatus take_latest(CapturedFrame& output, std::chrono::milliseconds timeout);
    void close() noexcept;
    [[nodiscard]] CaptureCounters counters() const noexcept;

private:
    const std::size_t       max_frame_bytes_;
    mutable std::mutex      mutex_;
    std::condition_variable ready_;
    CapturedFrame           published_;
    CaptureCounters         counters_;
    bool                    has_frame_ = false;
    bool                    closed_ = false;
};

[[nodiscard]] std::vector<turbo_capture_device_t> list_camera_devices();
[[nodiscard]] std::vector<turbo_video_native_mode_t>
list_camera_modes(const std::string& device_id);

class CameraCapture final
{
public:
    CameraCapture(const std::string& device_id,
                  const turbo_video_native_mode_t& mode,
                  LatestFrameMailbox& mailbox);
    ~CameraCapture();
    CameraCapture(const CameraCapture&)            = delete;
    CameraCapture& operator=(const CameraCapture&) = delete;

    void start();
    void stop() noexcept;

private:
    static void on_frame(turbo_capture_t*, const std::uint8_t*, std::size_t,
                         int, int, std::uint64_t, void*) noexcept;
    static void on_state(turbo_capture_t*, turbo_capture_state_t, void*) noexcept;

    turbo_capture_t*          capture_ = nullptr;
    turbo_video_native_mode_t mode_ {};
    LatestFrameMailbox*       mailbox_ = nullptr;
    bool                      started_ = false;
};

} // namespace kfcore::yolo::demo
