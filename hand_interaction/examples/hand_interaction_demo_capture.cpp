#include "hand_interaction_demo_capture.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace kfcore::hand_interaction::demo
{
namespace
{

bool checked_multiply(std::size_t left, std::size_t right, std::size_t& output) noexcept
{
    if (left != 0U && right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        return false;
    }
    output = left * right;
    return true;
}

} // namespace

std::optional<std::size_t> packed_frame_bytes(int width, int height, int format) noexcept
{
    if (width <= 0 || height <= 0)
    {
        return std::nullopt;
    }
    std::size_t pixels = 0U;
    if (!checked_multiply(static_cast<std::size_t>(width),
                          static_cast<std::size_t>(height), pixels))
    {
        return std::nullopt;
    }

    std::size_t bytes = 0U;
    switch (format)
    {
    case SALTS_VIDEO_CAPTURE_FORMAT_I420:
    case SALTS_VIDEO_CAPTURE_FORMAT_NV12:
        if ((width & 1) != 0 || (height & 1) != 0 || pixels >
                (std::numeric_limits<std::size_t>::max)() / 3U)
        {
            return std::nullopt;
        }
        return pixels * 3U / 2U;
    case SALTS_VIDEO_CAPTURE_FORMAT_RGB24:
        return checked_multiply(pixels, 3U, bytes) ? std::optional<std::size_t>(bytes)
                                                   : std::nullopt;
    case SALTS_VIDEO_CAPTURE_FORMAT_BGRA:
        return checked_multiply(pixels, 4U, bytes) ? std::optional<std::size_t>(bytes)
                                                   : std::nullopt;
    default:
        return std::nullopt;
    }
}

bool inference_view_compatible(int format) noexcept
{
    return format == SALTS_VIDEO_CAPTURE_FORMAT_RGB24 ||
           format == SALTS_VIDEO_CAPTURE_FORMAT_NV12 ||
           format == SALTS_VIDEO_CAPTURE_FORMAT_I420;
}

image::ImageView inference_view(const CapturedFrame& frame)
{
    const auto expected = packed_frame_bytes(frame.width, frame.height, frame.format);
    if (!inference_view_compatible(frame.format) || !expected.has_value() ||
        frame.pixels.empty() || frame.pixels.size() != *expected)
    {
        throw std::invalid_argument(
            "capture inference view requires packed RGB24, NV12, or I420 storage");
    }
    image::PixelFormat pixel_format = image::PixelFormat::Rgb8;
    if (frame.format == SALTS_VIDEO_CAPTURE_FORMAT_NV12)
    {
        pixel_format = image::PixelFormat::Nv12;
    }
    else if (frame.format == SALTS_VIDEO_CAPTURE_FORMAT_I420)
    {
        pixel_format = image::PixelFormat::I420;
    }
    const std::size_t row_stride = static_cast<std::size_t>(frame.width) *
        (frame.format == SALTS_VIDEO_CAPTURE_FORMAT_RGB24 ? 3U : 1U);
    return { frame.pixels.data(), frame.pixels.size(), frame.width, frame.height,
             row_stride, pixel_format, image::MemoryKind::Host };
}

LatestFrameMailbox::LatestFrameMailbox(std::size_t max_frame_bytes)
    : max_frame_bytes_(max_frame_bytes)
{
    if (max_frame_bytes_ == 0U)
    {
        throw std::invalid_argument("capture mailbox max_frame_bytes must be positive");
    }
    published_.pixels.reserve(max_frame_bytes_);
}

CapturedFrame LatestFrameMailbox::make_consumer_frame() const
{
    CapturedFrame result;
    result.pixels.reserve(max_frame_bytes_);
    return result;
}

bool LatestFrameMailbox::publish(const std::uint8_t* data, std::size_t size, int width,
                                 int height, int format,
                                 std::uint64_t timestamp_us) noexcept
{
    const auto expected = packed_frame_bytes(width, height, format);
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || data == nullptr || !expected.has_value() || *expected != size ||
        size > max_frame_bytes_ || published_.pixels.capacity() < max_frame_bytes_)
    {
        ++counters_.rejected_frames;
        return false;
    }

    if (has_frame_)
    {
        ++counters_.coalesced_frames;
    }
    published_.pixels.resize(size);
    std::memcpy(published_.pixels.data(), data, size);
    published_.width        = width;
    published_.height       = height;
    published_.format       = format;
    published_.timestamp_us = timestamp_us;
    published_.serial       = ++counters_.captured_frames;
    has_frame_              = true;
    ready_.notify_one();
    return true;
}

TakeStatus LatestFrameMailbox::take_latest(CapturedFrame& output,
                                           std::chrono::milliseconds timeout)
{
    if (output.pixels.capacity() < max_frame_bytes_)
    {
        throw std::invalid_argument(
            "capture consumer frame was not prepared for mailbox capacity");
    }
    if (timeout.count() < 0)
    {
        throw std::invalid_argument("capture wait timeout must be non-negative");
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (!has_frame_ && !closed_)
    {
        ready_.wait_for(lock, timeout, [&] { return has_frame_ || closed_; });
    }
    if (has_frame_)
    {
        std::swap(output, published_);
        has_frame_ = false;
        ++counters_.consumed_frames;
        return TakeStatus::Frame;
    }
    return closed_ ? TakeStatus::Closed : TakeStatus::Timeout;
}

void LatestFrameMailbox::close() noexcept
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
    }
    ready_.notify_all();
}

CaptureCounters LatestFrameMailbox::counters() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return counters_;
}

std::size_t LatestFrameMailbox::max_frame_bytes() const noexcept
{
    return max_frame_bytes_;
}

std::vector<salts_capture_device_t> list_camera_devices()
{
    std::vector<salts_capture_device_t> result(SALTS_CAPTURE_MAX_DEVICES);
    const int count = salts_capture_list_video_devices(result.data(),
                                                        static_cast<int>(result.size()));
    if (count < 0)
    {
        throw std::runtime_error("Turbo Capture failed to enumerate video devices");
    }
    result.resize(static_cast<std::size_t>(count));
    return result;
}

std::vector<salts_video_native_mode_t> list_camera_modes(const std::string& device_id)
{
    salts_video_device_t* device = nullptr;
    const int open_result = salts_video_device_open(device_id.c_str(), &device);
    if (open_result != SALTS_CAPTURE_OK || device == nullptr)
    {
        throw std::runtime_error("Turbo Capture failed to open video device: " + device_id);
    }

    try
    {
        std::size_t capacity = SALTS_CAPTURE_MAX_VIDEO_MODES;
        for (;;)
        {
            std::vector<salts_video_native_mode_t> modes(capacity);
            std::size_t                            count = 0U;
            const int list_result = salts_video_device_list_modes_all(
                device, modes.data(), modes.size(), &count);
            if (list_result != SALTS_CAPTURE_OK)
            {
                throw std::runtime_error("Turbo Capture failed to list video modes");
            }
            if (count < capacity)
            {
                modes.resize(count);
                salts_video_device_close(device);
                return modes;
            }
            if (capacity >= kMaximumListedCameraModes)
            {
                throw std::length_error("video device exposes more than the mode listing limit");
            }
            capacity = std::min(capacity * 2U, kMaximumListedCameraModes);
        }
    }
    catch (...)
    {
        salts_video_device_close(device);
        throw;
    }
}

CameraCapture::CameraCapture(const std::string& device_id,
                             const salts_video_native_mode_t& mode,
                             LatestFrameMailbox& mailbox)
    : mode_(mode)
    , mailbox_(&mailbox)
{
    salts_video_device_t* device = nullptr;
    const int open_result = salts_video_device_open(device_id.c_str(), &device);
    if (open_result != SALTS_CAPTURE_OK || device == nullptr)
    {
        throw std::runtime_error("Turbo Capture failed to open selected video device");
    }
    const int create_result = salts_video_device_create_capture(device, &mode_, &capture_);
    salts_video_device_close(device);
    if (create_result != SALTS_CAPTURE_OK || capture_ == nullptr)
    {
        throw std::runtime_error("Turbo Capture failed to create the selected video mode");
    }
    salts_video_capture_set_callback(capture_, &CameraCapture::on_frame, this);
    salts_capture_on_state(capture_, &CameraCapture::on_state);
}

CameraCapture::~CameraCapture()
{
    stop();
    salts_capture_destroy(capture_);
}

void CameraCapture::start()
{
    if (started_)
    {
        throw std::logic_error("camera capture is already running");
    }
    if (salts_capture_start(capture_) != SALTS_CAPTURE_OK)
    {
        throw std::runtime_error("Turbo Capture failed to start the selected video mode");
    }
    started_ = true;
}

void CameraCapture::stop() noexcept
{
    if (started_)
    {
        salts_capture_stop(capture_);
        started_ = false;
    }
}

void CameraCapture::on_frame(salts_capture_t*, const std::uint8_t* data, std::size_t size,
                             int width, int height, std::uint64_t timestamp,
                             void* user_data) noexcept
{
    auto* self = static_cast<CameraCapture*>(user_data);
    if (self != nullptr && self->mailbox_ != nullptr)
    {
        (void)self->mailbox_->publish(data, size, width, height, self->mode_.format,
                                      timestamp);
    }
}

void CameraCapture::on_state(salts_capture_t*, salts_capture_state_t state,
                             void* user_data) noexcept
{
    auto* self = static_cast<CameraCapture*>(user_data);
    if (state == SALTS_CAPTURE_STATE_ERROR && self != nullptr && self->mailbox_ != nullptr)
    {
        self->mailbox_->close();
    }
}

} // namespace kfcore::hand_interaction::demo
