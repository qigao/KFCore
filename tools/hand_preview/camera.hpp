#pragma once

#include <salts_capture.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <array>
#include <atomic>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace preview {
inline constexpr std::size_t kMaxFrameBytes = 64U * 1024U * 1024U;

// One callback producer, one UI consumer. Callback storage is borrowed;
// the published Mat owns its pixels. One pending frame is replaced explicitly
// (counted), never queued. Conversion and destruction happen outside the lock.
class Camera final {
public:
    static void list(int index = -1) {
        std::array<salts_capture_device_t, SALTS_CAPTURE_MAX_DEVICES> devices{};
        const int count = salts_capture_list_video_devices(devices.data(), int(devices.size()));
        if (count < 0) throw std::runtime_error("camera enumeration failed");
        for (int i = 0; i < count; ++i)
            std::cout << devices[i].index << ": " << devices[i].name << '\n';
        if (index >= 0) {
            Device device = open(index);
            const auto modes = enumerate(device.get());
            for (std::size_t i = 0; i < modes.second; ++i) {
                const auto& m = modes.first[i];
                std::cout << "mode " << i << ": " << m.width << 'x' << m.height
                          << " fps=" << salts_video_mode_fps(&m) << " format=" << m.format << '\n';
            }
        }
    }

    Camera(int index, int mode_index, std::size_t max_bytes) : max_bytes_(max_bytes) {
        device_ = open(index);
        const auto modes = enumerate(device_.get());
        if (mode_index < 0 || std::size_t(mode_index) >= modes.second)
            throw std::runtime_error("invalid mode index; use --list-modes CAMERA");
        const auto& mode = modes.first[mode_index];
        format_ = mode.format;
        salts_capture_t* handle = nullptr;
        if (salts_video_device_create_capture(device_.get(), &mode, &handle) != SALTS_CAPTURE_OK)
            throw std::runtime_error("cannot create camera capture for selected native mode");
        capture_.reset(handle);
        salts_video_capture_set_callback(capture_.get(), receive, this);
        if (salts_capture_start(capture_.get()) != SALTS_CAPTURE_OK)
            throw std::runtime_error("cannot start camera capture");
    }
    ~Camera() { capture_.reset(); }
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    bool take(cv::Mat& output, std::uint64_t& timestamp) {
        cv::Mat next;
        std::exception_ptr error;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            error = error_;
            std::swap(next, pending_);
            timestamp = timestamp_;
        }
        if (error) std::rethrow_exception(error);
        if (next.empty()) return false;
        output = std::move(next);
        return true;
    }
    std::uint64_t replaced() const { return replaced_.load(); }

private:
    using Device = std::unique_ptr<salts_video_device_t, decltype(&salts_video_device_close)>;
    struct CaptureDelete {
        void operator()(salts_capture_t* p) const { salts_capture_stop(p); salts_capture_destroy(p); }
    };
    static Device open(int index) {
        salts_video_device_t* device = nullptr;
        if (salts_video_device_open(std::to_string(index).c_str(), &device) != SALTS_CAPTURE_OK)
            throw std::runtime_error("cannot open camera " + std::to_string(index));
        return Device(device, salts_video_device_close);
    }
    static std::pair<std::array<salts_video_native_mode_t, SALTS_CAPTURE_MAX_VIDEO_MODES>, std::size_t>
    enumerate(salts_video_device_t* device) {
        std::pair<std::array<salts_video_native_mode_t, SALTS_CAPTURE_MAX_VIDEO_MODES>, std::size_t> modes{};
        if (salts_video_device_list_modes(device, modes.first.data(), modes.first.size(), &modes.second) != SALTS_CAPTURE_OK ||
            modes.second == 0 || modes.second >= modes.first.size())
            throw std::runtime_error("camera mode enumeration failed, empty or capacity exceeded");
        return modes;
    }
    static void receive(salts_capture_t*, const std::uint8_t* data, std::size_t len,
                        int width, int height, std::uint64_t timestamp, void* context) noexcept {
        auto& self = *static_cast<Camera*>(context);
        try {
            if (!data || width <= 0 || height <= 0 || len == 0 || len > self.max_bytes_ ||
                std::size_t(width) > self.max_bytes_ / 4U / std::size_t(height))
                throw std::runtime_error("camera frame exceeds byte limit or has invalid dimensions");
            cv::Mat bgr;
            const auto pixels = std::size_t(width) * height;
            const auto require_size = [&](std::size_t expected) {
                if (len != expected) throw std::runtime_error("camera frame byte count mismatch");
            };
            if (self.format_ == SALTS_VIDEO_CAPTURE_FORMAT_MJPEG) {
                bgr = cv::imdecode(cv::Mat(1, int(len), CV_8UC1, const_cast<std::uint8_t*>(data)), cv::IMREAD_COLOR);
                if (bgr.empty() || bgr.cols != width || bgr.rows != height)
                    throw std::runtime_error("MJPEG decode failed or dimensions mismatch");
            } else if (self.format_ == SALTS_VIDEO_CAPTURE_FORMAT_RGB24) {
                require_size(pixels * 3U);
                cv::cvtColor(cv::Mat(height, width, CV_8UC3, const_cast<std::uint8_t*>(data)), bgr, cv::COLOR_RGB2BGR);
            } else if (self.format_ == SALTS_VIDEO_CAPTURE_FORMAT_BGRA) {
                require_size(pixels * 4U);
                cv::cvtColor(cv::Mat(height, width, CV_8UC4, const_cast<std::uint8_t*>(data)), bgr, cv::COLOR_BGRA2BGR);
            } else if (self.format_ == SALTS_VIDEO_CAPTURE_FORMAT_I420 || self.format_ == SALTS_VIDEO_CAPTURE_FORMAT_NV12) {
                if (width % 2 || height % 2) throw std::runtime_error("YUV420 requires even dimensions");
                require_size(pixels * 3U / 2U);
                cv::cvtColor(cv::Mat(height + height / 2, width, CV_8UC1, const_cast<std::uint8_t*>(data)), bgr,
                    self.format_ == SALTS_VIDEO_CAPTURE_FORMAT_I420 ? cv::COLOR_YUV2BGR_I420 : cv::COLOR_YUV2BGR_NV12);
            } else throw std::runtime_error("unsupported camera pixel format");
            {
                std::lock_guard<std::mutex> lock(self.mutex_);
                if (!self.pending_.empty()) ++self.replaced_;
                std::swap(self.pending_, bgr);
                self.timestamp_ = timestamp;
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(self.mutex_);
            if (!self.error_) self.error_ = std::current_exception();
        }
    }
    std::mutex mutex_;
    cv::Mat pending_;
    std::exception_ptr error_;
    std::atomic<std::uint64_t> replaced_{0};
    std::uint64_t timestamp_ = 0;
    int format_ = 0;
    std::size_t max_bytes_;
    Device device_{nullptr, salts_video_device_close};
    std::unique_ptr<salts_capture_t, CaptureDelete> capture_;
};
} // namespace preview
