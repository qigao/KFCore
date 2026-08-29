#include "kfcore/vision_models/tensorrt.hpp"
#include "tensorrt_image_source.hpp"
#include "tinytest.hpp"

#include <cstdint>
#include <array>
#include <functional>
#include <string>
#include <type_traits>

using namespace kfcore::vision_models;

namespace
{

void check_error(const std::function<void()>& operation, VisionModelErrorCode code,
                 const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const VisionModelError& error)
    {
        threw = true;
        check(error.code() == code);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

static_assert(!std::is_copy_constructible_v<TensorRtHandBackend>);
static_assert(!std::is_copy_assignable_v<TensorRtHandBackend>);
static_assert(!std::is_copy_constructible_v<TensorRtVisionInput>);
static_assert(!std::is_copy_assignable_v<TensorRtVisionInput>);
static_assert(!std::is_copy_constructible_v<TensorRtFaceDetector>);
static_assert(!std::is_copy_assignable_v<TensorRtFaceDetector>);
static_assert(!std::is_copy_constructible_v<TensorRtFaceLandmarker>);
static_assert(!std::is_copy_assignable_v<TensorRtFaceLandmarker>);

spec("TensorRT vision model public validation")
{
    it("borrows CUDA frame input without invoking the staging operation")
    {
        static const std::uint8_t device_pixel[3] = { 1U, 2U, 3U };
        const kfcore::image::ImageView input {
            device_pixel, sizeof(device_pixel), 1, 1, 3,
            kfcore::image::PixelFormat::Bgr8,
            kfcore::image::MemoryKind::CudaDevice
        };
        std::size_t stage_calls = 0U;

        const auto actual = detail::stage_host_or_borrow_cuda(
            input, [&](const kfcore::image::ImageView&) {
                ++stage_calls;
                return kfcore::image::ImageView {};
            });

        check_true(stage_calls == 0U);
        check_true(actual.data == input.data);
        check(actual.memory_kind == kfcore::image::MemoryKind::CudaDevice);
    }

    it("stages host frame input exactly once")
    {
        static const std::uint8_t host_pixel[3] = { 1U, 2U, 3U };
        static const std::uint8_t device_pixel[3] = { 4U, 5U, 6U };
        const kfcore::image::ImageView input {
            host_pixel, sizeof(host_pixel), 1, 1, 3,
            kfcore::image::PixelFormat::Bgr8,
            kfcore::image::MemoryKind::Host
        };
        const kfcore::image::ImageView staged {
            device_pixel, sizeof(device_pixel), 1, 1, 3,
            kfcore::image::PixelFormat::Bgr8,
            kfcore::image::MemoryKind::CudaDevice
        };
        std::size_t stage_calls = 0U;

        const auto actual = detail::stage_host_or_borrow_cuda(
            input, [&](const kfcore::image::ImageView& observed) {
                ++stage_calls;
                check_true(observed.data == input.data);
                return staged;
            });

        check_true(stage_calls == 1U);
        check_true(actual.data == staged.data);
    }

    it("preserves packed NV12 metadata in one prepared CUDA frame")
    {
        const std::array<std::uint8_t, 6U> nv12 = {
            16U, 81U, 145U, 235U, 128U, 128U,
        };
        const kfcore::image::ImageView source {
            nv12.data(), nv12.size(), 2, 2, 2,
            kfcore::image::PixelFormat::Nv12,
            kfcore::image::MemoryKind::Host,
        };
        auto preparer = TensorRtVisionInput::create();
        const VisionFrameView prepared = preparer->prepare(source);
        check(prepared.source.data == source.data);
        check(prepared.source.pixel_format == kfcore::image::PixelFormat::Nv12);
        check(prepared.compute.data != source.data);
        check(prepared.compute.pixel_format == kfcore::image::PixelFormat::Nv12);
        check(prepared.compute.memory_kind == kfcore::image::MemoryKind::CudaDevice);
        check(prepared.compute.byte_size == nv12.size());
        check(prepared.compute.row_stride == (std::size_t)2U);
    }

    it("rejects missing hand engine paths")
    {
        check_error([&] { (void)TensorRtHandBackend::load({}, {}); },
                    VisionModelErrorCode::InvalidModelAsset, "Palm");
    }

    it("rejects missing face landmark engine paths")
    {
        check_error([&] { (void)TensorRtFaceLandmarker::load({}, {}); },
                    VisionModelErrorCode::InvalidModelAsset, "face landmark");
    }

    it("rejects missing face detector engine paths")
    {
        check_error([&] { (void)TensorRtFaceDetector::load({}, {}); },
                    VisionModelErrorCode::InvalidModelAsset, "face detector");
    }

    it("rejects invalid face detector thresholds before reading engines")
    {
        TensorRtVisionOptions options;
        options.face_detection_score_threshold = 1.01F;
        check_error([&] { (void)TensorRtFaceDetector::load("face.engine", options); },
                    VisionModelErrorCode::InvalidArgument, "face detection score");
    }

    it("rejects zero CUDA resource limits before reading engines")
    {
        HandTensorRtEnginePaths paths { "palm.engine", "hand.engine", "gesture.engine" };
        TensorRtVisionOptions options;
        options.max_engine_bytes = 0;
        check_error([&] { (void)TensorRtHandBackend::load(paths, options); },
                    VisionModelErrorCode::ResourceLimitExceeded, "positive");
    }
}
