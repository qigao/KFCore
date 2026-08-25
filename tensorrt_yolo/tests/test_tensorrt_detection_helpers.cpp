#include "detector_helpers.hpp"
#include "tinytest.hpp"

#include "kfcore/yolo/error.hpp"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

using namespace kfcore::yolo;
using namespace kfcore::yolo::detail;

namespace
{

template <typename Callable>
void expect_yolo_error(Callable&& callable, YoloErrorCode code, const char* message_fragment)
{
    try
    {
        callable();
        check(false);
    }
    catch (const YoloError& error)
    {
        check(error.code() == code);
        check(std::string(error.what()).find(message_fragment) != std::string::npos);
    }
}

ImageView host_view(const void* data, std::int32_t width, std::int32_t height,
                    std::size_t stride, PixelFormat format = PixelFormat::Rgb8)
{
    return { data, width, height, stride, format, MemoryKind::Host };
}

} // namespace

spec("TensorRT YOLO detection helpers")
{
    it("computes independent transforms for wide and tall images")
    {
        const LetterboxTransform wide = compute_letterbox_transform(960, 320, 640, 640);
        const LetterboxTransform tall = compute_letterbox_transform(320, 960, 640, 640);

        check(std::fabs(wide.scale - (2.0f / 3.0f)) < 1.0e-6f);
        check(std::fabs(wide.pad_x) < 1.0e-6f);
        check(std::fabs(wide.pad_y - (640.0f / 3.0f)) < 1.0e-4f);
        check(wide.source_width == 960);
        check(wide.source_height == 320);

        check(std::fabs(tall.scale - (2.0f / 3.0f)) < 1.0e-6f);
        check(std::fabs(tall.pad_x - (640.0f / 3.0f)) < 1.0e-4f);
        check(std::fabs(tall.pad_y) < 1.0e-6f);
        check(tall.source_width == 320);
        check(tall.source_height == 960);
    }

    it("rejects empty and over-limit batches")
    {
        const std::vector<ImageView> empty;
        expect_yolo_error(
            [&] { (void)prepare_batch(empty, 1, 2, 640, 640, TensorDataType::Float32, 4096); },
            YoloErrorCode::InvalidArgument, "empty");

        std::uint8_t pixels[12] {};
        const ImageView view = host_view(pixels, 2, 2, 6);
        expect_yolo_error(
            [&] {
                (void)prepare_batch({ view, view, view }, 1, 2, 2, 2,
                                    TensorDataType::Float32, 1024);
            },
            YoloErrorCode::ResourceLimitExceeded, "batch");
    }

    it("validates pointer dimensions stride format memory and checked source bytes")
    {
        std::uint8_t pixels[12] {};
        const ImageView valid = host_view(pixels, 2, 2, 6);
        const BatchInputPlan plan =
            prepare_batch({ valid }, 1, 1, 2, 2, TensorDataType::Float32, 1024);
        check(plan.images.size() == 1);
        check(plan.host_staging_bytes == 12);
        check(plan.input_bytes == 48);

        ImageView bad = valid;
        bad.data = nullptr;
        expect_yolo_error(
            [&] { (void)prepare_batch({ bad }, 1, 1, 2, 2, TensorDataType::Float32, 1024); },
            YoloErrorCode::InvalidArgument, "data");

        bad = valid;
        bad.width = 0;
        expect_yolo_error(
            [&] { (void)prepare_batch({ bad }, 1, 1, 2, 2, TensorDataType::Float32, 1024); },
            YoloErrorCode::InvalidArgument, "dimensions");

        bad = valid;
        bad.row_stride = 5;
        expect_yolo_error(
            [&] { (void)prepare_batch({ bad }, 1, 1, 2, 2, TensorDataType::Float32, 1024); },
            YoloErrorCode::InvalidArgument, "stride");

        bad = valid;
        bad.pixel_format = static_cast<PixelFormat>(99);
        expect_yolo_error(
            [&] { (void)prepare_batch({ bad }, 1, 1, 2, 2, TensorDataType::Float32, 1024); },
            YoloErrorCode::InvalidArgument, "pixel format");

        bad = valid;
        bad.memory_kind = static_cast<MemoryKind>(99);
        expect_yolo_error(
            [&] { (void)prepare_batch({ bad }, 1, 1, 2, 2, TensorDataType::Float32, 1024); },
            YoloErrorCode::InvalidArgument, "memory kind");

        bad = valid;
        bad.height = 3;
        bad.row_stride = (std::numeric_limits<std::size_t>::max)();
        expect_yolo_error(
            [&] { (void)prepare_batch({ bad }, 1, 1, 2, 2, TensorDataType::Float32, 1024); },
            YoloErrorCode::ResourceLimitExceeded, "overflow");

        expect_yolo_error(
            [&] { (void)prepare_batch({ valid }, 1, 1, 2, 2, TensorDataType::Float32, 11); },
            YoloErrorCode::ResourceLimitExceeded, "source bytes");
    }

    it("plans host and CUDA-device inputs without mixing their staging offsets")
    {
        std::uint8_t host_pixels[12] {};
        ImageView host = host_view(host_pixels, 2, 2, 6, PixelFormat::Bgr8);
        ImageView device = host;
        device.data = reinterpret_cast<const void*>(std::uintptr_t { 0x1000 });
        device.memory_kind = MemoryKind::CudaDevice;

        const BatchInputPlan plan =
            prepare_batch({ host, device }, 1, 2, 2, 2, TensorDataType::Float16, 1024);
        check(plan.host_staging_bytes == 12);
        check(plan.input_bytes == 48);
        check(plan.images[0].staging_offset == 0);
        check(plan.images[1].staging_offset == kNoStagingOffset);
    }

    it("computes bounded dynamic tensor byte sizes")
    {
        const DetectionBufferLayout fp32 =
            compute_detection_buffer_layout(2, 3, TensorDataType::Float32, 1024);
        check(fp32.num_dets_bytes == 8);
        check(fp32.boxes_bytes == 96);
        check(fp32.scores_bytes == 24);
        check(fp32.labels_bytes == 24);
        check(fp32.total_output_bytes == 152);

        const DetectionBufferLayout fp16 =
            compute_detection_buffer_layout(2, 3, TensorDataType::Float16, 1024);
        check(fp16.boxes_bytes == 48);
        check(fp16.scores_bytes == 12);
        check(fp16.total_output_bytes == 92);

        expect_yolo_error(
            [&] { (void)compute_detection_buffer_layout(2, 3, TensorDataType::Float32, 151); },
            YoloErrorCode::ResourceLimitExceeded, "outputs");
        expect_yolo_error(
            [&] {
                (void)compute_detection_buffer_layout(
                    (std::numeric_limits<std::size_t>::max)(), 2,
                    TensorDataType::Float32, (std::numeric_limits<std::size_t>::max)());
            },
            YoloErrorCode::ResourceLimitExceeded, "overflow");
    }

    it("inverse-transforms each batch image and preserves result order")
    {
        std::uint8_t wide_pixels[3] {};
        std::uint8_t tall_pixels[3] {};
        const std::vector<ImageView> images = {
            host_view(wide_pixels, 960, 320, 2880),
            host_view(tall_pixels, 320, 960, 960),
        };
        const std::vector<LetterboxTransform> transforms = {
            compute_letterbox_transform(960, 320, 640, 640),
            compute_letterbox_transform(320, 960, 640, 640),
        };
        const std::int32_t counts[] = { 1, 1 };
        const float boxes[] = {
            0.0f, 640.0f / 3.0f, 640.0f, 1280.0f / 3.0f,
            640.0f / 3.0f, 0.0f, 1280.0f / 3.0f, 640.0f,
        };
        const float scores[] = { 0.75f, 0.5f };
        const std::int32_t labels[] = { 3, 7 };
        const EfficientNmsOutputView outputs {
            counts, 2, boxes, 8, scores, 2, labels, 2, 1, TensorDataType::Float32,
        };

        const std::vector<DetectionFrame> results =
            decode_efficient_nms(images, transforms, outputs);
        check(results.size() == 2);
        check(results[0].image_width == 960);
        check(results[0].image_height == 320);
        check(results[0].detections.size() == 1);
        check(std::fabs(results[0].detections[0].box.right - 960.0f) < 1.0e-4f);
        check(std::fabs(results[0].detections[0].box.bottom - 320.0f) < 1.0e-4f);
        check(results[0].detections[0].class_id == 3);
        check(results[1].image_width == 320);
        check(results[1].image_height == 960);
        check(results[1].detections[0].class_id == 7);
    }

    it("decodes FP16 outputs and clamps only finite coordinates")
    {
        std::uint8_t pixels[3] {};
        const std::vector<ImageView> images = { host_view(pixels, 2, 2, 6) };
        const std::vector<LetterboxTransform> transforms = {
            compute_letterbox_transform(2, 2, 2, 2),
        };
        const std::int32_t counts[] = { 1 };
        const std::uint16_t boxes[] = { 0xbc00, 0xbc00, 0x4200, 0x4200 };
        const std::uint16_t scores[] = { 0x3800 };
        const std::int32_t labels[] = { 1 };
        const EfficientNmsOutputView outputs {
            counts, 1, boxes, 4, scores, 1, labels, 1, 1, TensorDataType::Float16,
        };

        const std::vector<DetectionFrame> results =
            decode_efficient_nms(images, transforms, outputs);
        const Detection& detection = results[0].detections[0];
        check(std::fabs(detection.box.left) < 1.0e-6f);
        check(std::fabs(detection.box.top) < 1.0e-6f);
        check(std::fabs(detection.box.right - 2.0f) < 1.0e-6f);
        check(std::fabs(detection.box.bottom - 2.0f) < 1.0e-6f);
        check(std::fabs(detection.score - 0.5f) < 1.0e-6f);
    }

    it("rejects inconsistent output sizes counts labels and invalid floating values")
    {
        std::uint8_t pixels[3] {};
        const std::vector<ImageView> images = { host_view(pixels, 2, 2, 6) };
        const std::vector<LetterboxTransform> transforms = {
            compute_letterbox_transform(2, 2, 2, 2),
        };
        std::int32_t counts[] = { 1 };
        float boxes[] = { 0.0f, 0.0f, 1.0f, 1.0f };
        float scores[] = { 0.5f };
        std::int32_t labels[] = { 0 };
        EfficientNmsOutputView outputs {
            counts, 1, boxes, 4, scores, 1, labels, 1, 1, TensorDataType::Float32,
        };

        outputs.boxes_count = 3;
        expect_yolo_error([&] { (void)decode_efficient_nms(images, transforms, outputs); },
                          YoloErrorCode::TensorRtFailure, "size");
        outputs.boxes_count = 4;

        counts[0] = -1;
        expect_yolo_error([&] { (void)decode_efficient_nms(images, transforms, outputs); },
                          YoloErrorCode::TensorRtFailure, "num_dets");
        counts[0] = 2;
        expect_yolo_error([&] { (void)decode_efficient_nms(images, transforms, outputs); },
                          YoloErrorCode::TensorRtFailure, "num_dets");
        counts[0] = 1;

        labels[0] = -1;
        expect_yolo_error([&] { (void)decode_efficient_nms(images, transforms, outputs); },
                          YoloErrorCode::TensorRtFailure, "label");
        labels[0] = 0;

        scores[0] = (std::numeric_limits<float>::quiet_NaN)();
        expect_yolo_error([&] { (void)decode_efficient_nms(images, transforms, outputs); },
                          YoloErrorCode::TensorRtFailure, "score");
        scores[0] = 0.5f;

        boxes[0] = (std::numeric_limits<float>::infinity)();
        expect_yolo_error([&] { (void)decode_efficient_nms(images, transforms, outputs); },
                          YoloErrorCode::TensorRtFailure, "box");
        boxes[0] = 1.0f;
        boxes[2] = 0.0f;
        expect_yolo_error([&] { (void)decode_efficient_nms(images, transforms, outputs); },
                          YoloErrorCode::TensorRtFailure, "inverted");
    }
}
