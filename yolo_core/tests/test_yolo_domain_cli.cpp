#include "yolo_domain_cli.hpp"

#include "tinytest.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace kfcore::yolo::demo;

namespace
{

turbo_video_native_mode_t mode(int width, int height, int fps, int format,
                               std::uint64_t id)
{
    return { width, height, static_cast<std::uint32_t>(fps), 1U, format, id };
}
} // namespace

spec("YOLOv8 domain application CLI")
{
    it("lists cameras without loading a model or selecting a backend")
    {
        const Arguments arguments = parse_arguments(
            { "yolov8_domain_demo", "--list-cameras" }, { true, true });
        check_true(arguments.list_cameras);
        check_false(arguments.backend.has_value());
        check_false(arguments.application.has_value());
    }

    it("selects the only compiled backend and defaults to 640x480 NV12 request")
    {
        const Arguments arguments = parse_arguments(
            { "yolov8_domain_demo", "--application", "parking", "--model",
              "parking.onnx", "--max-frames", "10", "--headless" },
            { true, false });
        check(arguments.backend == Backend::Cpu);
        check(arguments.application == DomainKind::Parking);
        check(arguments.capture.width == 640);
        check(arguments.capture.height == 480);
        check(arguments.capture.fps == 30);
        check(arguments.max_frames == 10U);
    }

    it("derives application model paths from the configured model root")
    {
        const std::filesystem::path root("C:/kfcore-models");
        const Arguments cpu = parse_arguments(
            { "demo", "--application", "parking" }, { true, false }, root,
            "rtx4060-sm89-trt11.2.1-default");
        check(cpu.model == root / "yolov8n-parking.onnx");

        const Arguments gpu = parse_arguments(
            { "demo", "--application", "football", "--backend", "tensorrt" },
            { true, true }, root, "rtx4060-sm89-trt11.2.1-default");
        check(gpu.model == root / "tensorrt" /
                               "rtx4060-sm89-trt11.2.1-default" /
                               "yolov8n-football.engine");

        const Arguments explicit_model = parse_arguments(
            { "demo", "--application", "drone", "--model", "custom.onnx" },
            { true, false }, root, "rtx4060-sm89-trt11.2.1-default");
        check(explicit_model.model == "custom.onnx");
    }

    it("requires an explicit backend when both are compiled")
    {
        check_throws_as(parse_arguments(
                            { "demo", "--application", "drone", "--model", "m.onnx" },
                            { true, true }),
                        std::invalid_argument);
        const Arguments arguments = parse_arguments(
            { "demo", "--application", "football", "--backend", "tensorrt",
              "--model", "football.engine" }, { true, true });
        check(arguments.backend == Backend::TensorRt);
    }

    it("accepts a bounded image-directory source and rejects source conflicts")
    {
        const Arguments arguments = parse_arguments(
            { "demo", "--application", "drone", "--model", "drone.onnx",
              "--images", "input", "--output", "output", "--max-frames", "25" },
            { true, false });
        check(arguments.images.has_value());
        check(arguments.output.has_value());
        check(arguments.max_frames == 25U);

        check_throws_as(parse_arguments(
                            { "demo", "--application", "drone", "--model", "m.onnx",
                              "--images", "in", "--output", "out", "--camera", "1" },
                            { true, false }),
                        std::invalid_argument);
        check_throws_as(parse_arguments(
                            { "demo", "--application", "drone", "--model", "m.onnx",
                              "--images", "in" }, { true, false }),
                        std::invalid_argument);
    }

    it("rejects unbounded headless capture invalid scores and unavailable backends")
    {
        check_throws_as(parse_arguments(
                            { "demo", "--application", "drone", "--model", "m.onnx",
                              "--headless" }, { true, false }),
                        std::invalid_argument);
        check_throws_as(parse_arguments(
                            { "demo", "--application", "drone", "--model", "m.onnx",
                              "--score-threshold", "1.1" }, { true, false }),
                        std::invalid_argument);
        check_throws_as(parse_arguments(
                            { "demo", "--application", "drone", "--backend", "tensorrt",
                              "--model", "m.engine" }, { true, false }),
                        std::invalid_argument);
    }

    it("requires the default exact NV12 mode and accepts explicit supported modes")
    {
        const std::vector<turbo_video_native_mode_t> modes {
            mode(640, 480, 30, TURBO_VIDEO_CAPTURE_FORMAT_RGB24, 1U),
            mode(640, 480, 30, TURBO_VIDEO_CAPTURE_FORMAT_NV12, 2U),
            mode(1280, 720, 30, TURBO_VIDEO_CAPTURE_FORMAT_BGRA, 3U),
        };
        CaptureRequest request;
        check(select_mode(modes, request) == 1U);
        request.mode_id = 3U;
        check(select_mode(modes, request) == 2U);
        request.mode_id = 99U;
        check_throws_as(select_mode(modes, request), std::invalid_argument);

        CaptureRequest default_request;
        check_throws_as(select_mode(
                            { mode(640, 480, 30, TURBO_VIDEO_CAPTURE_FORMAT_I420, 4U) },
                            default_request),
                        std::invalid_argument);
    }
}
