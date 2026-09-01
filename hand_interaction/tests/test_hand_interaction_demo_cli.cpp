#include "hand_interaction_demo_cli.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

namespace demo = kfcore::hand_interaction::demo;

turbo_video_native_mode_t mode(int width, int height, int fps, int format,
                               std::uint64_t id)
{
    return { width, height, static_cast<std::uint32_t>(fps), 1U, format, id };
}

} // namespace

spec("hand interaction demo CLI")
{
    it("lists cameras without selecting a backend or loading models")
    {
        const auto arguments = demo::parse_arguments(
            { "hand_interaction_demo", "--list-cameras" }, { true, true });
        check_true(arguments.list_cameras);
        check_false(arguments.backend.has_value());
    }

    it("selects the only compiled backend and defaults the capture request")
    {
        const auto arguments =
            demo::parse_arguments({ "hand_interaction_demo" }, { true, false });
        check(arguments.backend == demo::Backend::Cpu);
        check_equal(arguments.capture.camera_index, 0);
        check_equal(arguments.capture.width, 640);
        check_equal(arguments.capture.height, 480);
        check_equal(arguments.capture.fps, 30);
    }

    it("enables the optional horizontal palm-axis Wave gate explicitly")
    {
        const auto defaults = demo::parse_arguments(
            { "hand_interaction_demo" }, { true, false });
        const auto constrained = demo::parse_arguments(
            { "hand_interaction_demo", "--wave-require-horizontal-axis" },
            { true, false });

        check_false(defaults.wave_require_horizontal_axis);
        check_true(constrained.wave_require_horizontal_axis);
    }

    it("requires an explicit backend when both are compiled")
    {
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo" }, { true, true }),
                        std::invalid_argument);
    }

    it("rejects unavailable backends and malformed numeric options")
    {
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--backend", "tensorrt" },
                            { true, false }),
                        std::invalid_argument);
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--camera", "-1" },
                            { true, false }),
                        std::invalid_argument);
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--fps", "30fps" },
                            { true, false }),
                        std::invalid_argument);
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--max-frames", "0" },
                            { true, false }),
                        std::invalid_argument);
    }

    it("accepts an explicit bounded run length")
    {
        const auto arguments = demo::parse_arguments(
            { "hand_interaction_demo", "--max-frames", "100" },
            { true, false });
        check_equal(*arguments.max_frames, (std::uint64_t)100U);
    }

    it("accepts face mesh thresholds without model path options")
    {
        const auto arguments = demo::parse_arguments(
            { "hand_interaction_demo", "--face-score", "0.6", "--facemesh-score",
              "0.7" },
            { true, false });
        check_true(std::fabs(arguments.face_detection_score_threshold - 0.6F) <
                   0.0001F);
        check_true(std::fabs(arguments.face_landmark_score_threshold - 0.7F) <
                   0.0001F);

        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--face-score", "1.1" },
                            { true, false }),
                        std::invalid_argument);
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--face-detector", "model.onnx" },
                            { true, false }),
                        std::invalid_argument);
    }

    it("rejects unknown duplicate and missing-value options")
    {
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--wat", "1" }, { true, false }),
                        std::invalid_argument);
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--camera", "0", "--camera", "1" },
                            { true, false }),
                        std::invalid_argument);
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--camera" }, { true, false }),
                        std::invalid_argument);
    }

    it("selects an exact mode id")
    {
        const std::vector<turbo_video_native_mode_t> modes {
            mode(640, 480, 30, TURBO_VIDEO_CAPTURE_FORMAT_NV12, 7U),
            mode(1280, 720, 30, TURBO_VIDEO_CAPTURE_FORMAT_BGRA, 9U),
        };
        demo::CaptureRequest request;
        request.mode_id = 9U;
        check_equal(demo::select_mode(modes, request), (std::size_t)1U);
    }

    it("keeps the camera mode order instead of forcing a pixel format")
    {
        const std::vector<turbo_video_native_mode_t> modes {
            mode(640, 480, 30, TURBO_VIDEO_CAPTURE_FORMAT_I420, 1U),
            mode(640, 480, 30, TURBO_VIDEO_CAPTURE_FORMAT_NV12, 2U),
        };
        demo::CaptureRequest request;
        check_equal(demo::select_mode(modes, request), (std::size_t)0U);
    }

    it("rejects missing exact modes and MJPEG-only matches")
    {
        demo::CaptureRequest request;
        check_throws_as(demo::select_mode(
                            { mode(640, 480, 30, TURBO_VIDEO_CAPTURE_FORMAT_MJPEG, 1U) },
                            request),
                        std::invalid_argument);
        check_throws_as(demo::select_mode(
                            { mode(1280, 720, 30, TURBO_VIDEO_CAPTURE_FORMAT_NV12, 1U) },
                            request),
                        std::invalid_argument);
    }
}
