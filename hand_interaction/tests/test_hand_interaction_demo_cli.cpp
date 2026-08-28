#include "hand_interaction_demo_cli.hpp"
#include "tinytest.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

namespace demo = kfcore::hand_interaction::demo;

class ModelFixture final
{
public:
    ModelFixture()
        : root_(std::filesystem::temp_directory_path() / "kfcore_hand_demo_cli_test")
    {
        std::filesystem::remove_all(root_);
        write("hand_gesture_model/palm_detection/palm_detection_full_inf_post_192x192.onnx");
        write("hand_gesture_model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx");
        write("hand_gesture_model/keypoint_classifier/keypoint_classifier.onnx");
    }

    ~ModelFixture() { std::filesystem::remove_all(root_); }

    [[nodiscard]] std::string path() const { return root_.string(); }

private:
    void write(const std::filesystem::path& relative)
    {
        const auto path = root_ / relative;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary) << "model";
    }

    std::filesystem::path root_;
};

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

    it("selects the only compiled backend and derives CPU model paths")
    {
        ModelFixture models;
        const auto arguments = demo::parse_arguments(
            { "hand_interaction_demo", "--model-dir", models.path() }, { true, false });
        check(arguments.backend == demo::Backend::Cpu);
        check_true(arguments.palm_model.filename() ==
                   "palm_detection_full_inf_post_192x192.onnx");
        check_equal(arguments.capture.camera_index, 0);
        check_equal(arguments.capture.width, 1280);
        check_equal(arguments.capture.height, 720);
        check_equal(arguments.capture.fps, 30);
    }

    it("requires an explicit backend when both are compiled")
    {
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--model-dir", "." },
                            { true, true }),
                        std::invalid_argument);
    }

    it("rejects unavailable backends and malformed numeric options")
    {
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--backend", "tensorrt",
                              "--palm", __FILE__, "--hand", __FILE__,
                              "--classifier", __FILE__ },
                            { true, false }),
                        std::invalid_argument);
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--camera", "-1",
                              "--model-dir", "." },
                            { true, false }),
                        std::invalid_argument);
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--fps", "30fps",
                              "--model-dir", "." },
                            { true, false }),
                        std::invalid_argument);
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--max-frames", "0",
                              "--model-dir", "." },
                            { true, false }),
                        std::invalid_argument);
    }

    it("accepts an explicit bounded run length")
    {
        ModelFixture models;
        const auto arguments = demo::parse_arguments(
            { "hand_interaction_demo", "--max-frames", "100", "--model-dir",
              models.path() },
            { true, false });
        check_equal(*arguments.max_frames, (std::uint64_t)100U);
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

    it("validates required model files")
    {
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--model-dir",
                              "Z:/missing-kfcore-models" },
                            { true, false }),
                        std::invalid_argument);
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--backend", "tensorrt",
                              "--palm", __FILE__, "--hand", __FILE__ },
                            { false, true }),
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

    it("prefers NV12 then I420 for an exact geometry and frame rate")
    {
        const std::vector<turbo_video_native_mode_t> modes {
            mode(1280, 720, 30, TURBO_VIDEO_CAPTURE_FORMAT_RGB24, 1U),
            mode(1280, 720, 30, TURBO_VIDEO_CAPTURE_FORMAT_I420, 2U),
            mode(1280, 720, 30, TURBO_VIDEO_CAPTURE_FORMAT_NV12, 3U),
        };
        demo::CaptureRequest request;
        check_equal(demo::select_mode(modes, request), (std::size_t)2U);
    }

    it("rejects missing exact modes and MJPEG-only matches")
    {
        demo::CaptureRequest request;
        check_throws_as(demo::select_mode(
                            { mode(1280, 720, 30, TURBO_VIDEO_CAPTURE_FORMAT_MJPEG, 1U) },
                            request),
                        std::invalid_argument);
        check_throws_as(demo::select_mode(
                            { mode(640, 480, 30, TURBO_VIDEO_CAPTURE_FORMAT_NV12, 1U) },
                            request),
                        std::invalid_argument);
    }
}
