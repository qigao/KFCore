#include "hand_interaction_demo_cli.hpp"
#include "tinytest.hpp"

#include <filesystem>
#include <fstream>
#include <cmath>
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
        write("hand_gesture_model/tensorrt/test-profile/palm_detection.engine");
        write("hand_gesture_model/tensorrt/test-profile/hand_landmark.engine");
        write("hand_gesture_model/tensorrt/test-profile/keypoint_classifier.engine");
        write("yolov12n-face.onnx");
        write("MediaPipeFaceLandmarkDetector.onnx");
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
        check_equal(arguments.capture.width, 640);
        check_equal(arguments.capture.height, 480);
        check_equal(arguments.capture.fps, 30);
    }

    it("derives missing CPU and TensorRT paths from the configured model root")
    {
        ModelFixture models;
        const std::filesystem::path root(models.path());

        const auto cpu = demo::parse_arguments(
            { "hand_interaction_demo" }, { true, false }, root, "test-profile");
        check(cpu.palm_model ==
              root / "hand_gesture_model/palm_detection/"
                     "palm_detection_full_inf_post_192x192.onnx");

        const auto gpu = demo::parse_arguments(
            { "hand_interaction_demo", "--backend", "tensorrt" },
            { true, true }, root, "test-profile");
        check(gpu.palm_model == root / "hand_gesture_model/tensorrt/test-profile/"
                                      "palm_detection.engine");
        check(gpu.hand_model == root / "hand_gesture_model/tensorrt/test-profile/"
                                      "hand_landmark.engine");
        check(gpu.classifier_model ==
              root / "hand_gesture_model/tensorrt/test-profile/"
                     "keypoint_classifier.engine");

        const auto explicit_model = demo::parse_arguments(
            { "hand_interaction_demo", "--model-dir", models.path() },
            { true, false }, "Z:/unused-model-root", "unused-profile");
        check(explicit_model.palm_model.parent_path().filename() == "palm_detection");
    }

    it("enables the optional horizontal palm-axis Wave gate explicitly")
    {
        ModelFixture models;
        const auto defaults = demo::parse_arguments(
            { "hand_interaction_demo", "--model-dir", models.path() },
            { true, false });
        const auto constrained = demo::parse_arguments(
            { "hand_interaction_demo", "--wave-require-horizontal-axis",
              "--model-dir", models.path() },
            { true, false });

        check_false(defaults.wave_require_horizontal_axis);
        check_true(constrained.wave_require_horizontal_axis);
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

    it("enables face mesh only when both model paths are supplied")
    {
        ModelFixture models;
        const std::filesystem::path root(models.path());
        const auto arguments = demo::parse_arguments(
            { "hand_interaction_demo", "--model-dir", models.path(),
              "--face-detector", (root / "yolov12n-face.onnx").string(),
              "--facemesh", (root / "MediaPipeFaceLandmarkDetector.onnx").string(),
              "--face-score", "0.6", "--facemesh-score", "0.7" },
            { true, false });
        check_true(arguments.face_detector_model.has_value());
        check_true(arguments.face_landmark_model.has_value());
        check_true(std::fabs(arguments.face_detection_score_threshold - 0.6F) <
                   0.0001F);
        check_true(std::fabs(arguments.face_landmark_score_threshold - 0.7F) <
                   0.0001F);

        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--model-dir", models.path(),
                              "--face-detector",
                              (root / "yolov12n-face.onnx").string() },
                            { true, false }),
                        std::invalid_argument);
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--model-dir", models.path(),
                              "--face-score", "1.1" },
                            { true, false }),
                        std::invalid_argument);
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--model-dir", models.path(),
                              "--face-score", "0.6" },
                            { true, false }),
                        std::invalid_argument);
        check_throws_as(demo::parse_arguments(
                            { "hand_interaction_demo", "--model-dir", models.path(),
                              "--face-detector", "Z:/missing-face.onnx",
                              "--facemesh",
                              (root / "MediaPipeFaceLandmarkDetector.onnx").string() },
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

    it("defaults to 640x480 and prefers NV12 then I420")
    {
        const std::vector<turbo_video_native_mode_t> modes {
            mode(640, 480, 30, TURBO_VIDEO_CAPTURE_FORMAT_RGB24, 1U),
            mode(640, 480, 30, TURBO_VIDEO_CAPTURE_FORMAT_I420, 2U),
            mode(640, 480, 30, TURBO_VIDEO_CAPTURE_FORMAT_NV12, 3U),
        };
        demo::CaptureRequest request;
        check_equal(demo::select_mode(modes, request), (std::size_t)2U);
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
