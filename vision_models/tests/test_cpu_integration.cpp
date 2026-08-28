#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "geometry.hpp"
#include "kfcore/image_processor/cpu.hpp"
#include "kfcore/vision_models/cpu.hpp"
#include "onnx_session.hpp"
#include "tinytest.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

using namespace kfcore::vision_models;

namespace
{

constexpr std::size_t kMaximumModelBytes  = 256U * 1024U * 1024U;
constexpr std::size_t kMaximumOutputBytes = 16U * 1024U * 1024U;

detail::OnnxSessionOptions session_options()
{
    return { 1, 1, kMaximumModelBytes, kMaximumOutputBytes };
}

void check_finite(const std::vector<float>& values)
{
    for (float value : values)
    {
        check_true(std::isfinite(value));
    }
}

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

kfcore::image::BgrImage read_bgr(const std::filesystem::path& path)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* rgb = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
    if (rgb == nullptr)
    {
        throw std::runtime_error("failed to decode integration image");
    }
    const std::size_t bytes = static_cast<std::size_t>(width) * height * 3U;
    const kfcore::image::ImageView view {
        rgb, bytes, width, height, static_cast<std::size_t>(width) * 3U,
        kfcore::image::PixelFormat::Rgb8, kfcore::image::MemoryKind::Host
    };
    kfcore::image::BgrImage result = kfcore::image::CpuImageProcessor::copy_bgr(
        view, 64U * 1024U * 1024U);
    stbi_image_free(rgb);
    return result;
}

} // namespace

spec("CPU vision real-model integration")
{
    it("executes every trusted ONNX model with its exact scalar and shape contract")
    {
        Ort::Env environment(ORT_LOGGING_LEVEL_ERROR, "KFCoreVisionCpuIntegration");

        detail::OnnxSession palm(
            environment, std::filesystem::path(KFCORE_VISION_CPU_TEST_PALM),
            { "Palm",
              { { "input", detail::OnnxElementType::Float32, { 1, 3, 192, 192 } } },
              { { "pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y",
                  detail::OnnxElementType::Float32, { -1, 8 } } } },
            session_options());
        std::vector<float> palm_input(3U * 192U * 192U, 0.0F);
        const auto palm_output = palm.run(
            { { palm_input.data(), palm_input.size(), { 1, 3, 192, 192 } } });
        check(palm_output.size() == 1);
        check(palm_output[0].dimensions.size() == 2);
        check(palm_output[0].dimensions[0] >= 0);
        check(palm_output[0].dimensions[0] <= 2016);
        check(palm_output[0].dimensions[1] == 8);
        check_finite(palm_output[0].float_values);

        detail::OnnxSession hand(
            environment, std::filesystem::path(KFCORE_VISION_CPU_TEST_HAND),
            { "hand landmark",
              { { "input", detail::OnnxElementType::Float32, { -1, 3, 224, 224 } } },
              { { "xyz_x21", detail::OnnxElementType::Float32, { -1, 63 } },
                { "hand_score", detail::OnnxElementType::Float32, { -1, 1 } },
                { "lefthand_0_or_righthand_1", detail::OnnxElementType::Float32,
                  { -1, 1 } } } },
            session_options());
        std::vector<float> hand_input(3U * 224U * 224U, 0.0F);
        const auto hand_output = hand.run(
            { { hand_input.data(), hand_input.size(), { 1, 3, 224, 224 } } });
        check(hand_output.size() == 3);
        check(hand_output[0].float_values.size() == 63);
        check(hand_output[1].float_values.size() == 1);
        check(hand_output[2].float_values.size() == 1);
        check_finite(hand_output[0].float_values);
        check_finite(hand_output[1].float_values);
        check_finite(hand_output[2].float_values);

        detail::OnnxSession classifier(
            environment, std::filesystem::path(KFCORE_VISION_CPU_TEST_CLASSIFIER),
            { "keypoint classifier",
              { { "input", detail::OnnxElementType::Float32, { -1, 42 } } },
              { { "class_ids", detail::OnnxElementType::Int64, { -1 } } } },
            session_options());
        std::vector<float> classifier_input(42U, 0.0F);
        const auto classifier_output = classifier.run(
            { { classifier_input.data(), classifier_input.size(), { 1, 42 } } });
        check(classifier_output.size() == 1);
        check(classifier_output[0].element_type == detail::OnnxElementType::Int64);
        check(classifier_output[0].int64_values.size() == 1);

        detail::OnnxSession face(
            environment, std::filesystem::path(KFCORE_VISION_CPU_TEST_FACE),
            { "MediaPipe face landmark",
              { { "image", detail::OnnxElementType::Float32, { 1, 3, 192, 192 } } },
              { { "scores", detail::OnnxElementType::Float32, { 1 } },
                { "landmarks", detail::OnnxElementType::Float32, { 1, 468, 3 } } } },
            session_options());
        const kfcore::image::BgrImage face_image = read_bgr(
            KFCORE_VISION_CPU_TEST_FACE_IMAGE);
        const detail::FaceRoi face_roi = detail::make_face_roi(
            { 0.0F, 0.0F, static_cast<float>(face_image.width),
              static_cast<float>(face_image.height) }, 192);
        const kfcore::image::BgrImage face_crop =
            kfcore::image::CpuImageProcessor::warp_affine_bgr(
                face_image, 192, 192, face_roi.destination_to_source, 0.0F,
                64U * 1024U * 1024U);
        kfcore::image::PreprocessOptions face_preprocess;
        face_preprocess.output_format = kfcore::image::PixelFormat::Rgb8;
        face_preprocess.border_value = 0.0F;
        std::vector<float> face_input = kfcore::image::CpuImageProcessor::to_nchw(
            face_crop, face_preprocess, 16U * 1024U * 1024U);
        const auto face_output = face.run(
            { { face_input.data(), face_input.size(), { 1, 3, 192, 192 } } });
        check(face_output.size() == 2);
        check(face_output[0].float_values.size() == 1);
        check(face_output[1].float_values.size() == 468U * 3U);
        check_finite(face_output[0].float_values);
        check_finite(face_output[1].float_values);
        float maximum_face_xy = 0.0F;
        for (std::size_t index = 0; index < 468U; ++index)
        {
            maximum_face_xy = (std::max)(
                maximum_face_xy, std::fabs(face_output[1].float_values[index * 3U]));
            maximum_face_xy = (std::max)(
                maximum_face_xy, std::fabs(face_output[1].float_values[index * 3U + 1U]));
        }
        check_true(maximum_face_xy > 0.0F);
        check_true(maximum_face_xy <= 2.0F);
    }

    it("runs the OpenCV-free public hand and face preprocessing paths")
    {
        const HandOnnxModelPaths paths {
            KFCORE_VISION_CPU_TEST_PALM,
            KFCORE_VISION_CPU_TEST_HAND,
            KFCORE_VISION_CPU_TEST_CLASSIFIER,
        };
        CpuVisionOptions options;
        options.intra_op_threads = 1;
        options.inter_op_threads = 1;
        auto hand = CpuHandBackend::load(paths, options);
        auto face = CpuFaceLandmarker::load(KFCORE_VISION_CPU_TEST_FACE, options);

        const kfcore::image::BgrImage hand_image = read_bgr(
            KFCORE_VISION_CPU_TEST_HAND_IMAGE);
        const HandFrame hand_frame = hand->infer(hand_image.view());
        check(hand_frame.hands.size() <= options.max_hands);
        check_true(!hand_frame.hands.empty());
        check_true(std::isfinite(hand_frame.timings.total_ms));
        check_true(hand_frame.timings.preprocess_ms > 0.0);
        check_true(hand_frame.timings.palm_inference_ms > 0.0);
        check_true(hand_frame.timings.landmark_inference_ms > 0.0);
        check_true(hand_frame.timings.classifier_inference_ms > 0.0);
        for (const HandResult& result : hand_frame.hands)
        {
            check_true(std::isfinite(result.palm.confidence));
            check_true(std::isfinite(result.landmark_confidence));
            check(result.gesture != Gesture::Unknown);
        }
        kfcore::image::ImageView device_view = hand_image.view();
        device_view.memory_kind = kfcore::image::MemoryKind::CudaDevice;
        check_error([&] { (void)hand->infer(device_view); },
                    VisionModelErrorCode::InvalidArgument, "Host images only");

        HandPipelineOptions pipeline_options;
        pipeline_options.max_hands = options.max_hands;
        pipeline_options.tracker.minimum_consecutive_frames = 1;
        auto pipeline = HandPipeline::create(std::move(hand), pipeline_options);
        (void)pipeline->process(hand_image.view());
        const HandFrame tracked_frame = pipeline->process(hand_image.view());
        check_true(tracked_frame.timings.tracking_ms > 0.0);
        bool has_track = false;
        for (const HandResult& result : tracked_frame.hands)
        {
            has_track = has_track || result.track_id >= 0;
        }
        check_true(has_track);

        const kfcore::image::BgrImage face_image = read_bgr(
            KFCORE_VISION_CPU_TEST_FACE_IMAGE);
        check_error([&] { (void)face->infer(face_image.view(), {}); },
                    VisionModelErrorCode::InvalidArgument, "face box");
        const FaceLandmarkResult face_result =
            face->infer(face_image.view(),
                        { 0.0F, 0.0F, static_cast<float>(face_image.width),
                          static_cast<float>(face_image.height) });
        check_true(std::isfinite(face_result.confidence));
        check_true(std::isfinite(face_result.total_ms));
        for (const Point3f& point : face_result.landmarks)
        {
            check_true(std::isfinite(point.x));
            check_true(std::isfinite(point.y));
            check_true(std::isfinite(point.z));
        }
    }

    it("runs YOLOv12 face detection and MediaPipe landmarks as one CPU pipeline")
    {
        CpuVisionOptions options;
        options.intra_op_threads = 1;
        options.inter_op_threads = 1;
        options.face_detection_score_threshold = 0.25F;
        auto detector = CpuFaceDetector::load(
            KFCORE_VISION_CPU_TEST_FACE_DETECTOR, options);
        auto landmarker = CpuFaceLandmarker::load(
            KFCORE_VISION_CPU_TEST_FACE, options);
        auto pipeline = FaceMeshPipeline::create(
            std::move(detector), std::move(landmarker));

        const kfcore::image::BgrImage image = read_bgr(
            KFCORE_VISION_CPU_TEST_FACE_IMAGE);
        const FaceMeshFrame frame = pipeline->process(image.view());

        check_true(frame.detection.has_value());
        check_true(frame.landmarks.has_value());
        check_true(frame.timings.detection_preprocess_ms > 0.0);
        check_true(frame.timings.detection_inference_ms > 0.0);
        check_true(frame.timings.landmark_preprocess_ms > 0.0);
        check_true(frame.timings.landmark_inference_ms > 0.0);
        check_true(frame.timings.total_ms > 0.0);
        const RectF& box = frame.detection->box;
        check_true(box.x >= 0.0F);
        check_true(box.y >= 0.0F);
        check_true(box.x + box.width <= static_cast<float>(image.width));
        check_true(box.y + box.height <= static_cast<float>(image.height));
        for (const Point3f& point : frame.landmarks->landmarks)
        {
            check_true(std::isfinite(point.x));
            check_true(std::isfinite(point.y));
            check_true(std::isfinite(point.z));
        }
    }
}
