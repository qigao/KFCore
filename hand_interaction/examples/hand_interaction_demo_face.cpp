#include "hand_interaction_demo_face.hpp"

#if defined(KFCORE_HAND_DEMO_HAS_CPU)
#include "kfcore/vision_models/cpu.hpp"
#endif
#if defined(KFCORE_HAND_DEMO_HAS_TENSORRT)
#include "kfcore/vision_models/tensorrt.hpp"
#include "kfcore/yolo/tensorrt.hpp"
#endif

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace kfcore::hand_interaction::demo
{
namespace
{

#if defined(KFCORE_HAND_DEMO_HAS_TENSORRT)
class TensorRtYoloFaceDetector final : public vision_models::FaceDetectorBackend
{
public:
    TensorRtYoloFaceDetector(const std::filesystem::path& engine_path,
                             float score_threshold)
        : score_threshold_(score_threshold)
        , engine_(yolo::Engine::load(engine_path))
        , detector_(engine_->create_detector())
    {
    }

    vision_models::FaceDetectionResult infer(
        const image::ImageView& source) override
    {
        if (source.pixel_format != image::PixelFormat::Bgr8 &&
            source.pixel_format != image::PixelFormat::Rgb8)
        {
            throw std::invalid_argument(
                "TensorRT demo face detector requires BGR8 or RGB8 input");
        }
        const auto started = std::chrono::steady_clock::now();
        const yolo::ImageView view {
            source.data, source.width, source.height, source.row_stride,
            source.pixel_format == image::PixelFormat::Bgr8
                ? yolo::PixelFormat::Bgr8
                : yolo::PixelFormat::Rgb8,
            source.memory_kind == image::MemoryKind::Host
                ? yolo::MemoryKind::Host
                : yolo::MemoryKind::CudaDevice
        };
        const yolo::DetectionFrame detections = detector_->detect(view);
        vision_models::FaceDetectionResult result;
        for (const auto& detection : detections.detections)
        {
            if (detection.class_id != 0 || detection.score < score_threshold_ ||
                (result.face.has_value() &&
                 detection.score <= result.face->confidence))
            {
                continue;
            }
            result.face = vision_models::FaceDetection {
                { detection.box.left, detection.box.top,
                  detection.box.right - detection.box.left,
                  detection.box.bottom - detection.box.top },
                detection.score
            };
        }
        result.inference_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started)
                                  .count();
        result.total_ms = result.inference_ms;
        return result;
    }

private:
    float score_threshold_;
    std::shared_ptr<const yolo::Engine> engine_;
    std::unique_ptr<yolo::TensorRtDetector> detector_;
};
#endif

} // namespace

std::unique_ptr<vision_models::FaceMeshPipeline> make_face_pipeline(
    const Arguments& arguments)
{
    if (arguments.face_detector_model.has_value() !=
        arguments.face_landmark_model.has_value())
    {
        throw std::invalid_argument(
            "FaceMesh demo requires detector and landmark models together");
    }
    if (!arguments.face_detector_model.has_value())
    {
        return nullptr;
    }
    if (!std::isfinite(arguments.face_detection_score_threshold) ||
        arguments.face_detection_score_threshold < 0.0F ||
        arguments.face_detection_score_threshold > 1.0F ||
        !std::isfinite(arguments.face_landmark_score_threshold) ||
        arguments.face_landmark_score_threshold < 0.0F ||
        arguments.face_landmark_score_threshold > 1.0F)
    {
        throw std::invalid_argument(
            "FaceMesh demo score thresholds must be finite within [0,1]");
    }
    vision_models::FaceMeshPipelineOptions pipeline_options;
    pipeline_options.landmark_score_threshold =
        arguments.face_landmark_score_threshold;
    if (arguments.backend == Backend::Cpu)
    {
#if defined(KFCORE_HAND_DEMO_HAS_CPU)
        vision_models::CpuVisionOptions options;
        options.face_detection_score_threshold =
            arguments.face_detection_score_threshold;
        return vision_models::FaceMeshPipeline::create(
            vision_models::CpuFaceDetector::load(
                *arguments.face_detector_model, options),
            vision_models::CpuFaceLandmarker::load(
                *arguments.face_landmark_model, options),
            pipeline_options);
#else
        throw std::logic_error("CPU backend was not compiled into this demo");
#endif
    }
#if defined(KFCORE_HAND_DEMO_HAS_TENSORRT)
    return vision_models::FaceMeshPipeline::create(
        std::make_unique<TensorRtYoloFaceDetector>(
            *arguments.face_detector_model,
            arguments.face_detection_score_threshold),
        vision_models::TensorRtFaceLandmarker::load(
            *arguments.face_landmark_model),
        pipeline_options);
#else
    throw std::logic_error("TensorRT backend was not compiled into this demo");
#endif
}

} // namespace kfcore::hand_interaction::demo
