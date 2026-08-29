#include "hand_interaction_demo_face.hpp"

#if defined(KFCORE_HAND_DEMO_HAS_CPU)
#include "kfcore/vision_models/cpu.hpp"
#endif
#if defined(KFCORE_HAND_DEMO_HAS_TENSORRT)
#include "kfcore/vision_models/tensorrt.hpp"
#endif

#include <cmath>
#include <stdexcept>

namespace kfcore::hand_interaction::demo
{
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
    vision_models::TensorRtVisionOptions options;
    options.face_detection_score_threshold =
        arguments.face_detection_score_threshold;
    return vision_models::FaceMeshPipeline::create(
        vision_models::TensorRtFaceDetector::load(
            *arguments.face_detector_model, options),
        vision_models::TensorRtFaceLandmarker::load(
            *arguments.face_landmark_model, options),
        pipeline_options);
#else
    throw std::logic_error("TensorRT backend was not compiled into this demo");
#endif
}

} // namespace kfcore::hand_interaction::demo
