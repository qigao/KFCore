#include "hand_interaction_demo_face.hpp"

#include "kfcore/face_models/cpu.hpp"
#include "kfcore/face_models/tensorrt.hpp"

#include <cmath>
#include <stdexcept>

namespace kfcore::hand_interaction::demo
{
std::unique_ptr<face_models::FaceMeshPipeline> make_face_pipeline(
    const Arguments& arguments)
{
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
    face_models::FaceMeshPipelineOptions pipeline_options;
    pipeline_options.landmark_score_threshold =
        arguments.face_landmark_score_threshold;
    if (arguments.backend == Backend::Cpu)
    {
        face_models::CpuFaceMeshOptions options;
        options.face_detection_score_threshold =
            arguments.face_detection_score_threshold;
        return face_models::FaceMeshPipeline::create(
            face_models::CpuFaceDetector::load(options),
            face_models::CpuFaceLandmarker::load(options),
            pipeline_options);
    }
    face_models::TensorRtFaceMeshOptions options;
    options.face_detection_score_threshold =
        arguments.face_detection_score_threshold;
    return face_models::FaceMeshPipeline::create(
        face_models::TensorRtFaceDetector::load(options),
        face_models::TensorRtFaceLandmarker::load(options),
        pipeline_options);
}

} // namespace kfcore::hand_interaction::demo
