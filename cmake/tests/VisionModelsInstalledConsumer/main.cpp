#include <kfcore/vision_models/core.hpp>
#include <kfcore/vision_models/error.hpp>

#if defined(KFCORE_CONSUMER_HAS_HAND_INTERACTION)
#include <kfcore/hand_interaction/hand_interaction.hpp>
#endif

#if defined(KFCORE_CONSUMER_HAS_CPU)
#include <kfcore/vision_models/cpu.hpp>
#endif
#if defined(KFCORE_CONSUMER_HAS_TENSORRT)
#include <kfcore/vision_models/tensorrt.hpp>
#endif

int main()
{
    const kfcore::vision_models::VisionModelError error(
        kfcore::vision_models::VisionModelErrorCode::InvalidArgument,
        "installed consumer");
    bool valid = error.code() ==
                     kfcore::vision_models::VisionModelErrorCode::InvalidArgument &&
                 kfcore::vision_models::kHandLandmarkCount == 21 &&
                 kfcore::vision_models::kFaceLandmarkCount == 468;
    const auto borrowed = kfcore::vision_models::VisionFrameView::borrow({});
    valid = valid && borrowed.source.data == nullptr &&
            borrowed.compute.data == nullptr;
#if defined(KFCORE_CONSUMER_HAS_HAND_INTERACTION)
    const auto graph =
        kfcore::hand_interaction::build_hand_interaction_graph();
    valid = valid && !graph.actions.empty() && !graph.stateGraphs.empty();
#endif
#if defined(KFCORE_CONSUMER_HAS_CPU)
    const auto cpu_load = &kfcore::vision_models::CpuHandBackend::load;
    valid = valid && cpu_load != nullptr;
#endif
#if defined(KFCORE_CONSUMER_HAS_TENSORRT)
    const auto tensorrt_load = &kfcore::vision_models::TensorRtHandBackend::load;
    const auto face_detector_load =
        &kfcore::vision_models::TensorRtFaceDetector::load;
    const auto input_create =
        &kfcore::vision_models::TensorRtVisionInput::create;
    valid = valid && tensorrt_load != nullptr && face_detector_load != nullptr &&
            input_create != nullptr;
#endif
    return valid ? 0 : 1;
}
