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
    valid = valid && tensorrt_load != nullptr;
#endif
    return valid ? 0 : 1;
}
