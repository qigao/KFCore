#include "kfcore/hand_models/cuda_model_names.hpp"
#include "tinytest.hpp"

#include <string_view>

namespace names = kfcore::hand_models::cuda_model_names;

spec("CUDA hand model asset names")
{
    it("maps semantic roles to the fixed TensorRT engine assets")
    {
        check(names::engine_profile_directory ==
              std::string_view("hand_gesture_model/tensorrt"));
        check(names::palm_detector == std::string_view("palm_detection.engine"));
        check(names::hand_landmarker == std::string_view("hand_landmark.engine"));
        check(names::gesture_classifier ==
              std::string_view("keypoint_classifier.engine"));
    }
}
