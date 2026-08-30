#include "kfcore/hand_models/cpu_model_names.hpp"
#include "tinytest.hpp"

#include <string_view>

namespace names = kfcore::hand_models::cpu_model_names;

spec("CPU hand model asset names")
{
    it("maps semantic roles to the fixed yolo-models ONNX assets")
    {
        check(names::palm_detector == std::string_view(
            "hand_gesture_model/palm_detection/palm_detection_full_inf_post_192x192.onnx"));
        check(names::hand_landmarker == std::string_view(
            "hand_gesture_model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx"));
        check(names::gesture_classifier == std::string_view(
            "hand_gesture_model/keypoint_classifier/keypoint_classifier.onnx"));
    }
}
