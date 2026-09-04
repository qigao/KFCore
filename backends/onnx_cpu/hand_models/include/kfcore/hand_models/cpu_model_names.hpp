#pragma once

#include "kfcore/model_configuration.hpp"

namespace kfcore::hand_models::cpu_model_names
{

inline constexpr const char* default_model_root =
    kfcore::model_configuration::model_root;
inline constexpr char palm_detector[] =
    "hand_gesture_model/palm_detection/palm_detection_full_inf_post_192x192.onnx";
inline constexpr char hand_landmarker[] =
    "hand_gesture_model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx";
inline constexpr char gesture_classifier[] =
    "hand_gesture_model/keypoint_classifier/keypoint_classifier.onnx";

} // namespace kfcore::hand_models::cpu_model_names
