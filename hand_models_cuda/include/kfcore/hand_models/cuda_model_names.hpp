#pragma once

#include "kfcore/model_configuration.hpp"

namespace kfcore::hand_models::cuda_model_names
{

inline constexpr const char* default_model_root =
    kfcore::model_configuration::model_root;
inline constexpr char engine_profile_directory[] =
    "hand_gesture_model/tensorrt";
inline constexpr const char* default_engine_profile =
    kfcore::model_configuration::tensorrt_engine_profile;
inline constexpr char palm_detector[]       = "palm_detection.engine";
inline constexpr char hand_landmarker[]     = "hand_landmark.engine";
inline constexpr char gesture_classifier[] = "keypoint_classifier.engine";

} // namespace kfcore::hand_models::cuda_model_names
