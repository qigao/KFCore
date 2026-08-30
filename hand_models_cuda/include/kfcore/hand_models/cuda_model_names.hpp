#pragma once

namespace kfcore::hand_models::cuda_model_names
{

inline constexpr char engine_profile_directory[] =
    "hand_gesture_model/tensorrt";
inline constexpr char palm_detector[]       = "palm_detection.engine";
inline constexpr char hand_landmarker[]     = "hand_landmark.engine";
inline constexpr char gesture_classifier[] = "keypoint_classifier.engine";

} // namespace kfcore::hand_models::cuda_model_names
