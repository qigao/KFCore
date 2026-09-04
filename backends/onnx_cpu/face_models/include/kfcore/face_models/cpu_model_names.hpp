#pragma once

#include "kfcore/model_configuration.hpp"

namespace kfcore::face_models::cpu_model_names
{

inline constexpr const char* default_model_root =
    kfcore::model_configuration::model_root;
inline constexpr char face_detector[] = "yolov11n-face.onnx";
inline constexpr char face_mesh_landmarker[] =
    "MediaPipeFaceLandmarkDetector.onnx";
inline constexpr char face_68_landmarker[]   = "2dfan4.onnx";
inline constexpr char face_embedding[]       = "arcface_w600k_r50.onnx";
inline constexpr char age_gender_estimator[] = "age-gender.onnx";
inline constexpr char face_swapper[]         = "inswapper_128.onnx";
inline constexpr char face_restorer[]        = "gfpgan_1.4.onnx";
inline constexpr char face_swap_projection[] = "model_matrix.bin";

} // namespace kfcore::face_models::cpu_model_names
