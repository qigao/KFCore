#pragma once

namespace kfcore::face_models::cuda_model_names
{

inline constexpr char engine_profile_directory[] = "tensorrt";
inline constexpr char face_detector[]             = "yolov12n-face.engine";
inline constexpr char face_mesh_landmarker[]      = "face_landmark.engine";
inline constexpr char face_68_landmarker[]        = "2dfan4.engine";
inline constexpr char face_embedding[]            = "arcface_w600k_r50.engine";
inline constexpr char age_gender_estimator[]      = "age-gender.engine";
inline constexpr char face_swapper[]              = "inswapper_128.engine";
inline constexpr char face_restorer[]             = "gfpgan_1.4.engine";
inline constexpr char face_swap_projection[]      = "model_matrix.bin";

} // namespace kfcore::face_models::cuda_model_names
