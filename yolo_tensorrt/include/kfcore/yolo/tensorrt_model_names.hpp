#pragma once

#include "kfcore/model_configuration.hpp"

namespace kfcore::yolo::tensorrt_model_names
{

inline constexpr const char* default_model_root =
    kfcore::model_configuration::model_root;
inline constexpr char engine_profile_directory[] = "tensorrt";
inline constexpr const char* default_engine_profile =
    kfcore::model_configuration::tensorrt_engine_profile;
inline constexpr char person_detector[]           = "yolov8s.engine";

} // namespace kfcore::yolo::tensorrt_model_names
