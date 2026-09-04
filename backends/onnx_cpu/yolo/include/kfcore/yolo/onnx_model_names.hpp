#pragma once

#include "kfcore/model_configuration.hpp"

namespace kfcore::yolo::onnx_model_names
{

inline constexpr const char* default_model_root =
    kfcore::model_configuration::model_root;
inline constexpr char person_detector[] = "yolov8s.onnx";

} // namespace kfcore::yolo::onnx_model_names
