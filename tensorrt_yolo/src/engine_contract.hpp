#pragma once

#include "kfcore/yolo/error.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kfcore::yolo {

enum class TensorDataType { Float32, Float16, Int32 };
enum class TensorIoMode { Input, Output };

struct TensorDesc {
    std::string name;
    TensorIoMode mode;
    TensorDataType data_type;
    std::vector<std::int64_t> min_shape;
    std::vector<std::int64_t> opt_shape;
    std::vector<std::int64_t> max_shape;
};

struct EngineMetadata {
    std::vector<TensorDesc> tensors;
};

struct ContractNames {
    std::string images = "images";
    std::string num_dets = "num_dets";
    std::string boxes = "boxes";
    std::string scores = "scores";
    std::string labels = "labels";
};

struct ContractLimits {
    std::size_t max_batch = 16;
    std::size_t max_detections = 1000;
    std::size_t max_input_bytes = 256U * 1024U * 1024U;
    std::size_t max_output_bytes = 16U * 1024U * 1024U;
};

struct ValidatedTensor {
    TensorDesc descriptor;
    std::size_t max_elements;
    std::size_t max_bytes;
};

struct ValidatedContract {
    std::int64_t min_batch;
    std::int64_t opt_batch;
    std::int64_t max_batch;
    std::int64_t max_detections;
    std::int64_t input_height;
    std::int64_t input_width;
    TensorDataType floating_point_type;
    std::size_t input_bytes;
    std::size_t output_bytes;
    ValidatedTensor images;
    ValidatedTensor num_dets;
    ValidatedTensor boxes;
    ValidatedTensor scores;
    ValidatedTensor labels;
};

ValidatedContract validate_engine_contract(
    const EngineMetadata& metadata,
    const ContractNames& names,
    const ContractLimits& limits
);

}  // namespace kfcore::yolo
