#include "engine_contract.hpp"
#include "tinytest.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace kfcore::yolo;

namespace {

TensorDesc tensor(
    std::string name,
    TensorIoMode mode,
    TensorDataType data_type,
    std::vector<std::int64_t> min_shape,
    std::vector<std::int64_t> opt_shape,
    std::vector<std::int64_t> max_shape
) {
    return {std::move(name), mode, data_type, std::move(min_shape), std::move(opt_shape),
            std::move(max_shape)};
}

EngineMetadata valid_fp32_metadata() {
    return {{
        tensor("images", TensorIoMode::Input, TensorDataType::Float32,
               {1, 3, 640, 640}, {2, 3, 640, 640}, {4, 3, 640, 640}),
        tensor("num_dets", TensorIoMode::Output, TensorDataType::Int32,
               {1}, {2}, {4}),
        tensor("boxes", TensorIoMode::Output, TensorDataType::Float32,
               {1, 300, 4}, {2, 300, 4}, {4, 300, 4}),
        tensor("scores", TensorIoMode::Output, TensorDataType::Float32,
               {1, 300}, {2, 300}, {4, 300}),
        tensor("labels", TensorIoMode::Output, TensorDataType::Int32,
               {1, 300}, {2, 300}, {4, 300}),
    }};
}

ContractLimits limits() {
    return {};
}

void check_error(
    const EngineMetadata& metadata,
    const std::string& expected_message,
    YoloErrorCode expected_code = YoloErrorCode::EngineContractMismatch,
    ContractLimits selected_limits = limits()
) {
    bool threw = false;
    try {
        (void)validate_engine_contract(metadata, {}, selected_limits);
    } catch (const YoloError& error) {
        threw = true;
        check(error.code() == expected_code);
        check(std::string(error.what()).find(expected_message) != std::string::npos);
    }
    check(threw);
}

}  // namespace

spec("TensorRT YOLO engine contract") {
    it("accepts named dynamic-batch NCHW EfficientNMS tensors") {
        const ValidatedContract contract = validate_engine_contract(valid_fp32_metadata(), {}, limits());

        check(contract.max_batch == INT64_C(4));
        check(contract.max_detections == INT64_C(300));
        check(contract.input_height == INT64_C(640));
        check(contract.input_width == INT64_C(640));
        check(contract.floating_point_type == TensorDataType::Float32);
    }

    it("accepts FP16 boxes and scores") {
        EngineMetadata metadata = valid_fp32_metadata();
        metadata.tensors[0].data_type = TensorDataType::Float16;
        metadata.tensors[2].data_type = TensorDataType::Float16;
        metadata.tensors[3].data_type = TensorDataType::Float16;

        const ValidatedContract contract = validate_engine_contract(metadata, {}, limits());
        check(contract.floating_point_type == TensorDataType::Float16);
    }

    it("binds a valid contract by name instead of tensor vector order") {
        EngineMetadata metadata = valid_fp32_metadata();
        std::swap(metadata.tensors[0], metadata.tensors[4]);
        std::swap(metadata.tensors[1], metadata.tensors[3]);

        const ValidatedContract contract = validate_engine_contract(metadata, {}, limits());
        check(contract.images.descriptor.name == "images");
        check(contract.num_dets.descriptor.name == "num_dets");
        check(contract.boxes.descriptor.name == "boxes");
        check(contract.scores.descriptor.name == "scores");
        check(contract.labels.descriptor.name == "labels");
    }

    it("rejects missing duplicate and extra tensors by name") {
        EngineMetadata missing = valid_fp32_metadata();
        missing.tensors.pop_back();
        check_error(missing, "labels");

        EngineMetadata duplicate = valid_fp32_metadata();
        duplicate.tensors.push_back(duplicate.tensors[1]);
        check_error(duplicate, "num_dets");

        EngineMetadata extra = valid_fp32_metadata();
        extra.tensors.push_back(tensor("raw_head", TensorIoMode::Output, TensorDataType::Float32,
                                       {1}, {2}, {4}));
        check_error(extra, "raw_head");
    }

    it("rejects binding-order lookalikes with wrong names") {
        EngineMetadata metadata = valid_fp32_metadata();
        metadata.tensors[1].name = "output0";

        check_error(metadata, "num_dets");
    }

    it("rejects tensor mode type rank and shape mismatches") {
        EngineMetadata wrong_mode = valid_fp32_metadata();
        wrong_mode.tensors[0].mode = TensorIoMode::Output;
        check_error(wrong_mode, "images");

        EngineMetadata wrong_input_type = valid_fp32_metadata();
        wrong_input_type.tensors[0].data_type = TensorDataType::Int32;
        check_error(wrong_input_type, "images");

        EngineMetadata wrong_num_type = valid_fp32_metadata();
        wrong_num_type.tensors[1].data_type = TensorDataType::Float32;
        check_error(wrong_num_type, "num_dets");

        EngineMetadata wrong_labels_type = valid_fp32_metadata();
        wrong_labels_type.tensors[4].data_type = TensorDataType::Float32;
        check_error(wrong_labels_type, "labels");

        EngineMetadata wrong_output_mode = valid_fp32_metadata();
        wrong_output_mode.tensors[1].mode = TensorIoMode::Input;
        check_error(wrong_output_mode, "num_dets");

        EngineMetadata wrong_float_pair = valid_fp32_metadata();
        wrong_float_pair.tensors[3].data_type = TensorDataType::Float16;
        check_error(wrong_float_pair, "scores");

        EngineMetadata wrong_rank = valid_fp32_metadata();
        wrong_rank.tensors[2].max_shape.pop_back();
        check_error(wrong_rank, "boxes");

        EngineMetadata wrong_box_width = valid_fp32_metadata();
        wrong_box_width.tensors[2].min_shape[2] = 5;
        wrong_box_width.tensors[2].opt_shape[2] = 5;
        wrong_box_width.tensors[2].max_shape[2] = 5;
        check_error(wrong_box_width, "boxes");
    }

    it("rejects non-batch dynamic dimensions and inconsistent profiles") {
        EngineMetadata dynamic_height = valid_fp32_metadata();
        dynamic_height.tensors[0].min_shape[2] = 320;
        check_error(dynamic_height, "images");

        EngineMetadata profile_order = valid_fp32_metadata();
        profile_order.tensors[3].opt_shape[0] = 5;
        check_error(profile_order, "scores");

        EngineMetadata inconsistent_batch = valid_fp32_metadata();
        inconsistent_batch.tensors[4].max_shape[0] = 3;
        check_error(inconsistent_batch, "labels");

        EngineMetadata inconsistent_detections = valid_fp32_metadata();
        inconsistent_detections.tensors[3].min_shape[1] = 299;
        inconsistent_detections.tensors[3].opt_shape[1] = 299;
        inconsistent_detections.tensors[3].max_shape[1] = 299;
        check_error(inconsistent_detections, "profile dimension does not match boxes");
    }

    it("rejects non-positive dimensions") {
        EngineMetadata zero_dimension = valid_fp32_metadata();
        zero_dimension.tensors[0].min_shape[3] = 0;
        check_error(zero_dimension, "images");

        EngineMetadata negative_dimension = valid_fp32_metadata();
        negative_dimension.tensors[2].max_shape[1] = -1;
        check_error(negative_dimension, "boxes");
    }

    it("rejects configured limits and checked byte-count overflow") {
        EngineMetadata metadata = valid_fp32_metadata();
        ContractLimits limited = limits();
        limited.max_batch = 3;
        check_error(metadata, "batch", YoloErrorCode::ResourceLimitExceeded, limited);

        limited = limits();
        limited.max_detections = 299;
        check_error(metadata, "detections", YoloErrorCode::ResourceLimitExceeded, limited);

        limited = limits();
        limited.max_input_bytes = 1;
        check_error(metadata, "images", YoloErrorCode::ResourceLimitExceeded, limited);

        limited = limits();
        limited.max_output_bytes = 1;
        check_error(metadata, "outputs", YoloErrorCode::ResourceLimitExceeded, limited);

        limited = limits();
        limited.max_output_bytes = 0;
        check_error(metadata, "limits", YoloErrorCode::ResourceLimitExceeded, limited);

        EngineMetadata overflow = valid_fp32_metadata();
        for (TensorDesc& tensor_desc : overflow.tensors) {
            tensor_desc.min_shape[0] = 1;
            tensor_desc.opt_shape[0] = 1;
            tensor_desc.max_shape[0] = 1;
        }
        const std::int64_t excessive_detections =
            (std::numeric_limits<std::int64_t>::max)() / 4;
        for (std::size_t tensor_index : {std::size_t{2}, std::size_t{3}, std::size_t{4}}) {
            overflow.tensors[tensor_index].min_shape[1] = excessive_detections;
            overflow.tensors[tensor_index].opt_shape[1] = excessive_detections;
            overflow.tensors[tensor_index].max_shape[1] = excessive_detections;
        }
        limited = limits();
        limited.max_batch = (std::numeric_limits<std::size_t>::max)();
        limited.max_detections = (std::numeric_limits<std::size_t>::max)();
        limited.max_input_bytes = (std::numeric_limits<std::size_t>::max)();
        limited.max_output_bytes = (std::numeric_limits<std::size_t>::max)();
        check_error(overflow, "maximum buffer byte count overflowed",
                    YoloErrorCode::ResourceLimitExceeded, limited);
    }

    it("rejects overflowing total output bytes") {
        EngineMetadata metadata = valid_fp32_metadata();
        for (TensorDesc& tensor_desc : metadata.tensors) {
            tensor_desc.min_shape[0] = 1;
            tensor_desc.opt_shape[0] = 1;
            tensor_desc.max_shape[0] = 1;
        }
        constexpr std::int64_t excessive_detections = INT64_C(800000000000000000);
        for (std::size_t tensor_index : {std::size_t{2}, std::size_t{3}, std::size_t{4}}) {
            metadata.tensors[tensor_index].min_shape[1] = excessive_detections;
            metadata.tensors[tensor_index].opt_shape[1] = excessive_detections;
            metadata.tensors[tensor_index].max_shape[1] = excessive_detections;
        }
        ContractLimits unlimited = limits();
        unlimited.max_batch = (std::numeric_limits<std::size_t>::max)();
        unlimited.max_detections = (std::numeric_limits<std::size_t>::max)();
        unlimited.max_input_bytes = (std::numeric_limits<std::size_t>::max)();
        unlimited.max_output_bytes = (std::numeric_limits<std::size_t>::max)();

        check_error(metadata, "labels", YoloErrorCode::ResourceLimitExceeded, unlimited);
    }
}
