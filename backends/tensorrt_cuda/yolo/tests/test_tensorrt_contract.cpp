#include "engine_contract.hpp"
#include "tinytest.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace kfcore::yolo;

namespace {

TensorPhysicalLayout scalar_layout(TensorDataType data_type) {
    const std::int32_t scalar_bytes =
        data_type == TensorDataType::Float16 ? std::int32_t{2} : std::int32_t{4};
    return {TensorPhysicalFormat::Linear, -1, 1, scalar_bytes};
}

TensorDesc tensor(
    std::string name,
    TensorIoMode mode,
    TensorDataType data_type,
    std::vector<std::int64_t> min_shape,
    std::vector<std::int64_t> opt_shape,
    std::vector<std::int64_t> max_shape
) {
    return {std::move(name), mode, data_type, std::move(min_shape), std::move(opt_shape),
            std::move(max_shape), scalar_layout(data_type)};
}

EngineMetadata valid_fp32_metadata() {
    return {{
        tensor("images", TensorIoMode::Input, TensorDataType::Float32,
               {1, 3, 320, 480}, {2, 3, 640, 640}, {4, 3, 960, 1280}),
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

EngineMetadata valid_compact_nms_metadata() {
    return {{
        tensor("images", TensorIoMode::Input, TensorDataType::Float32,
               {1, 3, 320, 480}, {2, 3, 640, 640}, {4, 3, 960, 1280}),
        tensor("output0", TensorIoMode::Output, TensorDataType::Float32,
               {1, 300, 6}, {2, 300, 6}, {4, 300, 6}),
    }};
}

EngineMetadata valid_raw_yolo_metadata() {
    return {{
        tensor("images", TensorIoMode::Input, TensorDataType::Float32,
               {1, 3, 640, 640}, {1, 3, 640, 640}, {1, 3, 640, 640}),
        tensor("predictions", TensorIoMode::Output, TensorDataType::Float32,
               {1, 84, 8400}, {1, 84, 8400}, {1, 84, 8400}),
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

void check_physical_error(
    const EngineMetadata& metadata,
    const std::string& tensor_name,
    const std::string& expected_detail
) {
    bool threw = false;
    try {
        (void)validate_engine_contract(metadata, {}, limits());
    } catch (const YoloError& error) {
        threw = true;
        const std::string message(error.what());
        check(error.code() == YoloErrorCode::EngineContractMismatch);
        check(message.find(tensor_name) != std::string::npos);
        check(message.find(expected_detail) != std::string::npos);
    }
    check(threw);
}

const EfficientNmsContract& efficient_outputs(const ValidatedContract& contract) {
    return std::get<EfficientNmsContract>(contract.outputs);
}

}  // namespace

spec("TensorRT YOLO engine contract") {
    it("accepts only bounded known I/O tensor counts before metadata extraction") {
        validate_engine_io_tensor_count(2);
        validate_engine_io_tensor_count(5);

        for (const std::int32_t tensor_count : {
                 std::int32_t{0}, std::int32_t{1}, std::int32_t{3}, std::int32_t{4},
                 std::int32_t{6}, (std::numeric_limits<std::int32_t>::max)()}) {
            bool threw = false;
            try {
                validate_engine_io_tensor_count(tensor_count);
            } catch (const YoloError& error) {
                threw = true;
                check(error.code() == YoloErrorCode::EngineContractMismatch);
                check(std::string(error.what()).find("exactly two or five") != std::string::npos);
            }
            check(threw);
        }
    }

    it("accepts a named Compact NMS output and preserves its capacity") {
        const ValidatedContract contract =
            validate_engine_contract(valid_compact_nms_metadata(), {}, limits());

        check(contract.output_layout == DetectionOutputLayout::CompactNms);
        check(contract.min_batch == INT64_C(1));
        check(contract.opt_batch == INT64_C(2));
        check(contract.max_batch == INT64_C(4));
        check(contract.max_detections == INT64_C(300));
        check(contract.output_type == TensorDataType::Float32);
        const CompactNmsContract& compact = std::get<CompactNmsContract>(contract.outputs);
        check(compact.detections.descriptor.name == "output0");
        check(compact.detections.max_elements == std::size_t{7'200});
        check(compact.detections.max_bytes == std::size_t{28'800});
        check(contract.output_bytes == std::size_t{28'800});
    }

    it("accepts an FP16 Compact NMS output independently of the input type") {
        EngineMetadata metadata = valid_compact_nms_metadata();
        metadata.tensors[1].data_type = TensorDataType::Float16;
        metadata.tensors[1].physical_layout.bytes_per_component = 2;

        const ValidatedContract contract = validate_engine_contract(metadata, {}, limits());

        check(contract.output_layout == DetectionOutputLayout::CompactNms);
        check(contract.input_type == TensorDataType::Float32);
        check(contract.output_type == TensorDataType::Float16);
        check(contract.output_bytes == std::size_t{14'400});
    }

    it("accepts a raw YOLO detection head with a bounded post-NMS limit") {
        ContractNames names;
        names.detections = "predictions";
        const ValidatedContract contract =
            validate_engine_contract(valid_raw_yolo_metadata(), names, limits());

        check(contract.output_layout == DetectionOutputLayout::RawYolo);
        check(contract.max_detections == INT64_C(1000));
        const RawYoloContract& raw = std::get<RawYoloContract>(contract.outputs);
        check(raw.class_count == std::size_t{80});
        check(raw.candidate_count == std::size_t{8400});
        check(raw.predictions.max_elements == std::size_t{705600});
    }

    it("rejects raw YOLO class and candidate dimensions that vary by profile") {
        ContractNames names;
        names.detections = "predictions";
        EngineMetadata dynamic_classes = valid_raw_yolo_metadata();
        dynamic_classes.tensors[1].min_shape[1] = 83;
        bool threw = false;
        try {
            (void)validate_engine_contract(dynamic_classes, names, limits());
        } catch (const YoloError& error) {
            threw = true;
            check(error.code() == YoloErrorCode::EngineContractMismatch);
            check(std::string(error.what()).find("only batch") != std::string::npos);
        }
        check(threw);

        EngineMetadata dynamic_candidates = valid_raw_yolo_metadata();
        dynamic_candidates.tensors[1].min_shape[2] = 2100;
        threw = false;
        try {
            (void)validate_engine_contract(dynamic_candidates, names, limits());
        } catch (const YoloError& error) {
            threw = true;
            check(error.code() == YoloErrorCode::EngineContractMismatch);
            check(std::string(error.what()).find("only batch") != std::string::npos);
        }
        check(threw);
    }

    it("allows output names to overlap across mutually exclusive contracts") {
        EngineMetadata metadata = valid_fp32_metadata();
        metadata.tensors[4].name = "output0";
        ContractNames names;
        names.labels = "output0";

        const ValidatedContract contract = validate_engine_contract(metadata, names, limits());

        check(contract.output_layout == DetectionOutputLayout::EfficientNms);
        check(efficient_outputs(contract).labels.descriptor.name == "output0");
    }

    it("rejects malformed Compact NMS names shapes types layouts and limits") {
        EngineMetadata wrong_name = valid_compact_nms_metadata();
        wrong_name.tensors[1].name = "detections";
        check_error(wrong_name, "output0");

        EngineMetadata wrong_rank = valid_compact_nms_metadata();
        wrong_rank.tensors[1].min_shape.pop_back();
        wrong_rank.tensors[1].opt_shape.pop_back();
        wrong_rank.tensors[1].max_shape.pop_back();
        check_error(wrong_rank, "rank");

        EngineMetadata wrong_width = valid_compact_nms_metadata();
        wrong_width.tensors[1].min_shape[1] = 4;
        wrong_width.tensors[1].opt_shape[1] = 4;
        wrong_width.tensors[1].max_shape[1] = 4;
        wrong_width.tensors[1].min_shape[2] = 7;
        wrong_width.tensors[1].opt_shape[2] = 7;
        wrong_width.tensors[1].max_shape[2] = 7;
        check_error(wrong_width, "Compact NMS");

        EngineMetadata wrong_batch = valid_compact_nms_metadata();
        wrong_batch.tensors[1].max_shape[0] = 3;
        check_error(wrong_batch, "images");

        EngineMetadata wrong_type = valid_compact_nms_metadata();
        wrong_type.tensors[1].data_type = TensorDataType::Int32;
        check_error(wrong_type, "output0");

        EngineMetadata wrong_layout = valid_compact_nms_metadata();
        wrong_layout.tensors[1].physical_layout.format = TensorPhysicalFormat::Unsupported;
        check_error(wrong_layout, "physical format");

        ContractLimits limited = limits();
        limited.max_detections = 299;
        check_error(valid_compact_nms_metadata(), "detections",
                    YoloErrorCode::ResourceLimitExceeded, limited);

        limited = limits();
        limited.max_output_bytes = 28'799;
        check_error(valid_compact_nms_metadata(), "outputs",
                    YoloErrorCode::ResourceLimitExceeded, limited);
    }

    it("preserves named profile-zero dynamic batch and spatial dimensions") {
        const ValidatedContract contract = validate_engine_contract(valid_fp32_metadata(), {}, limits());

        check(contract.min_batch == INT64_C(1));
        check(contract.opt_batch == INT64_C(2));
        check(contract.max_batch == INT64_C(4));
        check(contract.max_detections == INT64_C(300));
        check(contract.min_input_height == INT64_C(320));
        check(contract.opt_input_height == INT64_C(640));
        check(contract.max_input_height == INT64_C(960));
        check(contract.min_input_width == INT64_C(480));
        check(contract.opt_input_width == INT64_C(640));
        check(contract.max_input_width == INT64_C(1280));
        check(contract.images.max_bytes == std::size_t{58'982'400});
        check(contract.input_type == TensorDataType::Float32);
        check(contract.output_type == TensorDataType::Float32);
    }

    it("accepts the EfficientNMS trailing singleton count dimension") {
        EngineMetadata metadata = valid_fp32_metadata();
        metadata.tensors[1].min_shape = {1, 1};
        metadata.tensors[1].opt_shape = {2, 1};
        metadata.tensors[1].max_shape = {4, 1};

        const ValidatedContract contract =
            validate_engine_contract(metadata, {}, limits());
        check(efficient_outputs(contract).num_dets.max_elements == std::size_t{4});
        check(efficient_outputs(contract).num_dets.max_bytes == std::size_t{16});
    }

    it("rejects a non-singleton EfficientNMS count dimension") {
        EngineMetadata metadata = valid_fp32_metadata();
        metadata.tensors[1].min_shape = {1, 2};
        metadata.tensors[1].opt_shape = {2, 2};
        metadata.tensors[1].max_shape = {4, 2};

        check_error(metadata, "trailing count dimension must be 1");
    }

    it("selects optimum spatial dimensions when no detector size is configured") {
        const ValidatedContract contract =
            validate_engine_contract(valid_fp32_metadata(), {}, limits());

        const SelectedInputSize selected = select_input_size(contract, std::nullopt);
        check(selected.height == 640);
        check(selected.width == 640);
    }

    it("accepts explicit minimum interior and maximum profile sizes") {
        const ValidatedContract contract =
            validate_engine_contract(valid_fp32_metadata(), {}, limits());
        const std::array<std::array<std::int32_t, 2>, 3> sizes = {{
            {{320, 480}},
            {{700, 900}},
            {{960, 1280}},
        }};

        for (const auto& size : sizes) {
            const SelectedInputSize selected = select_input_size(contract, size);
            check(selected.height == size[0]);
            check(selected.width == size[1]);
        }
    }

    it("rejects explicit spatial dimensions outside the profile range") {
        const ValidatedContract contract =
            validate_engine_contract(valid_fp32_metadata(), {}, limits());
        const std::array<std::array<std::int32_t, 2>, 6> invalid_sizes = {{
            {{0, 640}},
            {{640, 0}},
            {{319, 640}},
            {{961, 640}},
            {{640, 479}},
            {{640, 1281}},
        }};

        for (const auto& size : invalid_sizes) {
            bool threw = false;
            try {
                (void)select_input_size(contract, size);
            } catch (const YoloError& error) {
                threw = true;
                check(error.code() == YoloErrorCode::InvalidArgument);
                check(std::string(error.what()).find("input_size") != std::string::npos);
            }
            check(threw);
        }
    }

    it("accepts independent FP16 input and output types") {
        EngineMetadata metadata = valid_fp32_metadata();
        metadata.tensors[0].data_type = TensorDataType::Float16;
        metadata.tensors[2].data_type = TensorDataType::Float16;
        metadata.tensors[3].data_type = TensorDataType::Float16;
        metadata.tensors[0].physical_layout.bytes_per_component = 2;
        metadata.tensors[2].physical_layout.bytes_per_component = 2;
        metadata.tensors[3].physical_layout.bytes_per_component = 2;

        const ValidatedContract contract = validate_engine_contract(metadata, {}, limits());
        check(contract.input_type == TensorDataType::Float16);
        check(contract.output_type == TensorDataType::Float16);
    }

    it("accepts FP16 input with FP32 EfficientNMS outputs") {
        EngineMetadata metadata = valid_fp32_metadata();
        metadata.tensors[0].data_type = TensorDataType::Float16;
        metadata.tensors[0].physical_layout.bytes_per_component = 2;

        const ValidatedContract contract = validate_engine_contract(metadata, {}, limits());
        check(contract.input_type == TensorDataType::Float16);
        check(contract.output_type == TensorDataType::Float32);
        check(contract.images.max_bytes == std::size_t{29'491'200});
        check(efficient_outputs(contract).boxes.max_bytes == std::size_t{19'200});
    }

    it("accepts FP32 input with FP16 EfficientNMS outputs") {
        EngineMetadata metadata = valid_fp32_metadata();
        metadata.tensors[2].data_type = TensorDataType::Float16;
        metadata.tensors[3].data_type = TensorDataType::Float16;
        metadata.tensors[2].physical_layout.bytes_per_component = 2;
        metadata.tensors[3].physical_layout.bytes_per_component = 2;

        const ValidatedContract contract = validate_engine_contract(metadata, {}, limits());
        check(contract.input_type == TensorDataType::Float32);
        check(contract.output_type == TensorDataType::Float16);
        check(contract.images.max_bytes == std::size_t{58'982'400});
        check(efficient_outputs(contract).boxes.max_bytes == std::size_t{9'600});
    }

    it("binds a valid contract by name instead of tensor vector order") {
        EngineMetadata metadata = valid_fp32_metadata();
        std::swap(metadata.tensors[0], metadata.tensors[4]);
        std::swap(metadata.tensors[1], metadata.tensors[3]);

        const ValidatedContract contract = validate_engine_contract(metadata, {}, limits());
        check(contract.images.descriptor.name == "images");
        check(efficient_outputs(contract).num_dets.descriptor.name == "num_dets");
        check(efficient_outputs(contract).boxes.descriptor.name == "boxes");
        check(efficient_outputs(contract).scores.descriptor.name == "scores");
        check(efficient_outputs(contract).labels.descriptor.name == "labels");
    }

    it("rejects non-linear physical format on every required tensor") {
        const EngineMetadata baseline = valid_fp32_metadata();
        for (std::size_t index = 0; index < baseline.tensors.size(); ++index) {
            EngineMetadata metadata = baseline;
            metadata.tensors[index].physical_layout.format = TensorPhysicalFormat::Unsupported;
            check_physical_error(metadata, metadata.tensors[index].name, "physical format");
        }
    }

    it("rejects vectorized storage on every required tensor") {
        const EngineMetadata baseline = valid_fp32_metadata();
        for (std::size_t index = 0; index < baseline.tensors.size(); ++index) {
            EngineMetadata metadata = baseline;
            metadata.tensors[index].physical_layout.vectorized_dimension = 1;
            check_physical_error(metadata, metadata.tensors[index].name,
                                 "vectorized dimension");
        }
    }

    it("rejects non-scalar components on every required tensor") {
        const EngineMetadata baseline = valid_fp32_metadata();
        for (std::size_t index = 0; index < baseline.tensors.size(); ++index) {
            EngineMetadata metadata = baseline;
            metadata.tensors[index].physical_layout.components_per_element = 2;
            check_physical_error(metadata, metadata.tensors[index].name,
                                 "components per element");
        }
    }

    it("rejects component byte widths that differ from scalar dtype") {
        const EngineMetadata baseline = valid_fp32_metadata();
        for (std::size_t index = 0; index < baseline.tensors.size(); ++index) {
            EngineMetadata metadata = baseline;
            metadata.tensors[index].physical_layout.bytes_per_component = 1;
            check_physical_error(metadata, metadata.tensors[index].name,
                                 "bytes per component");
        }
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

    it("rejects dynamic channels outputs and inconsistent profile ordering") {
        EngineMetadata dynamic_channels = valid_fp32_metadata();
        dynamic_channels.tensors[0].min_shape[1] = 2;
        check_error(dynamic_channels, "images");

        EngineMetadata dynamic_output = valid_fp32_metadata();
        dynamic_output.tensors[2].min_shape[1] = 299;
        check_error(dynamic_output, "boxes");

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
