#include "engine_contract.hpp"

#include "checked_size.hpp"

#include <array>
#include <limits>
#include <new>
#include <string>
#include <utility>

namespace kfcore::yolo {
namespace {

constexpr std::size_t kImageRank = 4;
constexpr std::size_t kFlatCountRank = 1;
constexpr std::size_t kEfficientNmsCountRank = 2;
constexpr std::size_t kBoxesRank = 3;
constexpr std::size_t kDetectionsRank = 2;
constexpr std::size_t kCompactDetectionsRank = 3;
constexpr std::int64_t kImageChannels = 3;
constexpr std::int64_t kBoxCoordinates = 4;
constexpr std::int64_t kCompactDetectionValues = 6;

[[noreturn]] void contract_error(const std::string& stage, const std::string& detail) {
    throw YoloError(YoloErrorCode::EngineContractMismatch, stage + " stage: " + detail);
}

[[noreturn]] void resource_error(const std::string& stage, const std::string& detail) {
    throw YoloError(YoloErrorCode::ResourceLimitExceeded, stage + " stage: " + detail);
}

[[noreturn]] void invalid_error(const std::string& stage, const std::string& detail) {
    throw YoloError(YoloErrorCode::InvalidArgument, stage + " stage: " + detail);
}

std::size_t element_size(TensorDataType data_type, const std::string& name) {
    switch (data_type) {
    case TensorDataType::Float32:
    case TensorDataType::Int32:
        return sizeof(std::uint32_t);
    case TensorDataType::Float16:
        return sizeof(std::uint16_t);
    }
    contract_error(name, "unknown data type");
}

void validate_names(const ContractNames& names) {
    const std::array<const std::string*, 6> all_names = {
        &names.images, &names.num_dets, &names.boxes,
        &names.scores, &names.labels, &names.detections};
    for (const std::string* name : all_names) {
        if (name->empty()) {
            contract_error("names", "required tensor name must not be empty");
        }
    }
    const auto require_unique = [](const auto& contract_names) {
        for (std::size_t index = 0; index < contract_names.size(); ++index) {
            for (std::size_t previous = 0; previous < index; ++previous) {
                if (*contract_names[index] == *contract_names[previous]) {
                    contract_error("names", "required tensor names must be unique");
                }
            }
        }
    };
    require_unique(std::array<const std::string*, 5> {
        &names.images, &names.num_dets, &names.boxes, &names.scores, &names.labels});
    require_unique(std::array<const std::string*, 2> {&names.images, &names.detections});
}

struct RequiredTensor {
    const std::string& name;
    TensorIoMode mode;
    const TensorDesc* descriptor = nullptr;
};

template <std::size_t TensorCount>
void bind_required_tensors(
    const EngineMetadata& metadata,
    std::array<RequiredTensor, TensorCount>* required
) {
    const TensorDesc* unexpected = nullptr;
    for (const TensorDesc& tensor : metadata.tensors) {
        bool matched = false;
        for (RequiredTensor& candidate : *required) {
            if (tensor.name == candidate.name) {
                if (candidate.descriptor != nullptr) {
                    contract_error(tensor.name, "duplicate tensor name");
                }
                candidate.descriptor = &tensor;
                matched = true;
                break;
            }
        }
        if (!matched) {
            unexpected = &tensor;
        }
    }
    for (const RequiredTensor& candidate : *required) {
        if (candidate.descriptor == nullptr) {
            contract_error(candidate.name, "required tensor is missing");
        }
        if (candidate.descriptor->mode != candidate.mode) {
            contract_error(candidate.name, "unexpected tensor I/O mode");
        }
    }
    if (unexpected != nullptr) {
        contract_error(unexpected->name.empty() ? "metadata" : unexpected->name,
                       "unexpected tensor name");
    }
}

void validate_profile_shape(
    const TensorDesc& tensor,
    std::size_t rank,
    bool allow_dynamic_spatial = false
) {
    if (tensor.min_shape.size() != rank || tensor.opt_shape.size() != rank ||
        tensor.max_shape.size() != rank) {
        contract_error(tensor.name, "profile rank does not match the detection contract");
    }
    for (std::size_t index = 0; index < rank; ++index) {
        const std::int64_t minimum = tensor.min_shape[index];
        const std::int64_t optimum = tensor.opt_shape[index];
        const std::int64_t maximum = tensor.max_shape[index];
        if (minimum <= 0 || optimum <= 0 || maximum <= 0) {
            contract_error(tensor.name, "profile dimensions must be positive");
        }
        if (minimum > optimum || optimum > maximum) {
            contract_error(tensor.name, "profile dimensions must satisfy min <= opt <= max");
        }
        const bool dynamic = minimum != optimum || optimum != maximum;
        const bool spatial_dimension = allow_dynamic_spatial && (index == 2 || index == 3);
        if (index != 0 && dynamic && !spatial_dimension) {
            contract_error(tensor.name,
                           "only batch and image spatial dimensions may be dynamic");
        }
    }
}

void validate_count_profile_shape(const TensorDesc& tensor) {
    const std::size_t rank = tensor.min_shape.size();
    if (rank != kFlatCountRank && rank != kEfficientNmsCountRank) {
        contract_error(tensor.name, "profile rank does not match the detection contract");
    }
    validate_profile_shape(tensor, rank);
    if (rank == kEfficientNmsCountRank &&
        (tensor.min_shape[1] != 1 || tensor.opt_shape[1] != 1 ||
         tensor.max_shape[1] != 1)) {
        contract_error(tensor.name, "trailing count dimension must be 1");
    }
}

void require_type(const TensorDesc& tensor, TensorDataType expected) {
    if (tensor.data_type != expected) {
        contract_error(tensor.name, "unexpected tensor data type");
    }
}

void validate_physical_layout(const TensorDesc& tensor) {
    if (tensor.physical_layout.format != TensorPhysicalFormat::Linear) {
        contract_error(tensor.name, "physical format must be linear");
    }
    if (tensor.physical_layout.vectorized_dimension != -1) {
        contract_error(tensor.name, "vectorized dimension must be -1");
    }
    if (tensor.physical_layout.components_per_element != 1) {
        contract_error(tensor.name, "components per element must be 1");
    }
    const std::size_t expected_bytes = element_size(tensor.data_type, tensor.name);
    if (tensor.physical_layout.bytes_per_component <= 0 ||
        static_cast<std::size_t>(tensor.physical_layout.bytes_per_component) != expected_bytes) {
        contract_error(tensor.name, "bytes per component must match the scalar data type");
    }
}

void require_same_profile_dimension(
    const TensorDesc& expected,
    const TensorDesc& actual,
    std::size_t index
) {
    if (expected.min_shape[index] != actual.min_shape[index] ||
        expected.opt_shape[index] != actual.opt_shape[index] ||
        expected.max_shape[index] != actual.max_shape[index]) {
        contract_error(actual.name, "profile dimension does not match " + expected.name);
    }
}

std::size_t checked_dimension(std::int64_t dimension, const std::string& name) {
    const auto maximum = static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)());
    if (static_cast<std::uintmax_t>(dimension) > maximum) {
        resource_error(name, "dimension cannot be represented as size_t");
    }
    return static_cast<std::size_t>(dimension);
}

ValidatedTensor validate_capacity(const TensorDesc& tensor) {
    std::size_t elements = 1;
    for (const std::int64_t dimension : tensor.max_shape) {
        std::size_t next = 0;
        if (!detail::checked_multiply_size(elements, checked_dimension(dimension, tensor.name), &next)) {
            resource_error(tensor.name, "maximum element count overflowed");
        }
        elements = next;
    }
    std::size_t bytes = 0;
    if (!detail::checked_multiply_size(elements, element_size(tensor.data_type, tensor.name), &bytes)) {
        resource_error(tensor.name, "maximum buffer byte count overflowed");
    }
    return {tensor, elements, bytes};
}

void validate_limits(const ContractLimits& limits) {
    if (limits.max_batch == 0 || limits.max_detections == 0 || limits.max_input_bytes == 0 ||
        limits.max_output_bytes == 0) {
        resource_error("limits", "all configured limits must be positive");
    }
}

const TensorDesc& validate_common_input(const EngineMetadata& metadata,
                                        const ContractNames& names,
                                        const ContractLimits& limits) {
    const TensorDesc* images = nullptr;
    for (const TensorDesc& tensor : metadata.tensors) {
        if (tensor.name == names.images) {
            if (images != nullptr) {
                contract_error(tensor.name, "duplicate tensor name");
            }
            images = &tensor;
        }
    }
    if (images == nullptr) {
        contract_error(names.images, "required tensor is missing");
    }
    if (images->mode != TensorIoMode::Input) {
        contract_error(images->name, "unexpected tensor I/O mode");
    }
    validate_profile_shape(*images, kImageRank, true);
    if (images->data_type != TensorDataType::Float16 &&
        images->data_type != TensorDataType::Float32) {
        contract_error(images->name, "input data type must be Float16 or Float32");
    }
    validate_physical_layout(*images);
    if (images->min_shape[1] != kImageChannels) {
        contract_error(images->name, "input channel dimension must be 3");
    }
    const std::size_t max_batch = checked_dimension(images->max_shape[0], images->name);
    if (max_batch > limits.max_batch) {
        resource_error(images->name, "maximum batch exceeds configured batch limit");
    }
    return *images;
}

ValidatedContract make_contract(const TensorDesc& images,
                                DetectionOutputLayout output_layout,
                                TensorDataType output_type,
                                std::int64_t max_detections,
                                std::size_t output_bytes,
                                ValidatedTensor validated_images,
                                DetectionOutputContract outputs) {
    return {
        images.min_shape[0], images.opt_shape[0], images.max_shape[0], max_detections,
        images.min_shape[2], images.opt_shape[2], images.max_shape[2],
        images.min_shape[3], images.opt_shape[3], images.max_shape[3],
        output_layout, images.data_type, output_type, validated_images.max_bytes, output_bytes,
        std::move(validated_images), std::move(outputs)};
}

}  // namespace

void validate_engine_io_tensor_count(std::int32_t tensor_count) {
    if (tensor_count != 2 && tensor_count != 5) {
        contract_error("metadata extraction",
                       "engine must expose exactly two or five named I/O tensors");
    }
}

ValidatedContract validate_engine_contract(
    const EngineMetadata& metadata,
    const ContractNames& names,
    const ContractLimits& limits
) {
    try {
        validate_names(names);
        validate_limits(limits);
        const TensorDesc& images = validate_common_input(metadata, names, limits);
        ValidatedTensor validated_images = validate_capacity(images);
        if (validated_images.max_bytes > limits.max_input_bytes) {
            resource_error(images.name, "maximum input buffer exceeds configured byte limit");
        }

        if (metadata.tensors.size() == 2) {
            std::array<RequiredTensor, 2> required = {{
                {names.images, TensorIoMode::Input},
                {names.detections, TensorIoMode::Output},
            }};
            bind_required_tensors(metadata, &required);
            const TensorDesc& detections = *required[1].descriptor;
            validate_profile_shape(detections, kCompactDetectionsRank);
            if (detections.data_type != TensorDataType::Float16 &&
                detections.data_type != TensorDataType::Float32) {
                contract_error(detections.name, "output data type must be Float16 or Float32");
            }
            validate_physical_layout(detections);
            if (detections.min_shape[2] != kCompactDetectionValues) {
                contract_error(detections.name, "each Compact NMS row must contain six values");
            }
            require_same_profile_dimension(images, detections, 0);

            const std::size_t max_detections =
                checked_dimension(detections.max_shape[1], detections.name);
            if (max_detections > limits.max_detections) {
                resource_error(detections.name,
                               "maximum detections exceeds configured detection limit");
            }
            ValidatedTensor validated_detections = validate_capacity(detections);
            if (validated_detections.max_bytes > limits.max_output_bytes) {
                resource_error("outputs", "maximum output buffers exceed configured byte limit");
            }
            const std::size_t output_bytes = validated_detections.max_bytes;
            return make_contract(
                images, DetectionOutputLayout::CompactNms, detections.data_type,
                detections.max_shape[1], output_bytes, std::move(validated_images),
                CompactNmsContract {std::move(validated_detections)});
        }

        std::array<RequiredTensor, 5> required = {{
            {names.images, TensorIoMode::Input},
            {names.num_dets, TensorIoMode::Output},
            {names.boxes, TensorIoMode::Output},
            {names.scores, TensorIoMode::Output},
            {names.labels, TensorIoMode::Output},
        }};
        bind_required_tensors(metadata, &required);

        const TensorDesc& num_dets = *required[1].descriptor;
        const TensorDesc& boxes = *required[2].descriptor;
        const TensorDesc& scores = *required[3].descriptor;
        const TensorDesc& labels = *required[4].descriptor;

        validate_count_profile_shape(num_dets);
        validate_profile_shape(boxes, kBoxesRank);
        validate_profile_shape(scores, kDetectionsRank);
        validate_profile_shape(labels, kDetectionsRank);

        require_type(num_dets, TensorDataType::Int32);
        require_type(labels, TensorDataType::Int32);
        if (boxes.data_type != TensorDataType::Float16 &&
            boxes.data_type != TensorDataType::Float32) {
            contract_error(boxes.name, "output data type must be Float16 or Float32");
        }
        require_type(scores, boxes.data_type);
        for (const TensorDesc* tensor : {&num_dets, &boxes, &scores, &labels}) {
            validate_physical_layout(*tensor);
        }
        if (boxes.min_shape[2] != kBoxCoordinates) {
            contract_error(boxes.name, "box coordinate dimension must be 4");
        }

        for (const TensorDesc* tensor : {&num_dets, &boxes, &scores, &labels}) {
            require_same_profile_dimension(images, *tensor, 0);
        }
        require_same_profile_dimension(boxes, scores, 1);
        require_same_profile_dimension(boxes, labels, 1);

        const std::size_t max_detections = checked_dimension(boxes.max_shape[1], boxes.name);
        if (max_detections > limits.max_detections) {
            resource_error(boxes.name, "maximum detections exceeds configured detection limit");
        }

        ValidatedTensor validated_num_dets = validate_capacity(num_dets);
        ValidatedTensor validated_boxes = validate_capacity(boxes);
        ValidatedTensor validated_scores = validate_capacity(scores);
        ValidatedTensor validated_labels = validate_capacity(labels);
        std::size_t output_bytes = 0;
        for (const ValidatedTensor* tensor : {
                 &validated_num_dets, &validated_boxes, &validated_scores, &validated_labels}) {
            std::size_t next = 0;
            if (!detail::checked_add_size(output_bytes, tensor->max_bytes, &next)) {
                resource_error(tensor->descriptor.name, "total output buffer byte count overflowed");
            }
            output_bytes = next;
        }
        if (output_bytes > limits.max_output_bytes) {
            resource_error("outputs", "maximum output buffers exceed configured byte limit");
        }

        return make_contract(
            images, DetectionOutputLayout::EfficientNms, boxes.data_type, boxes.max_shape[1],
            output_bytes, std::move(validated_images),
            EfficientNmsContract {std::move(validated_num_dets), std::move(validated_boxes),
                                  std::move(validated_scores), std::move(validated_labels)});
    } catch (const std::bad_alloc&) {
        resource_error("metadata", "metadata validation allocation failed");
    } catch (const std::length_error&) {
        resource_error("metadata", "metadata validation exceeded container capacity");
    }
}

SelectedInputSize select_input_size(
    const ValidatedContract& contract,
    const std::optional<std::array<std::int32_t, 2>>& configured_size
) {
    std::int64_t height = contract.opt_input_height;
    std::int64_t width = contract.opt_input_width;
    if (configured_size.has_value()) {
        height = (*configured_size)[0];
        width = (*configured_size)[1];
        if (height <= 0 || width <= 0) {
            invalid_error("detector options", "input_size dimensions must be positive");
        }
    }
    if (height < contract.min_input_height || height > contract.max_input_height ||
        width < contract.min_input_width || width > contract.max_input_width) {
        invalid_error("detector options", "input_size must be within the engine profile range");
    }
    if (height > (std::numeric_limits<std::int32_t>::max)() ||
        width > (std::numeric_limits<std::int32_t>::max)()) {
        resource_error("detector options", "input_size exceeds detector dimension range");
    }
    return {static_cast<std::int32_t>(height), static_cast<std::int32_t>(width)};
}

}  // namespace kfcore::yolo
