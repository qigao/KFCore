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
constexpr std::size_t kCountRank = 1;
constexpr std::size_t kBoxesRank = 3;
constexpr std::size_t kDetectionsRank = 2;
constexpr std::int64_t kImageChannels = 3;
constexpr std::int64_t kBoxCoordinates = 4;

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
    const std::array<const std::string*, 5> all_names = {
        &names.images, &names.num_dets, &names.boxes, &names.scores, &names.labels};
    for (std::size_t index = 0; index < all_names.size(); ++index) {
        if (all_names[index]->empty()) {
            contract_error("names", "required tensor name must not be empty");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (*all_names[index] == *all_names[previous]) {
                contract_error("names", "required tensor names must be unique");
            }
        }
    }
}

struct RequiredTensor {
    const std::string& name;
    TensorIoMode mode;
    const TensorDesc* descriptor = nullptr;
};

void bind_required_tensors(
    const EngineMetadata& metadata,
    std::array<RequiredTensor, 5>* required
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
        contract_error(tensor.name, "profile rank does not match the EfficientNMS contract");
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

}  // namespace

ValidatedContract validate_engine_contract(
    const EngineMetadata& metadata,
    const ContractNames& names,
    const ContractLimits& limits
) {
    try {
        validate_names(names);
        validate_limits(limits);
        std::array<RequiredTensor, 5> required = {{
            {names.images, TensorIoMode::Input},
            {names.num_dets, TensorIoMode::Output},
            {names.boxes, TensorIoMode::Output},
            {names.scores, TensorIoMode::Output},
            {names.labels, TensorIoMode::Output},
        }};
        bind_required_tensors(metadata, &required);

        const TensorDesc& images = *required[0].descriptor;
        const TensorDesc& num_dets = *required[1].descriptor;
        const TensorDesc& boxes = *required[2].descriptor;
        const TensorDesc& scores = *required[3].descriptor;
        const TensorDesc& labels = *required[4].descriptor;

        validate_profile_shape(images, kImageRank, true);
        validate_profile_shape(num_dets, kCountRank);
        validate_profile_shape(boxes, kBoxesRank);
        validate_profile_shape(scores, kDetectionsRank);
        validate_profile_shape(labels, kDetectionsRank);

        if (images.data_type != TensorDataType::Float16 && images.data_type != TensorDataType::Float32) {
            contract_error(images.name, "input data type must be Float16 or Float32");
        }
        require_type(num_dets, TensorDataType::Int32);
        require_type(labels, TensorDataType::Int32);
        require_type(boxes, images.data_type);
        require_type(scores, images.data_type);
        for (const TensorDesc* tensor : {&images, &num_dets, &boxes, &scores, &labels}) {
            validate_physical_layout(*tensor);
        }
        if (images.min_shape[1] != kImageChannels) {
            contract_error(images.name, "input channel dimension must be 3");
        }
        if (boxes.min_shape[2] != kBoxCoordinates) {
            contract_error(boxes.name, "box coordinate dimension must be 4");
        }

        for (const TensorDesc* tensor : {&num_dets, &boxes, &scores, &labels}) {
            require_same_profile_dimension(images, *tensor, 0);
        }
        require_same_profile_dimension(boxes, scores, 1);
        require_same_profile_dimension(boxes, labels, 1);

        const std::size_t max_batch = checked_dimension(images.max_shape[0], images.name);
        const std::size_t max_detections = checked_dimension(boxes.max_shape[1], boxes.name);
        if (max_batch > limits.max_batch) {
            resource_error(images.name, "maximum batch exceeds configured batch limit");
        }
        if (max_detections > limits.max_detections) {
            resource_error(boxes.name, "maximum detections exceeds configured detection limit");
        }

        ValidatedTensor validated_images = validate_capacity(images);
        ValidatedTensor validated_num_dets = validate_capacity(num_dets);
        ValidatedTensor validated_boxes = validate_capacity(boxes);
        ValidatedTensor validated_scores = validate_capacity(scores);
        ValidatedTensor validated_labels = validate_capacity(labels);
        if (validated_images.max_bytes > limits.max_input_bytes) {
            resource_error(images.name, "maximum input buffer exceeds configured byte limit");
        }
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

        return {
            images.min_shape[0], images.opt_shape[0], images.max_shape[0], boxes.max_shape[1],
            images.min_shape[2], images.opt_shape[2], images.max_shape[2],
            images.min_shape[3], images.opt_shape[3], images.max_shape[3],
            images.data_type, validated_images.max_bytes, output_bytes,
            std::move(validated_images), std::move(validated_num_dets),
            std::move(validated_boxes), std::move(validated_scores), std::move(validated_labels)};
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
