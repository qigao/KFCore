#include "contracts.hpp"

#include "kfcore/face_models/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace kfcore::face_models::detail
{
namespace
{

    using kfcore::tensorrt::DataType;
    using kfcore::tensorrt::MemoryKind;
    using kfcore::tensorrt::TensorDescriptor;
    using kfcore::tensorrt::TensorIoMode;
    using kfcore::tensorrt::TensorShape;

    constexpr std::size_t kFloatBytes = sizeof(float);
    constexpr std::size_t kFace68ValuesPerImage =
        kFace68LandmarkCount * static_cast<std::size_t>(kFace68LandmarkWidth);
    constexpr std::size_t kFace68HeatmapValues =
        kFace68LandmarkCount * static_cast<std::size_t>(kFace68HeatmapExtent) *
        static_cast<std::size_t>(kFace68HeatmapExtent);
    constexpr char kFace68ModelName[]    = "Face68";
    constexpr char kArcFaceModelName[]   = "ArcFace";
    constexpr char kAgeGenderModelName[] = "AgeGender";
    constexpr char kInSwapperModelName[] = "InSwapper";
    constexpr char kGfpGanModelName[]     = "GFPGAN";

    [[noreturn]] void throw_invalid(const std::string& model, const std::string& detail)
    {
        throw FaceModelError(FaceModelErrorCode::InvalidArgument,
                             model + " option validation stage: " + detail);
    }

    [[noreturn]] void throw_contract(const std::string& model, const std::string& detail)
    {
        throw FaceModelError(FaceModelErrorCode::ModelContractMismatch,
                             model + " contract validation stage: " + detail);
    }

    [[noreturn]] void throw_view(const std::string& model, const std::string& detail)
    {
        throw FaceModelError(FaceModelErrorCode::InvalidTensorView,
                             model + " input validation stage: " + detail);
    }

    [[noreturn]] void throw_resource(const std::string& model, const std::string& detail)
    {
        throw FaceModelError(FaceModelErrorCode::ResourceLimitExceeded,
                             model + " resource validation stage: " + detail);
    }

    std::size_t checked_multiply(std::size_t left, std::size_t right,
                                 const std::string& model, const char* subject)
    {
        if (left != 0 && right > (std::numeric_limits<std::size_t>::max)() / left)
        {
            throw_resource(model, std::string(subject) + " size overflow");
        }
        return left * right;
    }

    std::size_t checked_add(std::size_t left, std::size_t right,
                            const std::string& model, const char* subject)
    {
        if (right > (std::numeric_limits<std::size_t>::max)() - left)
        {
            throw_resource(model, std::string(subject) + " size overflow");
        }
        return left + right;
    }

    std::size_t positive_size(std::int64_t value, const std::string& model,
                              const std::string& detail)
    {
        if (value <= 0)
        {
            throw_contract(model, detail + " must be positive");
        }
        const auto maximum = static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)());
        if (static_cast<std::uintmax_t>(value) > maximum)
        {
            throw_resource(model, detail + " cannot be represented as size_t");
        }
        return static_cast<std::size_t>(value);
    }

    void validate_names(const std::string& model, const std::vector<std::string>& names,
                        std::size_t max_batch, std::size_t max_output_bytes)
    {
        if (max_batch == 0)
        {
            throw_resource(model, "max_batch must be positive");
        }
        if (max_batch > static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)()))
        {
            throw_resource(model, "max_batch cannot be represented as a tensor dimension");
        }
        if (max_output_bytes == 0)
        {
            throw_resource(model, "max_output_bytes must be positive");
        }
        for (std::size_t index = 0; index < names.size(); ++index)
        {
            if (names[index].empty())
            {
                throw_invalid(model, "tensor names must not be empty");
            }
            for (std::size_t previous = 0; previous < index; ++previous)
            {
                if (names[previous] == names[index])
                {
                    throw_invalid(model, "tensor names must be distinct");
                }
            }
        }
    }

    void validate_unique_metadata(const std::vector<TensorDescriptor>& tensors,
                                  const std::string& model)
    {
        for (std::size_t index = 0; index < tensors.size(); ++index)
        {
            if (tensors[index].name.empty())
            {
                throw_contract(model, "engine exposes an empty tensor name");
            }
            for (std::size_t previous = 0; previous < index; ++previous)
            {
                if (tensors[previous].name == tensors[index].name)
                {
                    throw_contract(model, "duplicate tensor name: " + tensors[index].name);
                }
            }
        }
    }

    const TensorDescriptor& required_tensor(const std::vector<TensorDescriptor>& tensors,
                                            const std::string& name,
                                            const std::string& model)
    {
        for (const TensorDescriptor& tensor : tensors)
        {
            if (tensor.name == name)
            {
                return tensor;
            }
        }
        throw_contract(model, "required tensor is missing: " + name);
    }

    const TensorDescriptor* optional_tensor(const std::vector<TensorDescriptor>& tensors,
                                            const std::string& name)
    {
        for (const TensorDescriptor& tensor : tensors)
        {
            if (tensor.name == name)
            {
                return &tensor;
            }
        }
        return nullptr;
    }

    void reject_unexpected(const std::vector<TensorDescriptor>& tensors,
                           const std::vector<std::string>& allowed, const std::string& model)
    {
        for (const TensorDescriptor& tensor : tensors)
        {
            if (std::find(allowed.begin(), allowed.end(), tensor.name) == allowed.end())
            {
                throw_contract(model, "unexpected tensor: " + tensor.name);
            }
        }
    }

    void validate_common_descriptor(const TensorDescriptor& tensor, TensorIoMode expected_mode,
                                    const std::string& model)
    {
        if (tensor.mode != expected_mode)
        {
            throw_contract(model, tensor.name + " has the wrong I/O mode");
        }
        if (tensor.data_type != DataType::Float32)
        {
            throw_contract(model, tensor.name + " must use FP32 elements");
        }
    }

    bool valid_batch_declaration(std::int64_t value) noexcept
    {
        return value == -1 || value > 0;
    }

    void validate_input(const TensorDescriptor& tensor,
                        const std::array<std::int64_t, 3>& fixed_dimensions,
                        const std::string& model)
    {
        validate_common_descriptor(tensor, TensorIoMode::Input, model);
        if (tensor.declared_shape.size() != 4)
        {
            throw_contract(model, tensor.name + " input rank must be 4");
        }
        if (!valid_batch_declaration(tensor.declared_shape[0]))
        {
            throw_contract(model, tensor.name + " batch must be fixed positive or declared -1");
        }
        static constexpr std::array<const char*, 3> kDimensionNames = {
            "channel dimension", "height dimension", "width dimension"
        };
        for (std::size_t index = 0; index < fixed_dimensions.size(); ++index)
        {
            if (tensor.declared_shape[index + 1U] != fixed_dimensions[index])
            {
                throw_contract(model, tensor.name + " " + kDimensionNames[index] + " must be " +
                                          std::to_string(fixed_dimensions[index]));
            }
        }
        if (!tensor.profile)
        {
            throw_contract(model, tensor.name + " requires profile 0 bounds");
        }
        const auto& profile = *tensor.profile;
        if (profile.minimum.size() != 4 || profile.optimum.size() != 4 ||
            profile.maximum.size() != 4)
        {
            throw_contract(model, tensor.name + " profile rank must be 4");
        }
        for (std::size_t index = 0; index < fixed_dimensions.size(); ++index)
        {
            const std::size_t dimension = index + 1U;
            if (profile.minimum[dimension] != fixed_dimensions[index] ||
                profile.optimum[dimension] != fixed_dimensions[index] ||
                profile.maximum[dimension] != fixed_dimensions[index])
            {
                throw_contract(model, tensor.name + " profile spatial and channel dimensions "
                                          "must be fixed");
            }
        }
        const std::int64_t minimum = profile.minimum[0];
        const std::int64_t optimum = profile.optimum[0];
        const std::int64_t maximum = profile.maximum[0];
        if (minimum <= 0 || optimum < minimum || maximum < optimum)
        {
            throw_contract(model, tensor.name + " profile batch must satisfy 0 < min <= opt <= max");
        }
        if (tensor.declared_shape[0] != -1 &&
            (minimum != tensor.declared_shape[0] || optimum != tensor.declared_shape[0] ||
             maximum != tensor.declared_shape[0]))
        {
            throw_contract(model, tensor.name + " fixed batch must match every profile bound");
        }
    }

    void validate_vector_input(const TensorDescriptor& tensor, const TensorShape& fixed_tail,
                               const std::string& model)
    {
        validate_common_descriptor(tensor, TensorIoMode::Input, model);
        const std::size_t expected_rank = fixed_tail.size() + 1U;
        if (tensor.declared_shape.size() != expected_rank)
        {
            throw_contract(model, tensor.name + " input rank must be " +
                                      std::to_string(expected_rank));
        }
        if (!valid_batch_declaration(tensor.declared_shape[0]))
        {
            throw_contract(model, tensor.name + " batch must be fixed positive or declared -1");
        }
        for (std::size_t index = 0; index < fixed_tail.size(); ++index)
        {
            if (tensor.declared_shape[index + 1U] != fixed_tail[index])
            {
                throw_contract(model, tensor.name + " input dimension " +
                                          std::to_string(index + 1U) + " must be " +
                                          std::to_string(fixed_tail[index]));
            }
        }
        if (!tensor.profile)
        {
            throw_contract(model, tensor.name + " requires profile 0 bounds");
        }
        const auto& profile = *tensor.profile;
        if (profile.minimum.size() != expected_rank || profile.optimum.size() != expected_rank ||
            profile.maximum.size() != expected_rank)
        {
            throw_contract(model, tensor.name + " profile rank must be " +
                                      std::to_string(expected_rank));
        }
        for (std::size_t index = 0; index < fixed_tail.size(); ++index)
        {
            const std::size_t dimension = index + 1U;
            if (profile.minimum[dimension] != fixed_tail[index] ||
                profile.optimum[dimension] != fixed_tail[index] ||
                profile.maximum[dimension] != fixed_tail[index])
            {
                throw_contract(model, tensor.name + " profile non-batch dimensions must be fixed");
            }
        }
        const std::int64_t minimum = profile.minimum[0];
        const std::int64_t optimum = profile.optimum[0];
        const std::int64_t maximum = profile.maximum[0];
        if (minimum <= 0 || optimum < minimum || maximum < optimum)
        {
            throw_contract(model, tensor.name + " profile batch must satisfy 0 < min <= opt <= max");
        }
        if (tensor.declared_shape[0] != -1 &&
            (minimum != tensor.declared_shape[0] || optimum != tensor.declared_shape[0] ||
             maximum != tensor.declared_shape[0]))
        {
            throw_contract(model, tensor.name + " fixed batch must match every profile bound");
        }
    }

    void validate_output(const TensorDescriptor& tensor, const TensorShape& fixed_tail,
                         const std::string& model)
    {
        validate_common_descriptor(tensor, TensorIoMode::Output, model);
        if (tensor.profile)
        {
            throw_contract(model, tensor.name + " output must not expose profile bounds");
        }
        if (tensor.declared_shape.size() != fixed_tail.size() + 1U)
        {
            throw_contract(model, tensor.name + " output rank must be " +
                                      std::to_string(fixed_tail.size() + 1U));
        }
        if (!valid_batch_declaration(tensor.declared_shape[0]))
        {
            throw_contract(model, tensor.name + " batch must be fixed positive or declared -1");
        }
        for (std::size_t index = 0; index < fixed_tail.size(); ++index)
        {
            if (tensor.declared_shape[index + 1U] != fixed_tail[index])
            {
                throw_contract(model, tensor.name + " output dimension " +
                                          std::to_string(index + 1U) + " must be " +
                                          std::to_string(fixed_tail[index]));
            }
        }
    }

    BatchBounds resolve_batch(const TensorDescriptor& input,
                              const std::vector<const TensorDescriptor*>& outputs,
                              std::size_t configured_max, const std::string& model)
    {
        const std::size_t profile_min =
            positive_size(input.profile->minimum[0], model, input.name + " minimum batch");
        const std::size_t profile_max =
            positive_size(input.profile->maximum[0], model, input.name + " maximum batch");
        std::size_t fixed_batch = input.declared_shape[0] == -1
                                      ? 0
                                      : positive_size(input.declared_shape[0], model,
                                                      input.name + " fixed batch");
        for (const TensorDescriptor* output : outputs)
        {
            if (output->declared_shape[0] == -1)
            {
                continue;
            }
            const std::size_t output_batch =
                positive_size(output->declared_shape[0], model, output->name + " fixed batch");
            if (fixed_batch != 0 && fixed_batch != output_batch)
            {
                throw_contract(model, "fixed input and output batch declarations disagree");
            }
            fixed_batch = output_batch;
        }

        if (fixed_batch != 0)
        {
            if (fixed_batch < profile_min || fixed_batch > profile_max)
            {
                throw_contract(model, "fixed output batch lies outside input profile 0 bounds");
            }
            if (fixed_batch > configured_max)
            {
                throw_resource(model, "fixed engine batch exceeds adapter max_batch");
            }
            return { fixed_batch, fixed_batch };
        }

        const std::size_t maximum = (std::min)(profile_max, configured_max);
        if (maximum < profile_min)
        {
            throw_resource(model, "adapter max_batch is below the input profile minimum");
        }
        return { profile_min, maximum };
    }

    std::size_t output_capacity(std::size_t batch, std::size_t values_per_image,
                                const std::string& model)
    {
        return checked_multiply(batch, values_per_image, model, "output element count");
    }

    void enforce_output_bytes(std::size_t float_count, std::size_t limit,
                              const std::string& model)
    {
        const std::size_t bytes = checked_multiply(float_count, kFloatBytes, model, "output bytes");
        if (bytes > limit)
        {
            throw_resource(model, "adapter output bytes exceed max_output_bytes");
        }
    }

    SingleOutputContract validate_single_output_contract(
        const std::vector<TensorDescriptor>& tensors, const std::string& input_name,
        const std::string& output_name, std::size_t max_batch, std::size_t max_output_bytes,
        const std::array<std::int64_t, 3>& input_dimensions, std::int64_t output_width,
        const std::string& model)
    {
        validate_unique_metadata(tensors, model);
        const TensorDescriptor& input_tensor = required_tensor(tensors, input_name, model);
        const TensorDescriptor& output_tensor = required_tensor(tensors, output_name, model);
        reject_unexpected(tensors, { input_name, output_name }, model);
        validate_input(input_tensor, input_dimensions, model);
        validate_output(output_tensor, { output_width }, model);

        SingleOutputContract result;
        result.input_name  = input_name;
        result.output_name = output_name;
        result.batch = resolve_batch(input_tensor, { &output_tensor }, max_batch, model);
        result.output_float_capacity =
            output_capacity(result.batch.maximum, static_cast<std::size_t>(output_width), model);
        enforce_output_bytes(result.output_float_capacity, max_output_bytes, model);
        return result;
    }

} // namespace

void validate_face68_options(const Face68Options& options)
{
    validate_names(kFace68ModelName,
                   { options.input_name, options.landmark_output_name,
                     options.heatmap_output_name },
                   options.max_batch, options.engine.max_output_bytes);
}

void validate_arcface_options(const ArcFaceOptions& options)
{
    validate_names(kArcFaceModelName, { options.input_name, options.output_name },
                   options.max_batch, options.engine.max_output_bytes);
}

void validate_age_gender_options(const AgeGenderOptions& options)
{
    validate_names(kAgeGenderModelName, { options.input_name, options.output_name },
                   options.max_batch, options.engine.max_output_bytes);
}

void validate_inswapper_options(const InSwapperOptions& options)
{
    validate_names(kInSwapperModelName,
                   { options.target_input_name, options.source_input_name, options.output_name },
                   options.max_batch, options.engine.max_output_bytes);
}

void validate_gfpgan_options(const GfpGanOptions& options)
{
    validate_names(kGfpGanModelName, { options.input_name, options.output_name },
                   options.max_batch, options.engine.max_output_bytes);
}

AdapterCallGuard::AdapterCallGuard(std::atomic_flag& in_use, const char* model_name)
    : in_use_(in_use)
{
    if (in_use_.test_and_set(std::memory_order_acquire))
    {
        throw FaceModelError(FaceModelErrorCode::InvalidArgument,
                             std::string(model_name) +
                                 " inference stage: overlapping calls on one adapter are "
                                 "unsupported");
    }
}

AdapterCallGuard::~AdapterCallGuard()
{
    in_use_.clear(std::memory_order_release);
}

BorrowedInputGuard::BorrowedInputGuard(kfcore::tensorrt::TensorView&       target,
                                       const kfcore::tensorrt::TensorView& source) noexcept
    : target_(target)
{
    target_.data        = source.data;
    target_.byte_size   = source.byte_size;
    target_.memory_kind = source.memory_kind;
}

BorrowedInputGuard::~BorrowedInputGuard() noexcept
{
    target_.data        = nullptr;
    target_.byte_size   = 0;
    target_.memory_kind = kfcore::tensorrt::MemoryKind::Host;
}

BorrowedOutputGuard::BorrowedOutputGuard(
    kfcore::tensorrt::MutableTensorView& target,
    const kfcore::tensorrt::MutableTensorView& source) noexcept
    : target_(target)
    , previous_data_(target.data)
    , previous_byte_size_(target.byte_size)
    , previous_memory_kind_(target.memory_kind)
{
    target_.data        = source.data;
    target_.byte_size   = source.byte_size;
    target_.memory_kind = source.memory_kind;
}

BorrowedOutputGuard::~BorrowedOutputGuard() noexcept
{
    target_.data        = previous_data_;
    target_.byte_size   = previous_byte_size_;
    target_.memory_kind = previous_memory_kind_;
}

Face68Contract validate_face68_contract(const std::vector<TensorDescriptor>& tensors,
                                        const Face68Options& options)
{
    const std::string model = kFace68ModelName;
    validate_face68_options(options);
    validate_unique_metadata(tensors, model);
    const TensorDescriptor& input_tensor = required_tensor(tensors, options.input_name, model);
    const TensorDescriptor& landmarks =
        required_tensor(tensors, options.landmark_output_name, model);
    const TensorDescriptor* heatmaps =
        optional_tensor(tensors, options.heatmap_output_name);
    reject_unexpected(tensors,
                      { options.input_name, options.landmark_output_name,
                        options.heatmap_output_name },
                      model);
    validate_input(input_tensor,
                   { kFaceModelInputChannels, kFace68InputExtent, kFace68InputExtent }, model);
    validate_output(landmarks,
                    { static_cast<std::int64_t>(kFace68LandmarkCount), kFace68LandmarkWidth },
                    model);
    if (heatmaps != nullptr)
    {
        validate_output(*heatmaps,
                        { static_cast<std::int64_t>(kFace68LandmarkCount),
                          kFace68HeatmapExtent, kFace68HeatmapExtent },
                        model);
    }

    Face68Contract result;
    result.input_name           = options.input_name;
    result.landmark_output_name = options.landmark_output_name;
    result.heatmap_output_name  = options.heatmap_output_name;
    result.has_heatmaps         = heatmaps != nullptr;
    std::vector<const TensorDescriptor*> outputs = { &landmarks };
    if (heatmaps != nullptr)
    {
        outputs.push_back(heatmaps);
    }
    result.batch = resolve_batch(input_tensor, outputs, options.max_batch, model);
    result.landmark_float_capacity =
        output_capacity(result.batch.maximum, kFace68ValuesPerImage, model);
    if (result.has_heatmaps)
    {
        result.heatmap_float_capacity =
            output_capacity(result.batch.maximum, kFace68HeatmapValues, model);
    }
    const std::size_t total_floats = checked_add(result.landmark_float_capacity,
                                                 result.heatmap_float_capacity, model,
                                                 "aggregate output element count");
    enforce_output_bytes(total_floats, options.engine.max_output_bytes, model);
    return result;
}

SingleOutputContract validate_arcface_contract(const std::vector<TensorDescriptor>& tensors,
                                               const ArcFaceOptions& options)
{
    validate_arcface_options(options);
    return validate_single_output_contract(tensors, options.input_name, options.output_name,
                                           options.max_batch, options.engine.max_output_bytes,
                                           { kFaceModelInputChannels, kArcFaceInputExtent,
                                             kArcFaceInputExtent },
                                           static_cast<std::int64_t>(kArcFaceEmbeddingLength),
                                           kArcFaceModelName);
}

SingleOutputContract validate_age_gender_contract(
    const std::vector<TensorDescriptor>& tensors, const AgeGenderOptions& options)
{
    validate_age_gender_options(options);
    return validate_single_output_contract(tensors, options.input_name, options.output_name,
                                           options.max_batch, options.engine.max_output_bytes,
                                           { kFaceModelInputChannels, kAgeGenderInputExtent,
                                             kAgeGenderInputExtent },
                                           static_cast<std::int64_t>(kAgeGenderLogitCount),
                                           kAgeGenderModelName);
}

InSwapperContract validate_inswapper_contract(
    const std::vector<TensorDescriptor>& tensors, const InSwapperOptions& options)
{
    const std::string model = kInSwapperModelName;
    validate_inswapper_options(options);
    validate_unique_metadata(tensors, model);
    const TensorDescriptor& target = required_tensor(tensors, options.target_input_name, model);
    const TensorDescriptor& source = required_tensor(tensors, options.source_input_name, model);
    const TensorDescriptor& output_tensor = required_tensor(tensors, options.output_name, model);
    reject_unexpected(tensors,
                      { options.target_input_name, options.source_input_name, options.output_name },
                      model);
    validate_input(target,
                   { kFaceModelInputChannels, kInSwapperInputExtent, kInSwapperInputExtent },
                   model);
    validate_vector_input(source, { static_cast<std::int64_t>(kInSwapperEmbeddingLength) }, model);
    validate_output(output_tensor,
                    { kFaceModelInputChannels, kInSwapperInputExtent, kInSwapperInputExtent },
                    model);

    const BatchBounds target_batch =
        resolve_batch(target, { &output_tensor }, options.max_batch, model);
    const BatchBounds source_batch =
        resolve_batch(source, { &output_tensor }, options.max_batch, model);
    if (target_batch.minimum != source_batch.minimum || target_batch.maximum != source_batch.maximum)
    {
        throw_contract(model, "target and source batch bounds disagree");
    }

    InSwapperContract result;
    result.target_input_name = options.target_input_name;
    result.source_input_name = options.source_input_name;
    result.output_name = options.output_name;
    result.batch = target_batch;
    result.output_float_capacity =
        output_capacity(result.batch.maximum, kInSwapperOutputElementCount, model);
    enforce_output_bytes(result.output_float_capacity, options.engine.max_output_bytes, model);
    return result;
}

SingleOutputContract validate_gfpgan_contract(const std::vector<TensorDescriptor>& tensors,
                                              const GfpGanOptions& options)
{
    const std::string model = kGfpGanModelName;
    validate_gfpgan_options(options);
    validate_unique_metadata(tensors, model);
    const TensorDescriptor& input_tensor = required_tensor(tensors, options.input_name, model);
    const TensorDescriptor& output_tensor = required_tensor(tensors, options.output_name, model);
    reject_unexpected(tensors, { options.input_name, options.output_name }, model);
    validate_input(input_tensor,
                   { kFaceModelInputChannels, kGfpGanInputExtent, kGfpGanInputExtent }, model);
    validate_output(output_tensor,
                    { kFaceModelInputChannels, kGfpGanInputExtent, kGfpGanInputExtent }, model);

    SingleOutputContract result;
    result.input_name = options.input_name;
    result.output_name = options.output_name;
    result.batch = resolve_batch(input_tensor, { &output_tensor }, options.max_batch, model);
    result.output_float_capacity =
        output_capacity(result.batch.maximum, kGfpGanOutputElementCount, model);
    enforce_output_bytes(result.output_float_capacity, options.engine.max_output_bytes, model);
    return result;
}

void validate_prepared_input(const kfcore::tensorrt::TensorView& input,
                             const std::string& expected_name, const BatchBounds& batch,
                             const std::array<std::int64_t, 3>& fixed_dimensions,
                             const char* model_name)
{
    validate_prepared_vector_input(
        input, expected_name, batch,
        std::vector<std::int64_t>(fixed_dimensions.begin(), fixed_dimensions.end()), model_name);
}

void validate_prepared_vector_input(const kfcore::tensorrt::TensorView& input,
                                    const std::string& expected_name,
                                    const BatchBounds& batch,
                                    const std::vector<std::int64_t>& fixed_dimensions,
                                    const char* model_name)
{
    const std::string model(model_name);
    if (input.name != expected_name)
    {
        throw_view(model, "expected exact tensor name " + expected_name);
    }
    if (input.data_type != DataType::Float32)
    {
        throw_view(model, "prepared tensor must use FP32 elements");
    }
    if (input.memory_kind != MemoryKind::Host && input.memory_kind != MemoryKind::CudaDevice)
    {
        throw_view(model, "prepared tensor memory kind is invalid");
    }
    if (input.data == nullptr)
    {
        throw_view(model, "prepared tensor data must not be null");
    }
    const std::size_t expected_rank = fixed_dimensions.size() + 1U;
    if (input.shape.size() != expected_rank)
    {
        throw_view(model, "prepared tensor rank must be " + std::to_string(expected_rank));
    }
    if (input.shape[0] <= 0)
    {
        throw_view(model, "prepared tensor batch must be positive");
    }
    const auto maximum_batch =
        static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)());
    if (static_cast<std::uintmax_t>(input.shape[0]) > maximum_batch)
    {
        throw_view(model, "prepared tensor batch cannot be represented as size_t");
    }
    const std::size_t input_batch = static_cast<std::size_t>(input.shape[0]);
    if (input_batch < batch.minimum || input_batch > batch.maximum)
    {
        throw_view(model, "prepared tensor batch is outside adapter bounds");
    }
    for (std::size_t index = 0; index < fixed_dimensions.size(); ++index)
    {
        if (input.shape[index + 1U] != fixed_dimensions[index])
        {
            throw_view(model, "prepared tensor has incorrect fixed dimensions");
        }
    }
    std::size_t required_elements = input_batch;
    for (const std::int64_t dimension : fixed_dimensions)
    {
        required_elements = checked_multiply(required_elements,
                                             static_cast<std::size_t>(dimension), model,
                                             "prepared tensor element count");
    }
    const std::size_t required_bytes =
        checked_multiply(required_elements, kFloatBytes, model, "prepared tensor bytes");
    if (input.byte_size < required_bytes)
    {
        throw_view(model, "prepared tensor capacity is smaller than its shape");
    }
}

void validate_prepared_output(const kfcore::tensorrt::MutableTensorView& output,
                              const std::string& expected_name, std::int64_t expected_batch,
                              const std::array<std::int64_t, 3>& fixed_dimensions,
                              const char* model_name)
{
    const std::string model(model_name);
    if (output.name != expected_name)
    {
        throw_view(model, "expected exact output tensor name " + expected_name);
    }
    if (output.data_type != DataType::Float32)
    {
        throw_view(model, "prepared output tensor must use FP32 elements");
    }
    if (output.memory_kind != MemoryKind::Host && output.memory_kind != MemoryKind::CudaDevice)
    {
        throw_view(model, "prepared output tensor memory kind is invalid");
    }
    if (output.data == nullptr)
    {
        throw_view(model, "prepared output tensor data must not be null");
    }
    if (expected_batch <= 0)
    {
        throw_view(model, "prepared output tensor batch must be positive");
    }
    if (output.shape.size() != fixed_dimensions.size() + 1U ||
        output.shape[0] != expected_batch)
    {
        throw_view(model, "prepared output tensor has incorrect batch or rank");
    }
    std::size_t required_elements = static_cast<std::size_t>(expected_batch);
    for (std::size_t index = 0; index < fixed_dimensions.size(); ++index)
    {
        if (output.shape[index + 1U] != fixed_dimensions[index])
        {
            throw_view(model, "prepared output tensor has incorrect fixed dimensions");
        }
        required_elements = checked_multiply(required_elements,
                                             static_cast<std::size_t>(fixed_dimensions[index]),
                                             model, "prepared output tensor element count");
    }
    const std::size_t required_bytes = checked_multiply(
        required_elements, kFloatBytes, model, "prepared output tensor bytes");
    if (output.byte_size < required_bytes)
    {
        throw_view(model, "prepared output tensor capacity is smaller than its shape");
    }
}

[[noreturn]] void rethrow_tensorrt(const kfcore::tensorrt::TensorRtError& error,
                                   const char* model_name, const char* stage)
{
    FaceModelErrorCode code = FaceModelErrorCode::RuntimeFailure;
    using kfcore::tensorrt::TensorRtErrorCode;
    switch (error.code())
    {
    case TensorRtErrorCode::InvalidArgument:
        code = FaceModelErrorCode::InvalidArgument;
        break;
    case TensorRtErrorCode::EngineContractMismatch:
        code = FaceModelErrorCode::ModelContractMismatch;
        break;
    case TensorRtErrorCode::InvalidTensorView:
        code = FaceModelErrorCode::InvalidTensorView;
        break;
    case TensorRtErrorCode::ResourceLimitExceeded:
        code = FaceModelErrorCode::ResourceLimitExceeded;
        break;
    case TensorRtErrorCode::FileIo:
    case TensorRtErrorCode::EngineDeserialize:
    case TensorRtErrorCode::TensorRtFailure:
    case TensorRtErrorCode::CudaFailure:
    case TensorRtErrorCode::ConcurrentExecution:
        break;
    }
    throw FaceModelError(code, std::string(model_name) + " " + stage + " stage: " + error.what());
}

[[noreturn]] void throw_allocation_failure(const char* model_name, const char* stage)
{
    throw FaceModelError(FaceModelErrorCode::ResourceLimitExceeded,
                         std::string(model_name) + " " + stage + " stage: allocation failed");
}

[[noreturn]] void throw_capacity_failure(const char* model_name, const char* stage)
{
    throw FaceModelError(FaceModelErrorCode::ResourceLimitExceeded,
                         std::string(model_name) + " " + stage +
                             " stage: container capacity exceeded");
}

} // namespace kfcore::face_models::detail
