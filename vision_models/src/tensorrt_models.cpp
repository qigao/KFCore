#include "tensorrt_models.hpp"

#include "kfcore/vision_models/types.hpp"

#include <array>
#include <string>
#include <utility>

namespace kfcore::vision_models::detail
{
namespace
{

using tensorrt::DataType;
using tensorrt::TensorDescriptor;
using tensorrt::TensorIoMode;
using tensorrt::TensorShape;

constexpr char kPalmOutputName[] =
    "pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y";

struct ExpectedTensor
{
    const char*  name;
    TensorIoMode mode;
    DataType     type;
    TensorShape  declared_shape;
};

tensorrt::EngineOptions engine_options(const TensorRtModelOptions& options)
{
    tensorrt::EngineOptions result;
    result.device_id                   = options.device_id;
    result.max_serialized_engine_bytes = options.max_engine_bytes;
    result.max_input_bytes             = options.max_tensor_bytes;
    result.max_output_bytes            = options.max_output_bytes;
    return result;
}

const TensorDescriptor& require_tensor(const std::vector<TensorDescriptor>& actual,
                                       const ExpectedTensor& expected,
                                       const char* model)
{
    for (const TensorDescriptor& tensor : actual)
    {
        if (tensor.name != expected.name)
        {
            continue;
        }
        if (tensor.mode != expected.mode || tensor.data_type != expected.type ||
            tensor.declared_shape != expected.declared_shape)
        {
            throw tensorrt::TensorRtError(
                tensorrt::TensorRtErrorCode::EngineContractMismatch,
                std::string(model) + " engine tensor contract mismatch: " + expected.name);
        }
        return tensor;
    }
    throw tensorrt::TensorRtError(
        tensorrt::TensorRtErrorCode::EngineContractMismatch,
        std::string(model) + " engine lacks tensor: " + expected.name);
}

void validate_contract(const std::shared_ptr<const tensorrt::Engine>& engine,
                       const std::vector<ExpectedTensor>& expected,
                       const char* model)
{
    const auto& actual = engine->tensors();
    if (actual.size() != expected.size())
    {
        throw tensorrt::TensorRtError(
            tensorrt::TensorRtErrorCode::EngineContractMismatch,
            std::string(model) + " engine exposes an unexpected tensor count");
    }
    for (const ExpectedTensor& tensor : expected)
    {
        (void)require_tensor(actual, tensor, model);
    }
}

void validate_dynamic_batch(const std::shared_ptr<const tensorrt::Engine>& engine,
                            const char* input_name, std::size_t required_batch,
                            const char* model)
{
    for (const TensorDescriptor& tensor : engine->tensors())
    {
        if (tensor.name == input_name && tensor.mode == TensorIoMode::Input)
        {
            if (!tensor.profile || tensor.profile->minimum.empty() ||
                tensor.profile->maximum.empty() || tensor.profile->minimum[0] > 1 ||
                tensor.profile->maximum[0] < static_cast<std::int64_t>(required_batch))
            {
                throw tensorrt::TensorRtError(
                    tensorrt::TensorRtErrorCode::EngineContractMismatch,
                    std::string(model) + " engine batch profile cannot cover [1," +
                        std::to_string(required_batch) + "]");
            }
            return;
        }
    }
}

tensorrt::TensorView device_input(const image::TensorView& input, const char* name,
                                  std::int32_t extent)
{
    if (input.data == nullptr || input.memory_kind != image::MemoryKind::CudaDevice ||
        input.element_type != image::TensorElementType::Float32 ||
        input.layout != image::TensorLayout::Nchw || input.batch != 1 ||
        input.channels != 3 || input.height != extent || input.width != extent)
    {
        throw tensorrt::TensorRtError(
            tensorrt::TensorRtErrorCode::InvalidTensorView,
            std::string(name) + " expects a CUDA FP32 NCHW [1,3," +
                std::to_string(extent) + "," + std::to_string(extent) + "] tensor");
    }
    return { name, DataType::Float32, { 1, 3, extent, extent }, input.data,
             input.byte_size, tensorrt::MemoryKind::CudaDevice };
}

} // namespace

PalmTensorRtModel::PalmTensorRtModel(const std::filesystem::path& path,
                                     const TensorRtModelOptions& options)
    : engine_(tensorrt::Engine::load(path, engine_options(options)))
    , max_output_bytes_(options.max_palm_candidates * kPalmRowWidth * sizeof(float))
{
    validate_contract(engine_,
        { { "input", TensorIoMode::Input, DataType::Float32,
            { 1, 3, kPalmInputExtent, kPalmInputExtent } },
          { kPalmOutputName, TensorIoMode::Output, DataType::Float32, { -1, 8 } } },
        "Palm");
    executor_ = engine_->create_executor();
}

std::vector<float> PalmTensorRtModel::run(const image::TensorView& input)
{
    const auto outputs = executor_->run_dynamic(
        { device_input(input, "input", kPalmInputExtent) },
        { { kPalmOutputName, DataType::Float32, max_output_bytes_ } });
    if (outputs.size() != 1U || outputs[0].shape.size() != 2U ||
        outputs[0].shape[0] < 0 || outputs[0].shape[1] != 8)
    {
        throw tensorrt::TensorRtError(
            tensorrt::TensorRtErrorCode::EngineContractMismatch,
            "Palm runtime output shape is invalid");
    }
    return outputs[0].float32_values();
}

HandLandmarkTensorRtModel::HandLandmarkTensorRtModel(
    const std::filesystem::path& path, const TensorRtModelOptions& options)
    : engine_(tensorrt::Engine::load(path, engine_options(options)))
{
    validate_contract(engine_,
        { { "input", TensorIoMode::Input, DataType::Float32,
            { -1, 3, kHandLandmarkInputExtent, kHandLandmarkInputExtent } },
          { "xyz_x21", TensorIoMode::Output, DataType::Float32, { -1, 63 } },
          { "hand_score", TensorIoMode::Output, DataType::Float32, { -1, 1 } },
          { "lefthand_0_or_righthand_1", TensorIoMode::Output, DataType::Float32,
            { -1, 1 } } },
        "hand landmark");
    validate_dynamic_batch(engine_, "input", 1U, "hand landmark");
    executor_ = engine_->create_executor();
}

HandTensorRtOutputs HandLandmarkTensorRtModel::run(const image::TensorView& input)
{
    HandTensorRtOutputs result;
    result.landmarks.resize(kHandLandmarkCount * 3U);
    executor_->run(
        { device_input(input, "input", kHandLandmarkInputExtent) },
        { { "xyz_x21", DataType::Float32, { 1, 63 }, result.landmarks.data(),
            result.landmarks.size() * sizeof(float), tensorrt::MemoryKind::Host },
          { "hand_score", DataType::Float32, { 1, 1 }, &result.score,
            sizeof(result.score), tensorrt::MemoryKind::Host },
          { "lefthand_0_or_righthand_1", DataType::Float32, { 1, 1 },
            &result.handedness, sizeof(result.handedness), tensorrt::MemoryKind::Host } });
    return result;
}

KeypointClassifierTensorRtModel::KeypointClassifierTensorRtModel(
    const std::filesystem::path& path, const TensorRtModelOptions& options)
    : engine_(tensorrt::Engine::load(path, engine_options(options)))
{
    validate_contract(engine_,
        { { "input", TensorIoMode::Input, DataType::Float32, { -1, 42 } },
          { "class_ids", TensorIoMode::Output, DataType::Int64, { -1 } } },
        "keypoint classifier");
    validate_dynamic_batch(engine_, "input", options.max_hands, "keypoint classifier");
    executor_ = engine_->create_executor();
}

std::vector<std::int64_t> KeypointClassifierTensorRtModel::run(
    const std::vector<float>& input, std::size_t batch)
{
    if (batch == 0U || input.size() != batch * kHandLandmarkCount * 2U)
    {
        throw tensorrt::TensorRtError(tensorrt::TensorRtErrorCode::InvalidTensorView,
                                     "keypoint classifier input batch is invalid");
    }
    std::vector<std::int64_t> output(batch);
    executor_->run(
        { { "input", DataType::Float32,
            { static_cast<std::int64_t>(batch), 42 }, input.data(),
            input.size() * sizeof(float), tensorrt::MemoryKind::Host } },
        { { "class_ids", DataType::Int64, { static_cast<std::int64_t>(batch) },
            output.data(), output.size() * sizeof(std::int64_t),
            tensorrt::MemoryKind::Host } });
    return output;
}

FaceLandmarkTensorRtModel::FaceLandmarkTensorRtModel(
    const std::filesystem::path& path, const TensorRtModelOptions& options)
    : engine_(tensorrt::Engine::load(path, engine_options(options)))
{
    validate_contract(engine_,
        { { "image", TensorIoMode::Input, DataType::Float32,
            { 1, 3, kFaceLandmarkInputExtent, kFaceLandmarkInputExtent } },
          { "scores", TensorIoMode::Output, DataType::Float32, { 1 } },
          { "landmarks", TensorIoMode::Output, DataType::Float32,
            { 1, static_cast<std::int64_t>(kFaceLandmarkCount), 3 } } },
        "MediaPipe face landmark");
    executor_ = engine_->create_executor();
}

FaceTensorRtOutputs FaceLandmarkTensorRtModel::run(const image::TensorView& input)
{
    FaceTensorRtOutputs result;
    result.landmarks.resize(kFaceLandmarkCount * 3U);
    executor_->run(
        { device_input(input, "image", kFaceLandmarkInputExtent) },
        { { "scores", DataType::Float32, { 1 }, &result.score, sizeof(result.score),
            tensorrt::MemoryKind::Host },
          { "landmarks", DataType::Float32,
            { 1, static_cast<std::int64_t>(kFaceLandmarkCount), 3 },
            result.landmarks.data(), result.landmarks.size() * sizeof(float),
            tensorrt::MemoryKind::Host } });
    return result;
}

} // namespace kfcore::vision_models::detail
