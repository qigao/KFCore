#include "kfcore/yolo/tensorrt.hpp"

#include "engine_file.hpp"
#include "tensorrt_raii.hpp"

#include <NvInferPlugin.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::yolo
{
namespace
{

    constexpr std::uintmax_t kMaxSerializedEngineBytes =
        UINTMAX_C(1024) * UINTMAX_C(1024) * UINTMAX_C(1024);
    constexpr std::size_t  kTensorNameStorageLimit = 4096;
    constexpr std::int32_t kExpectedTensorCount    = 5;
    constexpr float        kMinimumBorderValue     = 0.0f;
    constexpr float        kMaximumBorderValue     = 255.0f;

    [[noreturn]] void throw_invalid(std::string message)
    {
        throw YoloError(YoloErrorCode::InvalidArgument, std::move(message));
    }

    [[noreturn]] void throw_file(std::string message)
    {
        throw YoloError(YoloErrorCode::FileIo, std::move(message));
    }

    [[noreturn]] void throw_deserialize(std::string message)
    {
        throw YoloError(YoloErrorCode::EngineDeserialize, std::move(message));
    }

    [[noreturn]] void throw_contract(std::string message)
    {
        throw YoloError(YoloErrorCode::EngineContractMismatch, std::move(message));
    }

    [[noreturn]] void throw_tensorrt(std::string message)
    {
        throw YoloError(YoloErrorCode::TensorRtFailure, std::move(message));
    }

    [[noreturn]] void throw_resource(std::string message)
    {
        throw YoloError(YoloErrorCode::ResourceLimitExceeded, std::move(message));
    }

    void validate_engine_options(const EngineOptions& options)
    {
        if (options.device_id < 0)
        {
            throw_invalid("engine options stage: device_id must not be negative");
        }
        if (options.max_batch == 0 || options.max_detections == 0 || options.max_input_bytes == 0 ||
            options.max_output_bytes == 0)
        {
            throw_invalid("engine options stage: all resource limits must be positive");
        }

        const std::array<const std::string*, 5> names = { {
            &options.tensor_names.images,
            &options.tensor_names.num_dets,
            &options.tensor_names.boxes,
            &options.tensor_names.scores,
            &options.tensor_names.labels,
        } };
        for (std::size_t index = 0; index < names.size(); ++index)
        {
            if (names[index]->empty())
            {
                throw_invalid("engine options stage: tensor names must not be empty");
            }
            if (names[index]->size() >= kTensorNameStorageLimit)
            {
                throw_invalid("engine options stage: tensor name exceeds TensorRT limit");
            }
            for (std::size_t other = index + 1; other < names.size(); ++other)
            {
                if (*names[index] == *names[other])
                {
                    throw_invalid("engine options stage: tensor names must be unique");
                }
            }
        }
    }

    void validate_detector_options(const DetectorOptions&   options,
                                   const ValidatedContract& contract)
    {
        if (options.input_size.has_value())
        {
            const std::int32_t height = (*options.input_size)[0];
            const std::int32_t width  = (*options.input_size)[1];
            if (height <= 0 || width <= 0)
            {
                throw_invalid("detector options stage: input_size dimensions must be positive");
            }
            if (static_cast<std::int64_t>(height) != contract.input_height ||
                static_cast<std::int64_t>(width) != contract.input_width)
            {
                throw_invalid(
                    "detector options stage: input_size must match the engine input dimensions");
            }
        }
        for (float value : options.mean)
        {
            if (!std::isfinite(value))
            {
                throw_invalid("detector options stage: mean values must be finite");
            }
        }
        for (float value : options.stddev)
        {
            if (!std::isfinite(value) || value <= 0.0f)
            {
                throw_invalid("detector options stage: stddev values must be finite and positive");
            }
        }
        if (!std::isfinite(options.border_value) || options.border_value < kMinimumBorderValue ||
            options.border_value > kMaximumBorderValue)
        {
            throw_invalid("detector options stage: border_value must be within [0, 255]");
        }
    }

    std::vector<std::byte> read_engine_file(const std::filesystem::path& path)
    {
        if (path.empty())
        {
            throw_file("engine file open stage: path must not be empty");
        }

        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream.is_open())
        {
            throw_file("engine file open stage: cannot open " + path.string());
        }
        const std::ifstream::pos_type end_position = stream.tellg();
        if (end_position <= std::ifstream::pos_type { 0 })
        {
            throw_deserialize("engine file read stage: serialized engine is empty");
        }
        const auto engine_bytes = static_cast<std::uintmax_t>(end_position);
        if (engine_bytes > kMaxSerializedEngineBytes ||
            engine_bytes > static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)()) ||
            engine_bytes >
                static_cast<std::uintmax_t>((std::numeric_limits<std::streamsize>::max)()))
        {
            throw_resource(
                "engine file read stage: serialized engine exceeds the 1 GiB hard limit");
        }

        return detail::read_engine_stream_exact(stream, static_cast<std::size_t>(engine_bytes));
    }

    TensorDataType tensor_data_type(nvinfer1::DataType type, const char* tensor_name)
    {
        switch (type)
        {
        case nvinfer1::DataType::kFLOAT:
            return TensorDataType::Float32;
        case nvinfer1::DataType::kHALF:
            return TensorDataType::Float16;
        case nvinfer1::DataType::kINT32:
            return TensorDataType::Int32;
        default:
            throw_contract(
                std::string("metadata extraction stage: unsupported data type for tensor ") +
                tensor_name);
        }
    }

    TensorIoMode tensor_io_mode(nvinfer1::TensorIOMode mode, const char* tensor_name)
    {
        switch (mode)
        {
        case nvinfer1::TensorIOMode::kINPUT:
            return TensorIoMode::Input;
        case nvinfer1::TensorIOMode::kOUTPUT:
            return TensorIoMode::Output;
        default:
            throw_contract(std::string("metadata extraction stage: invalid I/O mode for tensor ") +
                           tensor_name);
        }
    }

    std::vector<std::int64_t> dimensions(const nvinfer1::Dims& dims, const char* tensor_name,
                                         const char* selector)
    {
        constexpr std::int32_t storage_dimensions =
            static_cast<std::int32_t>(sizeof(dims.d) / sizeof(dims.d[0]));
        if (dims.nbDims < 0 || dims.nbDims > storage_dimensions)
        {
            throw_contract(std::string("metadata extraction stage: invalid ") + selector +
                           " dimensions for tensor " + tensor_name);
        }
        std::vector<std::int64_t> result;
        result.reserve(static_cast<std::size_t>(dims.nbDims));
        for (std::int32_t index = 0; index < dims.nbDims; ++index)
        {
            result.push_back(dims.d[index]);
        }
        return result;
    }

    struct BatchProfile
    {
        std::int64_t minimum;
        std::int64_t optimum;
        std::int64_t maximum;
    };

    BatchProfile input_batch_profile(const nvinfer1::ICudaEngine& engine,
                                     const std::string&           input_name)
    {
        const nvinfer1::Dims minimum =
            engine.getProfileShape(input_name.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);
        const nvinfer1::Dims optimum =
            engine.getProfileShape(input_name.c_str(), 0, nvinfer1::OptProfileSelector::kOPT);
        const nvinfer1::Dims maximum =
            engine.getProfileShape(input_name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
        if (minimum.nbDims < 1 || optimum.nbDims != minimum.nbDims ||
            maximum.nbDims != minimum.nbDims)
        {
            throw_contract("metadata extraction stage: invalid images optimization profile");
        }
        return { minimum.d[0], optimum.d[0], maximum.d[0] };
    }

    std::vector<std::int64_t> output_profile_shape(const nvinfer1::Dims& shape,
                                                   const char* tensor_name, std::int64_t batch)
    {
        std::vector<std::int64_t> result = dimensions(shape, tensor_name, "network");
        if (!result.empty() && result[0] == -1)
        {
            result[0] = batch;
        }
        return result;
    }

    EngineMetadata extract_metadata(const nvinfer1::ICudaEngine& engine,
                                    const EngineOptions&         options)
    {
        if (engine.getNbOptimizationProfiles() != 1)
        {
            throw_contract(
                "metadata extraction stage: exactly one optimization profile is required");
        }
        if (engine.getTensorIOMode(options.tensor_names.images.c_str()) !=
            nvinfer1::TensorIOMode::kINPUT)
        {
            throw_contract("metadata extraction stage: configured images tensor is not an input");
        }
        const BatchProfile batches = input_batch_profile(engine, options.tensor_names.images);

        const std::int32_t tensor_count = engine.getNbIOTensors();
        if (tensor_count != kExpectedTensorCount)
        {
            throw_contract(
                "metadata extraction stage: engine must expose exactly five named I/O tensors");
        }

        EngineMetadata metadata;
        metadata.tensors.reserve(static_cast<std::size_t>(tensor_count));
        for (std::int32_t index = 0; index < tensor_count; ++index)
        {
            const char* name = engine.getIOTensorName(index);
            if (name == nullptr || *name == '\0')
            {
                throw_contract("metadata extraction stage: engine returned an empty tensor name");
            }
            if (engine.getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE)
            {
                throw_contract(
                    std::string("metadata extraction stage: tensor must use device memory: ") +
                    name);
            }

            const nvinfer1::TensorIOMode trt_mode = engine.getTensorIOMode(name);
            TensorDesc                   descriptor;
            descriptor.name      = name;
            descriptor.mode      = tensor_io_mode(trt_mode, name);
            descriptor.data_type = tensor_data_type(engine.getTensorDataType(name), name);
            if (trt_mode == nvinfer1::TensorIOMode::kINPUT)
            {
                descriptor.min_shape =
                    dimensions(engine.getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMIN),
                               name, "minimum profile");
                descriptor.opt_shape =
                    dimensions(engine.getProfileShape(name, 0, nvinfer1::OptProfileSelector::kOPT),
                               name, "optimum profile");
                descriptor.max_shape =
                    dimensions(engine.getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX),
                               name, "maximum profile");
            }
            else
            {
                const nvinfer1::Dims shape = engine.getTensorShape(name);
                descriptor.min_shape       = output_profile_shape(shape, name, batches.minimum);
                descriptor.opt_shape       = output_profile_shape(shape, name, batches.optimum);
                descriptor.max_shape       = output_profile_shape(shape, name, batches.maximum);
            }
            metadata.tensors.push_back(std::move(descriptor));
        }
        return metadata;
    }

    ContractNames contract_names(const TensorNames& names)
    {
        return { names.images, names.num_dets, names.boxes, names.scores, names.labels };
    }

    ContractLimits contract_limits(const EngineOptions& options)
    {
        return {
            options.max_batch,
            options.max_detections,
            options.max_input_bytes,
            options.max_output_bytes,
        };
    }

} // namespace

Engine::Engine(std::shared_ptr<const State> state)
    : state_(std::move(state))
{
}

std::shared_ptr<const Engine> Engine::load(const std::filesystem::path& engine_path,
                                           const EngineOptions&         options)
{
    validate_engine_options(options);
    std::vector<std::byte> bytes = read_engine_file(engine_path);

    try
    {
        auto state     = std::make_shared<State>();
        state->options = options;
        detail::check_cuda(cudaSetDevice(options.device_id), "cudaSetDevice",
                           "engine deserialization");
        if (!initLibNvInferPlugins(&state->logger, ""))
        {
            throw_tensorrt("plugin initialization stage: initLibNvInferPlugins failed");
        }
        state->runtime.reset(nvinfer1::createInferRuntime(state->logger));
        if (!state->runtime)
        {
            throw_tensorrt("runtime creation stage: createInferRuntime returned null");
        }
        state->engine.reset(state->runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!state->engine)
        {
            throw_deserialize("engine deserialization stage: TensorRT rejected the engine");
        }

        EngineMetadata metadata = extract_metadata(*state->engine, options);
        state->contract = validate_engine_contract(metadata, contract_names(options.tensor_names),
                                                   contract_limits(options));
        return std::shared_ptr<const Engine>(new Engine(std::move(state)));
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("engine construction stage: allocation failed");
    }
    catch (const std::length_error&)
    {
        throw_resource("engine construction stage: container capacity exceeded");
    }
}

std::unique_ptr<TensorRtDetector> Engine::create_detector(const DetectorOptions& options) const
{
    if (!state_ || !state_->engine)
    {
        throw_invalid("detector creation stage: engine state is unavailable");
    }
    validate_detector_options(options, state_->contract);
    detail::check_cuda(cudaSetDevice(state_->options.device_id), "cudaSetDevice",
                       "detector creation");

    detail::TensorRtOwner<nvinfer1::IExecutionContext> context(
        state_->engine->createExecutionContext());
    if (!context)
    {
        throw_tensorrt("execution context creation stage: createExecutionContext returned null");
    }

    try
    {
        auto impl = std::make_unique<TensorRtDetector::Impl>(state_, std::move(context), options);
        const ValidatedContract& contract = state_->contract;
        impl->input_device.reserve(contract.images.max_bytes, state_->options.max_input_bytes);
        impl->input_host.reserve(contract.images.max_bytes, state_->options.max_input_bytes);

        impl->num_dets_device.reserve(contract.num_dets.max_bytes,
                                      state_->options.max_output_bytes);
        impl->boxes_device.reserve(contract.boxes.max_bytes, state_->options.max_output_bytes);
        impl->scores_device.reserve(contract.scores.max_bytes, state_->options.max_output_bytes);
        impl->labels_device.reserve(contract.labels.max_bytes, state_->options.max_output_bytes);
        impl->num_dets_host.reserve(contract.num_dets.max_bytes, state_->options.max_output_bytes);
        impl->boxes_host.reserve(contract.boxes.max_bytes, state_->options.max_output_bytes);
        impl->scores_host.reserve(contract.scores.max_bytes, state_->options.max_output_bytes);
        impl->labels_host.reserve(contract.labels.max_bytes, state_->options.max_output_bytes);
        return std::unique_ptr<TensorRtDetector>(new TensorRtDetector(std::move(impl)));
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("detector creation stage: allocation failed");
    }
    catch (const std::length_error&)
    {
        throw_resource("detector creation stage: container capacity exceeded");
    }
}

TensorRtDetector::TensorRtDetector(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

TensorRtDetector::~TensorRtDetector() = default;

} // namespace kfcore::yolo
