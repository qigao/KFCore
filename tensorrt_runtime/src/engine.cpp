#include "engine.hpp"

#include "engine_file.hpp"
#include "cuda_device.hpp"
#include "tensor_validation.hpp"
#include "tensorrt_version.hpp"

#include "kfcore/tensorrt/error.hpp"

#include <NvInferPlugin.h>
#include <NvInferVersion.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::tensorrt
{
namespace
{

    constexpr std::size_t kTensorNameStorageLimit = 4096;

    [[noreturn]] void throw_invalid(std::string message)
    {
        throw TensorRtError(TensorRtErrorCode::InvalidArgument, std::move(message));
    }

    [[noreturn]] void throw_deserialize(std::string message)
    {
        throw TensorRtError(TensorRtErrorCode::EngineDeserialize, std::move(message));
    }

    [[noreturn]] void throw_contract(std::string message)
    {
        throw TensorRtError(TensorRtErrorCode::EngineContractMismatch, std::move(message));
    }

    [[noreturn]] void throw_tensorrt(std::string message)
    {
        throw TensorRtError(TensorRtErrorCode::TensorRtFailure, std::move(message));
    }

    [[noreturn]] void throw_resource(std::string message)
    {
        throw TensorRtError(TensorRtErrorCode::ResourceLimitExceeded, std::move(message));
    }

    void validate_engine_options(const EngineOptions& options)
    {
        if (options.device_id < 0)
        {
            throw_invalid("engine options stage: device_id must not be negative");
        }
        if (options.max_serialized_engine_bytes == 0 || options.max_tensor_count == 0 ||
            options.max_input_bytes == 0 || options.max_output_bytes == 0)
        {
            throw_resource("engine options stage: all resource limits must be positive");
        }
    }

    void initialize_plugins_once()
    {
        static detail::TensorRtLogger plugin_logger;
        static std::once_flag         plugin_once;
        std::call_once(plugin_once,
                       []
                       {
                           if (!initLibNvInferPlugins(&plugin_logger, ""))
                           {
                               // A throwing call_once invocation is not committed, so a later
                               // Engine::load may retry initialization.
                               throw_tensorrt(
                                   "plugin initialization stage: initLibNvInferPlugins failed");
                           }
                       });
    }

    DataType data_type(nvinfer1::DataType type, const char* tensor_name)
    {
        switch (type)
        {
        case nvinfer1::DataType::kFLOAT:
            return DataType::Float32;
        case nvinfer1::DataType::kHALF:
            return DataType::Float16;
        case nvinfer1::DataType::kINT8:
            return DataType::Int8;
        case nvinfer1::DataType::kINT32:
            return DataType::Int32;
        case nvinfer1::DataType::kBOOL:
            return DataType::Bool;
        case nvinfer1::DataType::kUINT8:
            return DataType::UInt8;
        case nvinfer1::DataType::kBF16:
            return DataType::BFloat16;
        case nvinfer1::DataType::kINT64:
            return DataType::Int64;
        default:
            throw_contract(std::string("metadata extraction stage: unsupported data type for tensor ") +
                           tensor_name);
        }
    }

    TensorIoMode io_mode(nvinfer1::TensorIOMode mode, const char* tensor_name)
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

    TensorShape shape_from_dims(const nvinfer1::Dims& dims, const char* tensor_name,
                                const char* selector)
    {
        if (dims.nbDims < 0 || dims.nbDims > nvinfer1::Dims::MAX_DIMS)
        {
            throw_contract(std::string("metadata extraction stage: invalid ") + selector +
                           " dimensions for tensor " + tensor_name);
        }
        TensorShape shape;
        shape.reserve(static_cast<std::size_t>(dims.nbDims));
        for (std::int32_t index = 0; index < dims.nbDims; ++index)
        {
            shape.push_back(dims.d[index]);
        }
        return shape;
    }

    void validate_physical_layout(const nvinfer1::ICudaEngine& engine, const char* tensor_name,
                                  DataType type)
    {
        const nvinfer1::TensorFormat format = engine.getTensorFormat(tensor_name, 0);
        const std::int32_t vectorized_dimension =
            engine.getTensorVectorizedDim(tensor_name, 0);
        const std::int32_t components = engine.getTensorComponentsPerElement(tensor_name, 0);
        const std::int32_t component_bytes = engine.getTensorBytesPerComponent(tensor_name, 0);
        const std::size_t  expected_bytes = detail::scalar_byte_size(type);

        if (format != nvinfer1::TensorFormat::kLINEAR || vectorized_dimension != -1 ||
            (components != -1 && components != 1) ||
            (component_bytes != -1 &&
             component_bytes != static_cast<std::int32_t>(expected_bytes)))
        {
            throw_contract(std::string("metadata extraction stage: tensor must use an unpacked ") +
                           "linear scalar device format: " + tensor_name);
        }
    }

    std::vector<TensorDescriptor> extract_metadata(nvinfer1::ICudaEngine& engine,
                                                   const EngineOptions& options)
    {
        if (engine.getNbOptimizationProfiles() < 1)
        {
            throw_contract("metadata extraction stage: optimization profile 0 is unavailable");
        }

        const std::int32_t tensor_count = engine.getNbIOTensors();
        if (tensor_count <= 0)
        {
            throw_contract("metadata extraction stage: engine has no I/O tensors");
        }
        if (static_cast<std::uintmax_t>(tensor_count) >
            static_cast<std::uintmax_t>(options.max_tensor_count))
        {
            throw_resource("metadata extraction stage: tensor count exceeds configured limit");
        }

        std::vector<TensorDescriptor> tensors;
        tensors.reserve(static_cast<std::size_t>(tensor_count));
        for (std::int32_t index = 0; index < tensor_count; ++index)
        {
            const char* name = engine.getIOTensorName(index);
            if (name == nullptr || *name == '\0')
            {
                throw_contract("metadata extraction stage: engine returned an empty tensor name");
            }
            const std::string tensor_name(name);
            if (tensor_name.size() >= kTensorNameStorageLimit)
            {
                throw_contract("metadata extraction stage: tensor name exceeds TensorRT limit");
            }
            if (engine.getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE)
            {
                throw_contract("metadata extraction stage: tensor must use device memory: " +
                               tensor_name);
            }
            if (engine.isShapeInferenceIO(name))
            {
                throw_contract("metadata extraction stage: shape-inference I/O is unsupported: " +
                               tensor_name);
            }

            TensorDescriptor descriptor;
            descriptor.name      = tensor_name;
            descriptor.mode      = io_mode(engine.getTensorIOMode(name), name);
            descriptor.data_type = data_type(engine.getTensorDataType(name), name);
            descriptor.declared_shape =
                shape_from_dims(engine.getTensorShape(name), name, "network declaration");
            validate_physical_layout(engine, name, descriptor.data_type);

#if NV_TENSORRT_MAJOR > 10 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 11)
            static_assert(detail::tensorrt_supports_alias_query(NV_TENSORRT_MAJOR,
                                                                 NV_TENSORRT_MINOR),
                          "TensorRT alias-query version gate is inconsistent");
            if (descriptor.mode == TensorIoMode::Output &&
                engine.getAliasedInputTensor(name) != nullptr)
            {
                throw_contract("metadata extraction stage: aliased I/O is unsupported: " +
                               tensor_name);
            }
#else
            static_assert(!detail::tensorrt_supports_alias_query(NV_TENSORRT_MAJOR,
                                                                  NV_TENSORRT_MINOR),
                          "TensorRT alias-query version gate is inconsistent");
            // TensorRT before 10.11 does not expose engine I/O alias metadata.
#endif

            if (descriptor.mode == TensorIoMode::Input)
            {
                TensorProfile profile;
                profile.minimum = shape_from_dims(
                    engine.getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMIN), name,
                    "minimum profile");
                profile.optimum = shape_from_dims(
                    engine.getProfileShape(name, 0, nvinfer1::OptProfileSelector::kOPT), name,
                    "optimum profile");
                profile.maximum = shape_from_dims(
                    engine.getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX), name,
                    "maximum profile");
                descriptor.profile = std::move(profile);
            }
            tensors.push_back(std::move(descriptor));
        }

        detail::validate_tensor_metadata(tensors, options);
        return tensors;
    }

} // namespace

Engine::Engine(std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl))
{
}

Engine::~Engine()
{
    if (!impl_)
    {
        return;
    }
    const int device_id = impl_->options.device_id;
    try
    {
        detail::CudaDeviceScope device_scope(device_id);
        impl_.reset();
        device_scope.restore();
    }
    catch (...)
    {
        impl_.reset();
    }
}

std::shared_ptr<const Engine> Engine::load(const std::filesystem::path& engine_path,
                                           const EngineOptions&         options)
{
    validate_engine_options(options);
    std::vector<std::byte> bytes =
        detail::read_engine_file_bounded(engine_path, options.max_serialized_engine_bytes);

    try
    {
        auto impl     = std::make_shared<Impl>();
        impl->options = options;
        detail::CudaDeviceScope device_scope(options.device_id);
        initialize_plugins_once();
        impl->runtime.reset(nvinfer1::createInferRuntime(impl->logger));
        if (!impl->runtime)
        {
            throw_tensorrt("runtime creation stage: createInferRuntime returned null");
        }
        impl->engine.reset(impl->runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!impl->engine)
        {
            throw_deserialize("engine deserialization stage: TensorRT rejected the engine");
        }
        impl->tensors = extract_metadata(*impl->engine, options);
        std::shared_ptr<const Engine> result(new Engine(std::move(impl)));
        device_scope.restore();
        return result;
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

const std::vector<TensorDescriptor>& Engine::tensors() const noexcept
{
    return impl_->tensors;
}

} // namespace kfcore::tensorrt
