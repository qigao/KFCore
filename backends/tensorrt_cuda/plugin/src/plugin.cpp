#include "kfcore/runtime/abi/backend_v1.h"
#include "kfcore/tensorrt/error.hpp"
#include "kfcore/tensorrt/runtime.hpp"
#include "kfcore/tensorrt/types.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

thread_local std::string g_last_error;

struct DeviceInfo
{
    std::string id;
    std::string name;
    int ordinal = 0;
};

std::string_view view(kf_string_view_v1 value)
{
    if (value.size == 0U)
    {
        return {};
    }
    if (value.data == nullptr ||
        value.size > static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)()))
    {
        throw std::invalid_argument("invalid ABI string view");
    }
    return std::string_view(value.data, static_cast<std::size_t>(value.size));
}

kf_string_view_v1 abi_view(std::string_view value) noexcept
{
    return { value.data(), static_cast<std::uint64_t>(value.size()) };
}

std::filesystem::path abi_path(kf_string_view_v1 value)
{
    const std::string_view text = view(value);
    if (text.empty())
    {
        throw std::invalid_argument("artifact path is empty");
    }
    return std::filesystem::u8path(text.begin(), text.end());
}

kf_data_type_v1 abi_type(kfcore::tensorrt::DataType type)
{
    using kfcore::tensorrt::DataType;
    switch (type)
    {
    case DataType::Float32: return KF_DATA_TYPE_V1_FLOAT32;
    case DataType::Float16: return KF_DATA_TYPE_V1_FLOAT16;
    case DataType::Int8: return KF_DATA_TYPE_V1_INT8;
    case DataType::Int32: return KF_DATA_TYPE_V1_INT32;
    case DataType::Int64: return KF_DATA_TYPE_V1_INT64;
    case DataType::UInt8: return KF_DATA_TYPE_V1_UINT8;
    case DataType::Bool: return KF_DATA_TYPE_V1_BOOL;
    case DataType::BFloat16: return KF_DATA_TYPE_V1_BFLOAT16;
    }
    throw std::invalid_argument("unsupported TensorRT data type");
}

kfcore::tensorrt::DataType runtime_type(kf_data_type_v1 type)
{
    using kfcore::tensorrt::DataType;
    switch (type)
    {
    case KF_DATA_TYPE_V1_FLOAT32: return DataType::Float32;
    case KF_DATA_TYPE_V1_FLOAT16: return DataType::Float16;
    case KF_DATA_TYPE_V1_INT8: return DataType::Int8;
    case KF_DATA_TYPE_V1_INT32: return DataType::Int32;
    case KF_DATA_TYPE_V1_INT64: return DataType::Int64;
    case KF_DATA_TYPE_V1_UINT8: return DataType::UInt8;
    case KF_DATA_TYPE_V1_BOOL: return DataType::Bool;
    case KF_DATA_TYPE_V1_BFLOAT16: return DataType::BFloat16;
    default: throw std::invalid_argument("unsupported ABI data type");
    }
}

kfcore::tensorrt::MemoryKind runtime_memory(kf_memory_kind_v1 kind)
{
    using kfcore::tensorrt::MemoryKind;
    switch (kind)
    {
    case KF_MEMORY_KIND_V1_HOST: return MemoryKind::Host;
    case KF_MEMORY_KIND_V1_DEVICE: return MemoryKind::CudaDevice;
    case KF_MEMORY_KIND_V1_PINNED_HOST:
        throw std::invalid_argument("pinned-host tensor I/O is not advertised by TensorRT ABI v1");
    default: throw std::invalid_argument("unsupported ABI memory kind");
    }
}

kf_status_v1 map_exception() noexcept
{
    try
    {
        throw;
    }
    catch (const kfcore::tensorrt::TensorRtError& error)
    {
        g_last_error = error.what();
        using kfcore::tensorrt::TensorRtErrorCode;
        switch (error.code())
        {
        case TensorRtErrorCode::InvalidArgument:
        case TensorRtErrorCode::InvalidTensorView:
            return KF_STATUS_V1_INVALID_ARGUMENT;
        case TensorRtErrorCode::FileIo:
        case TensorRtErrorCode::EngineDeserialize:
        case TensorRtErrorCode::EngineContractMismatch:
            return KF_STATUS_V1_INCOMPATIBLE_ARTIFACT;
        case TensorRtErrorCode::ResourceLimitExceeded:
            return KF_STATUS_V1_OUT_OF_MEMORY;
        case TensorRtErrorCode::CudaFailure:
            return KF_STATUS_V1_DEVICE_UNAVAILABLE;
        case TensorRtErrorCode::ConcurrentExecution:
        case TensorRtErrorCode::TensorRtFailure:
            return KF_STATUS_V1_RUNTIME_FAILURE;
        }
        return KF_STATUS_V1_RUNTIME_FAILURE;
    }
    catch (const std::bad_alloc& error)
    {
        g_last_error = error.what();
        return KF_STATUS_V1_OUT_OF_MEMORY;
    }
    catch (const std::invalid_argument& error)
    {
        g_last_error = error.what();
        return KF_STATUS_V1_INVALID_ARGUMENT;
    }
    catch (const std::exception& error)
    {
        g_last_error = error.what();
        return KF_STATUS_V1_RUNTIME_FAILURE;
    }
    catch (...)
    {
        g_last_error = "unknown TensorRT backend failure";
        return KF_STATUS_V1_INTERNAL_FAILURE;
    }
}

const DeviceInfo* find_device(const std::vector<DeviceInfo>& devices, std::string_view id)
{
    const auto iterator = std::find_if(devices.begin(), devices.end(),
                                       [id](const DeviceInfo& device) { return device.id == id; });
    return iterator == devices.end() ? nullptr : &*iterator;
}

std::vector<DeviceInfo> enumerate_cuda_devices()
{
    int count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&count);
    if (count_status != cudaSuccess)
    {
        throw std::runtime_error(std::string("cudaGetDeviceCount failed: ") +
                                 cudaGetErrorString(count_status));
    }

    std::vector<DeviceInfo> result;
    result.reserve(static_cast<std::size_t>((std::max)(count, 0)));
    for (int ordinal = 0; ordinal < count; ++ordinal)
    {
        cudaDeviceProp properties {};
        const cudaError_t status = cudaGetDeviceProperties(&properties, ordinal);
        if (status != cudaSuccess)
        {
            throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") +
                                     cudaGetErrorString(status));
        }
        DeviceInfo device;
        device.id = "cuda:" + std::to_string(ordinal);
        device.name = properties.name;
        device.ordinal = ordinal;
        result.push_back(std::move(device));
    }
    return result;
}

} // namespace

struct kf_backend_handle_v1_t
{
    std::vector<DeviceInfo> devices;
};

struct kf_model_handle_v1_t
{
    kf_backend_handle_v1 backend = nullptr;
    std::shared_ptr<const kfcore::tensorrt::Engine> engine;
    std::string device_id;
};

struct kf_context_handle_v1_t
{
    kf_model_handle_v1 model = nullptr;
    std::unique_ptr<kfcore::tensorrt::Executor> executor;
};

namespace
{

kf_status_v1 create_backend(const kf_backend_create_info_v1* info,
                            kf_backend_handle_v1* out_backend) noexcept
{
    try
    {
        if (info == nullptr || info->struct_size < sizeof(kf_backend_create_info_v1) ||
            out_backend == nullptr)
        {
            return KF_STATUS_V1_INVALID_ARGUMENT;
        }
        auto backend = std::make_unique<kf_backend_handle_v1_t>();
        backend->devices = enumerate_cuda_devices();
        *out_backend = backend.release();
        g_last_error.clear();
        return KF_STATUS_V1_OK;
    }
    catch (...)
    {
        return map_exception();
    }
}

void destroy_backend(kf_backend_handle_v1 backend) noexcept
{
    delete backend;
}

kf_status_v1 get_backend_info(kf_backend_handle_v1 backend,
                              kf_backend_info_v1* out_info) noexcept
{
    if (backend == nullptr || out_info == nullptr ||
        out_info->struct_size < sizeof(kf_backend_info_v1))
    {
        return KF_STATUS_V1_INVALID_ARGUMENT;
    }
    static constexpr std::string_view kId = "tensorrt";
    static constexpr std::string_view kName = "KFCore TensorRT";
    *out_info = {
        sizeof(kf_backend_info_v1), abi_view(kId), abi_view(kName),
        1U, 0U, 0U,
        KF_BACKEND_CAP_V1_HOST_MEMORY | KF_BACKEND_CAP_V1_DEVICE_MEMORY,
    };
    return KF_STATUS_V1_OK;
}

kf_status_v1 get_device_count(kf_backend_handle_v1 backend, std::uint64_t* out_count) noexcept
{
    if (backend == nullptr || out_count == nullptr)
    {
        return KF_STATUS_V1_INVALID_ARGUMENT;
    }
    *out_count = static_cast<std::uint64_t>(backend->devices.size());
    return KF_STATUS_V1_OK;
}

kf_status_v1 get_device_info(kf_backend_handle_v1 backend,
                             std::uint64_t index,
                             kf_device_info_v1* out_info) noexcept
{
    if (backend == nullptr || out_info == nullptr ||
        out_info->struct_size < sizeof(kf_device_info_v1) ||
        index >= backend->devices.size())
    {
        return KF_STATUS_V1_INVALID_ARGUMENT;
    }
    const DeviceInfo& device = backend->devices[static_cast<std::size_t>(index)];
    *out_info = {
        sizeof(kf_device_info_v1), abi_view(device.id), abi_view(device.name),
        KF_BACKEND_CAP_V1_HOST_MEMORY | KF_BACKEND_CAP_V1_DEVICE_MEMORY,
    };
    return KF_STATUS_V1_OK;
}

kf_status_v1 probe_artifact(kf_backend_handle_v1 backend,
                            const kf_artifact_desc_v1* artifact,
                            kf_string_view_v1 device_id,
                            kf_artifact_probe_v1* out_probe) noexcept
{
    try
    {
        if (backend == nullptr || artifact == nullptr || out_probe == nullptr ||
            artifact->struct_size < sizeof(kf_artifact_desc_v1) ||
            out_probe->struct_size < sizeof(kf_artifact_probe_v1))
        {
            return KF_STATUS_V1_INVALID_ARGUMENT;
        }
        const std::string_view format = view(artifact->format);
        const std::string_view device = view(device_id);
        const std::filesystem::path path = abi_path(artifact->path_utf8);
        std::error_code error;
        const bool regular_file = std::filesystem::is_regular_file(path, error) && !error;
        const bool supported = format == "tensorrt-engine" && regular_file &&
                               find_device(backend->devices, device) != nullptr;
        *out_probe = {
            sizeof(kf_artifact_probe_v1),
            supported ? KF_ARTIFACT_SUPPORT_V1_SUPPORTED : KF_ARTIFACT_SUPPORT_V1_UNSUPPORTED,
            supported ? (KF_BACKEND_CAP_V1_HOST_MEMORY | KF_BACKEND_CAP_V1_DEVICE_MEMORY) : 0U,
        };
        return KF_STATUS_V1_OK;
    }
    catch (...)
    {
        return map_exception();
    }
}

kf_status_v1 load_model(kf_backend_handle_v1 backend,
                        const kf_model_load_info_v1* info,
                        kf_model_handle_v1* out_model) noexcept
{
    try
    {
        if (backend == nullptr || info == nullptr || out_model == nullptr ||
            info->struct_size < sizeof(kf_model_load_info_v1) ||
            info->artifact.struct_size < sizeof(kf_artifact_desc_v1))
        {
            return KF_STATUS_V1_INVALID_ARGUMENT;
        }
        if (view(info->artifact.format) != "tensorrt-engine")
        {
            g_last_error = "TensorRT backend requires artifact format tensorrt-engine";
            return KF_STATUS_V1_INCOMPATIBLE_ARTIFACT;
        }
        const std::string device_id(view(info->device_id));
        const DeviceInfo* device = find_device(backend->devices, device_id);
        if (device == nullptr)
        {
            g_last_error = "requested CUDA device is unavailable";
            return KF_STATUS_V1_DEVICE_UNAVAILABLE;
        }

        kfcore::tensorrt::EngineOptions options;
        options.device_id = device->ordinal;
        auto model = std::make_unique<kf_model_handle_v1_t>();
        model->backend = backend;
        model->device_id = device_id;
        model->engine = kfcore::tensorrt::Engine::load(abi_path(info->artifact.path_utf8), options);
        *out_model = model.release();
        g_last_error.clear();
        return KF_STATUS_V1_OK;
    }
    catch (...)
    {
        return map_exception();
    }
}

void destroy_model(kf_model_handle_v1 model) noexcept
{
    delete model;
}

kf_status_v1 get_tensor_count(kf_model_handle_v1 model, std::uint64_t* out_count) noexcept
{
    if (model == nullptr || !model->engine || out_count == nullptr)
    {
        return KF_STATUS_V1_INVALID_ARGUMENT;
    }
    *out_count = static_cast<std::uint64_t>(model->engine->tensors().size());
    return KF_STATUS_V1_OK;
}

kf_status_v1 get_tensor_info(kf_model_handle_v1 model,
                             std::uint64_t index,
                             kf_tensor_desc_v1* out_info) noexcept
{
    try
    {
        if (model == nullptr || !model->engine || out_info == nullptr ||
            out_info->struct_size < sizeof(kf_tensor_desc_v1) ||
            index >= model->engine->tensors().size())
        {
            return KF_STATUS_V1_INVALID_ARGUMENT;
        }
        const auto& tensor = model->engine->tensors()[static_cast<std::size_t>(index)];
        *out_info = {
            sizeof(kf_tensor_desc_v1),
            abi_view(tensor.name),
            abi_type(tensor.data_type),
            tensor.declared_shape.empty() ? nullptr : tensor.declared_shape.data(),
            static_cast<std::uint32_t>(tensor.declared_shape.size()),
            tensor.mode == kfcore::tensorrt::TensorIoMode::Input ?
                KF_TENSOR_IO_V1_INPUT : KF_TENSOR_IO_V1_OUTPUT,
        };
        return KF_STATUS_V1_OK;
    }
    catch (...)
    {
        return map_exception();
    }
}

kf_status_v1 create_context(kf_model_handle_v1 model,
                            const kf_context_create_info_v1* info,
                            kf_context_handle_v1* out_context) noexcept
{
    try
    {
        if (model == nullptr || !model->engine || info == nullptr || out_context == nullptr ||
            info->struct_size < sizeof(kf_context_create_info_v1))
        {
            return KF_STATUS_V1_INVALID_ARGUMENT;
        }
        auto context = std::make_unique<kf_context_handle_v1_t>();
        context->model = model;
        context->executor = model->engine->create_executor();
        *out_context = context.release();
        return KF_STATUS_V1_OK;
    }
    catch (...)
    {
        return map_exception();
    }
}

void destroy_context(kf_context_handle_v1 context) noexcept
{
    delete context;
}

kf_status_v1 run(kf_context_handle_v1 context,
                 const kf_tensor_view_v1* inputs,
                 std::uint64_t input_count,
                 kf_mutable_tensor_view_v1* outputs,
                 std::uint64_t output_count) noexcept
{
    try
    {
        if (context == nullptr || context->model == nullptr || !context->executor ||
            (input_count != 0U && inputs == nullptr) ||
            (output_count != 0U && outputs == nullptr) ||
            input_count > static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)()) ||
            output_count > static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)()))
        {
            return KF_STATUS_V1_INVALID_ARGUMENT;
        }

        std::vector<kfcore::tensorrt::TensorView> runtime_inputs;
        runtime_inputs.reserve(static_cast<std::size_t>(input_count));
        for (std::uint64_t index = 0U; index < input_count; ++index)
        {
            const kf_tensor_view_v1& input = inputs[index];
            if (input.struct_size < sizeof(kf_tensor_view_v1) ||
                (input.rank != 0U && input.dimensions == nullptr))
            {
                return KF_STATUS_V1_INVALID_ARGUMENT;
            }
            if (input.memory_kind == KF_MEMORY_KIND_V1_DEVICE &&
                view(input.device_id) != context->model->device_id)
            {
                g_last_error = "device tensor does not belong to the model CUDA device";
                return KF_STATUS_V1_INVALID_ARGUMENT;
            }
            kfcore::tensorrt::TensorView tensor;
            tensor.name = std::string(view(input.name));
            tensor.data_type = runtime_type(input.data_type);
            if (input.rank != 0U)
            {
                tensor.shape.assign(input.dimensions, input.dimensions + input.rank);
            }
            tensor.data = input.data;
            tensor.byte_size = static_cast<std::size_t>(input.byte_size);
            tensor.memory_kind = runtime_memory(input.memory_kind);
            runtime_inputs.push_back(std::move(tensor));
        }

        std::vector<kfcore::tensorrt::MutableTensorView> runtime_outputs;
        runtime_outputs.reserve(static_cast<std::size_t>(output_count));
        for (std::uint64_t index = 0U; index < output_count; ++index)
        {
            const kf_mutable_tensor_view_v1& output = outputs[index];
            if (output.struct_size < sizeof(kf_mutable_tensor_view_v1) ||
                (output.rank != 0U && output.dimensions == nullptr))
            {
                return KF_STATUS_V1_INVALID_ARGUMENT;
            }
            if (output.memory_kind == KF_MEMORY_KIND_V1_DEVICE &&
                view(output.device_id) != context->model->device_id)
            {
                g_last_error = "device output does not belong to the model CUDA device";
                return KF_STATUS_V1_INVALID_ARGUMENT;
            }
            kfcore::tensorrt::MutableTensorView tensor;
            tensor.name = std::string(view(output.name));
            tensor.data_type = runtime_type(output.data_type);
            if (output.rank != 0U)
            {
                tensor.shape.assign(output.dimensions, output.dimensions + output.rank);
            }
            tensor.data = output.data;
            tensor.byte_size = static_cast<std::size_t>(output.byte_size);
            tensor.memory_kind = runtime_memory(output.memory_kind);
            runtime_outputs.push_back(std::move(tensor));
        }

        context->executor->run(runtime_inputs, runtime_outputs);
        g_last_error.clear();
        return KF_STATUS_V1_OK;
    }
    catch (...)
    {
        return map_exception();
    }
}

kf_status_v1 format_last_error(kf_backend_handle_v1,
                               char* buffer,
                               std::uint64_t capacity,
                               std::uint64_t* out_required) noexcept
{
    if (out_required == nullptr)
    {
        return KF_STATUS_V1_INVALID_ARGUMENT;
    }
    const std::uint64_t required = static_cast<std::uint64_t>(g_last_error.size()) + 1U;
    *out_required = required;
    if (buffer == nullptr || capacity == 0U)
    {
        return KF_STATUS_V1_OK;
    }
    if (capacity < required)
    {
        return KF_STATUS_V1_INVALID_ARGUMENT;
    }
    std::memcpy(buffer, g_last_error.c_str(), static_cast<std::size_t>(required));
    return KF_STATUS_V1_OK;
}

} // namespace

#if defined(_WIN32)
#define KFCORE_BACKEND_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define KFCORE_BACKEND_EXPORT __attribute__((visibility("default")))
#else
#define KFCORE_BACKEND_EXPORT
#endif

extern "C" KFCORE_BACKEND_EXPORT kf_status_v1
kfcore_backend_query_v1(std::uint32_t requested_abi_major,
                        std::uint32_t requested_abi_minor,
                        kf_backend_api_v1* out_api) noexcept
{
    if (out_api == nullptr || out_api->struct_size < sizeof(kf_backend_api_v1) ||
        requested_abi_major != KFCORE_BACKEND_ABI_V1_MAJOR ||
        requested_abi_minor > KFCORE_BACKEND_ABI_V1_MINOR)
    {
        return KF_STATUS_V1_UNSUPPORTED;
    }

    *out_api = {
        sizeof(kf_backend_api_v1),
        KFCORE_BACKEND_ABI_V1_MAJOR,
        KFCORE_BACKEND_ABI_V1_MINOR,
        &create_backend,
        &destroy_backend,
        &get_backend_info,
        &get_device_count,
        &get_device_info,
        &probe_artifact,
        &load_model,
        &destroy_model,
        &get_tensor_count,
        &get_tensor_info,
        &create_context,
        &destroy_context,
        &run,
        &format_last_error,
    };
    return KF_STATUS_V1_OK;
}
