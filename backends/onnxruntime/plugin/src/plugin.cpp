#include "kfcore/runtime/abi/backend_v1.h"

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#ifdef KFCORE_ONNX_CUDA
#include <onnxruntime_session_options_config_keys.h>
#include <cuda_runtime_api.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace
{

thread_local std::string g_last_error;

constexpr std::size_t kBackendInfoV10Size =
    offsetof(kf_backend_info_v1, execution_runtime_major);
constexpr std::size_t kDeviceInfoV10Size =
    offsetof(kf_device_info_v1, compute_capability_major);
constexpr std::size_t kBackendApiV11Size =
    offsetof(kf_backend_api_v1, run_dynamic);

struct RuntimeLibrary final
{
    RuntimeLibrary()
    {
        const char* root = std::getenv("ONNXRUNTIME_ROOT");
        if (root == nullptr || *root == '\0')
            throw std::runtime_error("ONNXRUNTIME_ROOT is required to load ONNX Runtime");
        const std::filesystem::path configured =
            std::filesystem::path(root) / KFCORE_ONNX_RUNTIME_RELATIVE_PATH;
        if (!configured.is_absolute())
            throw std::runtime_error("ONNXRUNTIME_ROOT must resolve ONNX Runtime to an absolute path");

        std::error_code path_error;
        const std::filesystem::path path = std::filesystem::canonical(configured, path_error);
        if (path_error || !std::filesystem::is_regular_file(path, path_error) || path_error)
            throw std::runtime_error("ONNX Runtime library is not a canonical regular file: " +
                                     configured.u8string());

#if defined(_WIN32)
        module = LoadLibraryExW(path.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (module == nullptr)
            throw std::runtime_error("LoadLibraryExW failed for ONNX Runtime: " + path.u8string());
        using GetApiBase = const OrtApiBase* (ORT_API_CALL*)();
        const auto get_api_base =
            reinterpret_cast<GetApiBase>(GetProcAddress(module, "OrtGetApiBase"));
        if (get_api_base == nullptr)
        {
            close();
            throw std::runtime_error("GetProcAddress failed for OrtGetApiBase");
        }
        const OrtApiBase* base = get_api_base();
#else
        module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (module == nullptr)
        {
            const char* error = dlerror();
            throw std::runtime_error(std::string("dlopen failed for ONNX Runtime: ") +
                                     path.string() + (error == nullptr ? "" : ": ") +
                                     (error == nullptr ? "" : error));
        }
        dlerror();
        using GetApiBase = const OrtApiBase* (*)();
        const auto get_api_base = reinterpret_cast<GetApiBase>(dlsym(module, "OrtGetApiBase"));
        const char* symbol_error = dlerror();
        if (symbol_error != nullptr || get_api_base == nullptr)
        {
            const std::string detail = symbol_error == nullptr ? "" : std::string(": ") + symbol_error;
            close();
            throw std::runtime_error("dlsym failed for OrtGetApiBase" + detail);
        }
        const OrtApiBase* base = get_api_base();
#endif
        const OrtApi* api = base == nullptr ? nullptr : base->GetApi(ORT_API_VERSION);
        if (api == nullptr)
        {
            close();
            throw std::runtime_error("ONNX Runtime API version mismatch");
        }
        Ort::InitApi(api);
    }

    ~RuntimeLibrary() { close(); }
    RuntimeLibrary(const RuntimeLibrary&) = delete;
    RuntimeLibrary& operator=(const RuntimeLibrary&) = delete;

private:
    void close() noexcept
    {
#if defined(_WIN32)
        if (module != nullptr)
        {
            (void)FreeLibrary(module);
            module = nullptr;
        }
#else
        if (module != nullptr)
        {
            (void)dlclose(module);
            module = nullptr;
        }
#endif
    }
#if defined(_WIN32)
    HMODULE module = nullptr;
#else
    void* module = nullptr;
#endif
};

struct DeviceInfo
{
    std::string id;
    std::string name;
    bool cuda = false;
    int ordinal = -1;
};

struct TensorMeta
{
    std::string name;
    ONNXTensorElementDataType ort_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    kf_data_type_v1 abi_type = 0U;
    std::vector<std::int64_t> shape;
    bool input = true;
};

std::string_view view(kf_string_view_v1 value)
{
    if (value.size == 0U) return {};
    if (value.data == nullptr ||
        value.size > static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)()))
        throw std::invalid_argument("invalid ABI string view");
    return std::string_view(value.data, static_cast<std::size_t>(value.size));
}

kf_string_view_v1 abi_view(std::string_view value) noexcept
{
    return { value.data(), static_cast<std::uint64_t>(value.size()) };
}

std::filesystem::path abi_path(kf_string_view_v1 value)
{
    const std::string_view text = view(value);
    if (text.empty()) throw std::invalid_argument("artifact path is empty");
    return std::filesystem::u8path(text.begin(), text.end());
}

kf_data_type_v1 abi_type(ONNXTensorElementDataType type)
{
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return KF_DATA_TYPE_V1_FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return KF_DATA_TYPE_V1_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return KF_DATA_TYPE_V1_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return KF_DATA_TYPE_V1_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return KF_DATA_TYPE_V1_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return KF_DATA_TYPE_V1_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return KF_DATA_TYPE_V1_BOOL;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return KF_DATA_TYPE_V1_BFLOAT16;
    default: throw std::invalid_argument("unsupported ONNX tensor element type");
    }
}

ONNXTensorElementDataType ort_type(kf_data_type_v1 type)
{
    switch (type)
    {
    case KF_DATA_TYPE_V1_FLOAT32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case KF_DATA_TYPE_V1_FLOAT16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case KF_DATA_TYPE_V1_INT8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case KF_DATA_TYPE_V1_INT32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case KF_DATA_TYPE_V1_INT64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case KF_DATA_TYPE_V1_UINT8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case KF_DATA_TYPE_V1_BOOL: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    case KF_DATA_TYPE_V1_BFLOAT16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    default: throw std::invalid_argument("unsupported ABI tensor element type");
    }
}

std::size_t element_size(ONNXTensorElementDataType type)
{
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return sizeof(std::uint16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return sizeof(std::int8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return sizeof(std::int32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return sizeof(std::int64_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return sizeof(std::uint8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return sizeof(bool);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return sizeof(std::uint16_t);
    default: throw std::invalid_argument("unsupported tensor element size");
    }
}

std::size_t required_bytes(const std::int64_t* dimensions,
                           std::uint32_t rank,
                           ONNXTensorElementDataType type,
                           bool allow_zero = false)
{
    if (rank != 0U && dimensions == nullptr)
        throw std::invalid_argument("tensor dimensions are null");
    std::size_t count = 1U;
    for (std::uint32_t index = 0U; index < rank; ++index)
    {
        if (dimensions[index] < 0 || (!allow_zero && dimensions[index] == 0))
            throw std::invalid_argument("runtime tensor dimensions are invalid");
        const auto dimension = static_cast<std::uintmax_t>(dimensions[index]);
        if (dimension > static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)()) ||
            (count != 0U && static_cast<std::size_t>(dimension) >
                (std::numeric_limits<std::size_t>::max)() / count))
            throw std::overflow_error("tensor element count overflow");
        count *= static_cast<std::size_t>(dimension);
    }
    const std::size_t scalar = element_size(type);
    if (count > (std::numeric_limits<std::size_t>::max)() / scalar)
        throw std::overflow_error("tensor byte count overflow");
    return count * scalar;
}

std::size_t required_bytes(const std::vector<std::int64_t>& dimensions,
                           ONNXTensorElementDataType type,
                           bool allow_zero = false)
{
    if (dimensions.size() > static_cast<std::size_t>((std::numeric_limits<std::uint32_t>::max)()))
        throw std::overflow_error("tensor rank exceeds ABI range");
    return required_bytes(dimensions.data(), static_cast<std::uint32_t>(dimensions.size()),
                          type, allow_zero);
}

void validate_shape(const TensorMeta& expected,
                    const std::int64_t* dimensions,
                    std::uint32_t rank)
{
    if (rank != expected.shape.size())
        throw std::invalid_argument("runtime tensor rank does not match model contract");
    for (std::uint32_t index = 0U; index < rank; ++index)
    {
        const std::int64_t actual = dimensions[index];
        const std::int64_t declared = expected.shape[index];
        if (actual <= 0 || (declared > 0 && actual != declared))
            throw std::invalid_argument("runtime tensor dimensions do not match model contract");
    }
}

const DeviceInfo* find_device(const std::vector<DeviceInfo>& devices, std::string_view id)
{
    const auto iterator = std::find_if(devices.begin(), devices.end(),
        [id](const DeviceInfo& device) { return device.id == id; });
    return iterator == devices.end() ? nullptr : &*iterator;
}

std::vector<DeviceInfo> enumerate_devices()
{
    std::vector<DeviceInfo> result;
    result.push_back({ "cpu", "CPU", false, -1 });
#ifdef KFCORE_ONNX_CUDA
    int count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&count);
    if (count_status == cudaErrorNoDevice || count_status == cudaErrorInsufficientDriver)
    {
        (void)cudaGetLastError();
        return result;
    }
    if (count_status != cudaSuccess)
        throw std::runtime_error(std::string("cudaGetDeviceCount failed: ") +
                                 cudaGetErrorString(count_status));
    for (int ordinal = 0; ordinal < count; ++ordinal)
    {
        cudaDeviceProp properties {};
        const cudaError_t status = cudaGetDeviceProperties(&properties, ordinal);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") +
                                     cudaGetErrorString(status));
        result.push_back({ "cuda:" + std::to_string(ordinal), properties.name, true, ordinal });
    }
#endif
    return result;
}

kf_status_v1 map_exception() noexcept
{
    try
    {
        throw;
    }
    catch (const Ort::Exception& error)
    {
        g_last_error = error.what();
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
    catch (const std::overflow_error& error)
    {
        g_last_error = error.what();
        return KF_STATUS_V1_OUT_OF_MEMORY;
    }
    catch (const std::exception& error)
    {
        g_last_error = error.what();
        return KF_STATUS_V1_RUNTIME_FAILURE;
    }
    catch (...)
    {
        g_last_error = "unknown ONNX Runtime backend failure";
        return KF_STATUS_V1_INTERNAL_FAILURE;
    }
}

} // namespace

struct kf_backend_handle_v1_t
{
    std::unique_ptr<RuntimeLibrary> library;
    std::unique_ptr<Ort::Env> environment;
    std::vector<DeviceInfo> devices;
};

struct kf_model_handle_v1_t
{
    kf_backend_handle_v1 backend = nullptr;
    std::unique_ptr<Ort::Session> session;
    std::vector<TensorMeta> inputs;
    std::vector<TensorMeta> outputs;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::string device_id;
    bool cuda = false;
    int cuda_ordinal = -1;
};

struct kf_context_handle_v1_t
{
    kf_model_handle_v1 model = nullptr;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

namespace
{

void collect_metadata(kf_model_handle_v1 model)
{
    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t input_count = model->session->GetInputCount();
    const std::size_t output_count = model->session->GetOutputCount();
    model->inputs.reserve(input_count);
    model->outputs.reserve(output_count);
    for (std::size_t index = 0U; index < input_count; ++index)
    {
        auto name = model->session->GetInputNameAllocated(index, allocator);
        auto info = model->session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo();
        model->inputs.push_back({name.get(), info.GetElementType(),
                                 abi_type(info.GetElementType()), info.GetShape(), true});
    }
    for (std::size_t index = 0U; index < output_count; ++index)
    {
        auto name = model->session->GetOutputNameAllocated(index, allocator);
        auto info = model->session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo();
        model->outputs.push_back({name.get(), info.GetElementType(),
                                  abi_type(info.GetElementType()), info.GetShape(), false});
    }
    model->input_names.reserve(model->inputs.size());
    for (const auto& input : model->inputs) model->input_names.push_back(input.name.c_str());
    model->output_names.reserve(model->outputs.size());
    for (const auto& output : model->outputs) model->output_names.push_back(output.name.c_str());
}

Ort::MemoryInfo memory_info(kf_memory_kind_v1 kind,
                            const kf_model_handle_v1 model,
                            kf_string_view_v1 device_id)
{
    if (kind == KF_MEMORY_KIND_V1_HOST)
        return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    if (kind == KF_MEMORY_KIND_V1_PINNED_HOST)
        throw std::invalid_argument("pinned-host tensor I/O is not advertised by ONNX Runtime ABI v1");
    if (kind == KF_MEMORY_KIND_V1_DEVICE)
    {
        if (!model->cuda || view(device_id) != model->device_id)
            throw std::invalid_argument("device tensor does not belong to the model CUDA device");
        return Ort::MemoryInfo("Cuda", OrtDeviceAllocator, model->cuda_ordinal, OrtMemTypeDefault);
    }
    throw std::invalid_argument("unknown tensor memory kind");
}

template <typename View>
const View& named_view(const View* values, std::uint64_t count, const std::string& name)
{
    const View* found = nullptr;
    for (std::uint64_t index = 0U; index < count; ++index)
    {
        if (view(values[index].name) == name)
        {
            if (found != nullptr) throw std::invalid_argument("duplicate tensor binding name");
            found = &values[index];
        }
    }
    if (found == nullptr) throw std::invalid_argument("missing tensor binding: " + name);
    return *found;
}

std::vector<Ort::Value> make_input_values(kf_model_handle_v1 model,
                                          const kf_tensor_view_v1* inputs,
                                          std::uint64_t input_count,
                                          std::vector<Ort::MemoryInfo>& memory)
{
    if (input_count != model->inputs.size() || (input_count != 0U && inputs == nullptr))
        throw std::invalid_argument("input tensor count does not match model contract");
    std::vector<Ort::Value> values;
    memory.reserve(model->inputs.size());
    values.reserve(model->inputs.size());
    for (const TensorMeta& meta : model->inputs)
    {
        const auto& input = named_view(inputs, input_count, meta.name);
        if (input.struct_size < sizeof(kf_tensor_view_v1) ||
            ort_type(input.data_type) != meta.ort_type)
            throw std::invalid_argument("input tensor type does not match model contract");
        validate_shape(meta, input.dimensions, input.rank);
        const std::size_t bytes = required_bytes(input.dimensions, input.rank, meta.ort_type);
        if (input.data == nullptr || input.byte_size < bytes)
            throw std::invalid_argument("input tensor storage is smaller than its runtime shape");
        memory.push_back(memory_info(input.memory_kind, model, input.device_id));
        values.emplace_back(Ort::Value::CreateTensor(
            memory.back(), const_cast<void*>(input.data), static_cast<std::size_t>(input.byte_size),
            input.dimensions, input.rank, meta.ort_type));
    }
    return values;
}

struct ContextUse final
{
    explicit ContextUse(kf_context_handle_v1 context) : context_(context)
    {
        if (context_->in_use.test_and_set(std::memory_order_acquire))
            throw std::runtime_error("concurrent calls on one ONNX Runtime execution context are unsupported");
    }
    ~ContextUse() { context_->in_use.clear(std::memory_order_release); }
    kf_context_handle_v1 context_;
};

kf_status_v1 create_backend(const kf_backend_create_info_v1* info,
                            kf_backend_handle_v1* out_backend) noexcept
{
    try
    {
        if (info == nullptr || info->struct_size < sizeof(kf_backend_create_info_v1) ||
            out_backend == nullptr)
            return KF_STATUS_V1_INVALID_ARGUMENT;
        auto backend = std::make_unique<kf_backend_handle_v1_t>();
        backend->library = std::make_unique<RuntimeLibrary>();
        backend->environment = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR,
                                                          "kfcore_backend_onnxruntime");
        backend->devices = enumerate_devices();
        *out_backend = backend.release();
        g_last_error.clear();
        return KF_STATUS_V1_OK;
    }
    catch (...)
    {
        return map_exception();
    }
}

void destroy_backend(kf_backend_handle_v1 backend) noexcept { delete backend; }

kf_status_v1 get_backend_info(kf_backend_handle_v1 backend,
                              kf_backend_info_v1* out_info) noexcept
{
    if (backend == nullptr || out_info == nullptr || out_info->struct_size < kBackendInfoV10Size)
        return KF_STATUS_V1_INVALID_ARGUMENT;
    const std::uint32_t caller_size = out_info->struct_size;
    static constexpr std::string_view kId = "onnxruntime";
    static constexpr std::string_view kName = "KFCore ONNX Runtime";
    std::uint64_t capabilities = KF_BACKEND_CAP_V1_HOST_MEMORY |
                                 KF_BACKEND_CAP_V1_DYNAMIC_HOST_OUTPUT;
#ifdef KFCORE_ONNX_CUDA
    if (backend->devices.size() > 1U) capabilities |= KF_BACKEND_CAP_V1_DEVICE_MEMORY;
#endif
    out_info->struct_size = sizeof(kf_backend_info_v1);
    out_info->backend_id = abi_view(kId);
    out_info->backend_name = abi_view(kName);
    out_info->backend_version_major = 1U;
    out_info->backend_version_minor = 2U;
    out_info->backend_version_patch = 0U;
    out_info->capabilities = capabilities;
    if (caller_size >= sizeof(kf_backend_info_v1))
    {
        out_info->execution_runtime_major = 0U;
        out_info->execution_runtime_minor = 0U;
        out_info->execution_runtime_patch = 0U;
    }
    return KF_STATUS_V1_OK;
}

kf_status_v1 get_device_count(kf_backend_handle_v1 backend,
                              std::uint64_t* out_count) noexcept
{
    if (backend == nullptr || out_count == nullptr) return KF_STATUS_V1_INVALID_ARGUMENT;
    *out_count = static_cast<std::uint64_t>(backend->devices.size());
    return KF_STATUS_V1_OK;
}

kf_status_v1 get_device_info(kf_backend_handle_v1 backend,
                             std::uint64_t index,
                             kf_device_info_v1* out_info) noexcept
{
    if (backend == nullptr || out_info == nullptr || out_info->struct_size < kDeviceInfoV10Size ||
        index >= backend->devices.size())
        return KF_STATUS_V1_INVALID_ARGUMENT;
    const std::uint32_t caller_size = out_info->struct_size;
    const DeviceInfo& device = backend->devices[static_cast<std::size_t>(index)];
    out_info->struct_size = sizeof(kf_device_info_v1);
    out_info->device_id = abi_view(device.id);
    out_info->device_name = abi_view(device.name);
    out_info->capabilities = KF_BACKEND_CAP_V1_HOST_MEMORY |
                             KF_BACKEND_CAP_V1_DYNAMIC_HOST_OUTPUT |
                             (device.cuda ? KF_BACKEND_CAP_V1_DEVICE_MEMORY : 0U);
    if (caller_size >= sizeof(kf_device_info_v1))
    {
        out_info->compute_capability_major = 0U;
        out_info->compute_capability_minor = 0U;
    }
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
            return KF_STATUS_V1_INVALID_ARGUMENT;
        const DeviceInfo* device = find_device(backend->devices, view(device_id));
        std::error_code error;
        const bool regular = std::filesystem::is_regular_file(abi_path(artifact->path_utf8), error) && !error;
        const bool supported = view(artifact->format) == "onnx" && regular && device != nullptr;
        const std::uint64_t capabilities = supported
            ? (KF_BACKEND_CAP_V1_HOST_MEMORY | KF_BACKEND_CAP_V1_DYNAMIC_HOST_OUTPUT |
               (device->cuda ? KF_BACKEND_CAP_V1_DEVICE_MEMORY : 0U))
            : 0U;
        *out_probe = {sizeof(kf_artifact_probe_v1),
            supported ? KF_ARTIFACT_SUPPORT_V1_SUPPORTED : KF_ARTIFACT_SUPPORT_V1_UNSUPPORTED,
            capabilities};
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
        if (backend == nullptr || backend->environment == nullptr || info == nullptr ||
            out_model == nullptr || info->struct_size < sizeof(kf_model_load_info_v1) ||
            info->artifact.struct_size < sizeof(kf_artifact_desc_v1))
            return KF_STATUS_V1_INVALID_ARGUMENT;
        if (view(info->artifact.format) != "onnx")
        {
            g_last_error = "ONNX Runtime backend requires artifact format onnx";
            return KF_STATUS_V1_INCOMPATIBLE_ARTIFACT;
        }
        const std::string device_id(view(info->device_id));
        const DeviceInfo* device = find_device(backend->devices, device_id);
        if (device == nullptr)
        {
            g_last_error = "requested ONNX Runtime device is unavailable";
            return KF_STATUS_V1_DEVICE_UNAVAILABLE;
        }

        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef KFCORE_ONNX_CUDA
        if (device->cuda)
        {
            OrtCUDAProviderOptionsV2* raw_options = nullptr;
            Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&raw_options));
            const auto release = [](OrtCUDAProviderOptionsV2* value)
            {
                Ort::GetApi().ReleaseCUDAProviderOptions(value);
            };
            const std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(release)> cuda_options(raw_options, release);
            const std::string ordinal = std::to_string(device->ordinal);
            const char* keys[] = {"device_id", "use_tf32"};
            const char* values[] = {ordinal.c_str(), "0"};
            Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
                cuda_options.get(), keys, values, sizeof(keys) / sizeof(keys[0])));
            options.AppendExecutionProvider_CUDA_V2(*cuda_options);
            options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback,
                                   KFCORE_ONNX_CUDA_ALLOW_CPU_NODES ? "0" : "1");
        }
#endif
        const std::filesystem::path path = abi_path(info->artifact.path_utf8);
        auto model = std::make_unique<kf_model_handle_v1_t>();
        model->backend = backend;
        model->device_id = device_id;
        model->cuda = device->cuda;
        model->cuda_ordinal = device->ordinal;
#ifdef _WIN32
        model->session = std::make_unique<Ort::Session>(*backend->environment, path.c_str(), options);
#else
        model->session = std::make_unique<Ort::Session>(*backend->environment, path.string().c_str(), options);
#endif
        collect_metadata(model.get());
        *out_model = model.release();
        g_last_error.clear();
        return KF_STATUS_V1_OK;
    }
    catch (...)
    {
        return map_exception();
    }
}

void destroy_model(kf_model_handle_v1 model) noexcept { delete model; }

kf_status_v1 get_tensor_count(kf_model_handle_v1 model,
                              std::uint64_t* out_count) noexcept
{
    if (model == nullptr || out_count == nullptr) return KF_STATUS_V1_INVALID_ARGUMENT;
    *out_count = static_cast<std::uint64_t>(model->inputs.size() + model->outputs.size());
    return KF_STATUS_V1_OK;
}

kf_status_v1 get_tensor_info(kf_model_handle_v1 model,
                             std::uint64_t index,
                             kf_tensor_desc_v1* out_info) noexcept
{
    try
    {
        if (model == nullptr || out_info == nullptr ||
            out_info->struct_size < sizeof(kf_tensor_desc_v1) ||
            index >= model->inputs.size() + model->outputs.size())
            return KF_STATUS_V1_INVALID_ARGUMENT;
        const bool input = index < model->inputs.size();
        const TensorMeta& tensor = input
            ? model->inputs[static_cast<std::size_t>(index)]
            : model->outputs[static_cast<std::size_t>(index - model->inputs.size())];
        if (tensor.shape.size() > static_cast<std::size_t>((std::numeric_limits<std::uint32_t>::max)()))
            throw std::overflow_error("tensor rank exceeds ABI v1 range");
        *out_info = {sizeof(kf_tensor_desc_v1), abi_view(tensor.name), tensor.abi_type,
            tensor.shape.empty() ? nullptr : tensor.shape.data(),
            static_cast<std::uint32_t>(tensor.shape.size()),
            input ? KF_TENSOR_IO_V1_INPUT : KF_TENSOR_IO_V1_OUTPUT};
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
        if (model == nullptr || model->session == nullptr || info == nullptr ||
            out_context == nullptr || info->struct_size < sizeof(kf_context_create_info_v1))
            return KF_STATUS_V1_INVALID_ARGUMENT;
        auto context = std::make_unique<kf_context_handle_v1_t>();
        context->model = model;
        *out_context = context.release();
        return KF_STATUS_V1_OK;
    }
    catch (...)
    {
        return map_exception();
    }
}

void destroy_context(kf_context_handle_v1 context) noexcept { delete context; }

kf_status_v1 run(kf_context_handle_v1 context,
                 const kf_tensor_view_v1* inputs,
                 std::uint64_t input_count,
                 kf_mutable_tensor_view_v1* outputs,
                 std::uint64_t output_count) noexcept
{
    try
    {
        if (context == nullptr || context->model == nullptr || context->model->session == nullptr ||
            output_count != context->model->outputs.size() ||
            (output_count != 0U && outputs == nullptr))
            return KF_STATUS_V1_INVALID_ARGUMENT;
        ContextUse use(context);
        std::vector<Ort::MemoryInfo> input_memory;
        auto input_values = make_input_values(context->model, inputs, input_count, input_memory);

        std::vector<Ort::MemoryInfo> output_memory;
        std::vector<Ort::Value> output_values;
        output_memory.reserve(context->model->outputs.size());
        output_values.reserve(context->model->outputs.size());
        for (const TensorMeta& meta : context->model->outputs)
        {
            const auto& output = named_view(outputs, output_count, meta.name);
            if (output.struct_size < sizeof(kf_mutable_tensor_view_v1) ||
                ort_type(output.data_type) != meta.ort_type)
                throw std::invalid_argument("output tensor type does not match model contract");
            validate_shape(meta, output.dimensions, output.rank);
            const std::size_t bytes = required_bytes(output.dimensions, output.rank, meta.ort_type);
            if (output.data == nullptr || output.byte_size < bytes)
                throw std::invalid_argument("output tensor storage is smaller than its runtime shape");
            output_memory.push_back(memory_info(output.memory_kind, context->model, output.device_id));
            output_values.emplace_back(Ort::Value::CreateTensor(
                output_memory.back(), output.data, static_cast<std::size_t>(output.byte_size),
                output.dimensions, output.rank, meta.ort_type));
        }

        Ort::RunOptions run_options;
        context->model->session->Run(run_options,
            context->model->input_names.data(), input_values.data(), input_values.size(),
            context->model->output_names.data(), output_values.data(), output_values.size());
        g_last_error.clear();
        return KF_STATUS_V1_OK;
    }
    catch (...)
    {
        return map_exception();
    }
}

kf_status_v1 run_dynamic(kf_context_handle_v1 context,
                         const kf_tensor_view_v1* inputs,
                         std::uint64_t input_count,
                         kf_dynamic_output_v1* outputs,
                         std::uint64_t output_count) noexcept
{
    try
    {
        if (context == nullptr || context->model == nullptr || context->model->session == nullptr ||
            output_count != context->model->outputs.size() ||
            (output_count != 0U && outputs == nullptr))
            return KF_STATUS_V1_INVALID_ARGUMENT;
        ContextUse use(context);
        std::vector<Ort::MemoryInfo> input_memory;
        auto input_values = make_input_values(context->model, inputs, input_count, input_memory);

        for (const TensorMeta& meta : context->model->outputs)
        {
            const auto& output = named_view(outputs, output_count, meta.name);
            if (output.struct_size < sizeof(kf_dynamic_output_v1) ||
                output.memory_kind != KF_MEMORY_KIND_V1_HOST || output.data == nullptr ||
                output.capacity_bytes == 0U || ort_type(output.data_type) != meta.ort_type)
                throw std::invalid_argument("dynamic output binding is invalid");
        }

        Ort::IoBinding binding(*context->model->session);
        for (std::size_t index = 0U; index < context->model->inputs.size(); ++index)
            binding.BindInput(context->model->input_names[index], input_values[index]);
        const Ort::MemoryInfo cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        for (const char* name : context->model->output_names)
            binding.BindOutput(name, cpu);

        Ort::RunOptions run_options;
        context->model->session->Run(run_options, binding);
        binding.SynchronizeOutputs();
        auto actual_values = binding.GetOutputValues();
        if (actual_values.size() != context->model->outputs.size())
            throw std::runtime_error("ONNX Runtime dynamic output count mismatch");

        for (std::size_t index = 0U; index < context->model->outputs.size(); ++index)
        {
            const TensorMeta& meta = context->model->outputs[index];
            auto& output = const_cast<kf_dynamic_output_v1&>(named_view(outputs, output_count, meta.name));
            auto info = actual_values[index].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != meta.ort_type)
                throw std::runtime_error("ONNX Runtime dynamic output type mismatch");
            const auto shape = info.GetShape();
            if (shape.size() > KFCORE_TENSOR_MAX_RANK_V1)
                throw std::runtime_error("ONNX Runtime dynamic output rank exceeds ABI capacity");
            const std::size_t bytes = required_bytes(shape, meta.ort_type, true);
            if (bytes > static_cast<std::size_t>(output.capacity_bytes))
                throw std::runtime_error("ONNX Runtime dynamic output exceeds caller capacity");
            if (bytes != 0U)
            {
                void* data = nullptr;
                Ort::ThrowOnError(Ort::GetApi().GetTensorMutableData(actual_values[index], &data));
                if (data == nullptr) throw std::runtime_error("ONNX Runtime returned null output storage");
                std::memcpy(output.data, data, bytes);
            }
            std::fill(output.dimensions,
                      output.dimensions + KFCORE_TENSOR_MAX_RANK_V1, INT64_C(0));
            for (std::size_t dimension = 0U; dimension < shape.size(); ++dimension)
                output.dimensions[dimension] = shape[dimension];
            output.rank = static_cast<std::uint32_t>(shape.size());
            output.byte_size = static_cast<std::uint64_t>(bytes);
        }
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
    if (out_required == nullptr) return KF_STATUS_V1_INVALID_ARGUMENT;
    const std::uint64_t required = static_cast<std::uint64_t>(g_last_error.size()) + 1U;
    *out_required = required;
    if (buffer == nullptr || capacity == 0U) return KF_STATUS_V1_OK;
    if (capacity < required) return KF_STATUS_V1_INVALID_ARGUMENT;
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
    if (out_api == nullptr || requested_abi_major != KFCORE_BACKEND_ABI_V1_MAJOR ||
        requested_abi_minor > KFCORE_BACKEND_ABI_V1_MINOR)
        return KF_STATUS_V1_UNSUPPORTED;

    const std::uint32_t caller_size = out_api->struct_size;
    const std::size_t required_size = requested_abi_minor >= 2U
        ? sizeof(kf_backend_api_v1)
        : kBackendApiV11Size;
    if (caller_size < required_size)
        return KF_STATUS_V1_UNSUPPORTED;

    kf_backend_api_v1 api {};
    api.struct_size = static_cast<std::uint32_t>(required_size);
    api.abi_major = KFCORE_BACKEND_ABI_V1_MAJOR;
    api.abi_minor = requested_abi_minor;
    api.create_backend = &create_backend;
    api.destroy_backend = &destroy_backend;
    api.get_backend_info = &get_backend_info;
    api.get_device_count = &get_device_count;
    api.get_device_info = &get_device_info;
    api.probe_artifact = &probe_artifact;
    api.load_model = &load_model;
    api.destroy_model = &destroy_model;
    api.get_tensor_count = &get_tensor_count;
    api.get_tensor_info = &get_tensor_info;
    api.create_context = &create_context;
    api.destroy_context = &destroy_context;
    api.run = &run;
    api.format_last_error = &format_last_error;
    if (requested_abi_minor >= 2U)
        api.run_dynamic = &run_dynamic;

    std::memcpy(out_api, &api, required_size);
    return KF_STATUS_V1_OK;
}
