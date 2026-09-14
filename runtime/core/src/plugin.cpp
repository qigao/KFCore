#include "kfcore/runtime/plugin.hpp"

#include "dynamic_library.hpp"
#include "kfcore/runtime/abi/backend_v1.h"
#include "kfcore/runtime/error.hpp"

#include <algorithm>
#include <limits>
#include <system_error>
#include <utility>

namespace kfcore::runtime
{
namespace
{

kf_string_view_v1 abi_string(std::string_view value) noexcept
{
    return { value.data(), static_cast<std::uint64_t>(value.size()) };
}

std::string copy_string(kf_string_view_v1 value)
{
    if (value.size == 0U)
    {
        return {};
    }
    if (value.data == nullptr || value.size > static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)()))
    {
        throw RuntimeError(RuntimeErrorCode::BackendFailure,
                           "runtime backend returned an invalid string view");
    }
    return std::string(value.data, static_cast<std::size_t>(value.size));
}

kf_data_type_v1 to_abi(DataType type)
{
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
    throw RuntimeError(RuntimeErrorCode::InvalidArgument, "unknown runtime data type");
}

DataType from_abi(kf_data_type_v1 type)
{
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
    default:
        throw RuntimeError(RuntimeErrorCode::BackendFailure,
                           "runtime backend returned an unknown data type");
    }
}

kf_memory_kind_v1 to_abi(MemoryKind kind)
{
    switch (kind)
    {
    case MemoryKind::Host: return KF_MEMORY_KIND_V1_HOST;
    case MemoryKind::PinnedHost: return KF_MEMORY_KIND_V1_PINNED_HOST;
    case MemoryKind::Device: return KF_MEMORY_KIND_V1_DEVICE;
    }
    throw RuntimeError(RuntimeErrorCode::InvalidArgument, "unknown runtime memory kind");
}

std::uint32_t checked_rank(const TensorShape& shape)
{
    if (shape.size() > static_cast<std::size_t>((std::numeric_limits<std::uint32_t>::max)()))
    {
        throw RuntimeError(RuntimeErrorCode::InvalidArgument,
                           "tensor rank exceeds ABI v1 range");
    }
    return static_cast<std::uint32_t>(shape.size());
}

std::filesystem::path canonical_artifact_path(const std::filesystem::path& path)
{
    if (path.empty())
    {
        throw RuntimeError(RuntimeErrorCode::InvalidArgument,
                           "model artifact path must not be empty");
    }
    std::error_code error;
    const std::filesystem::path result = std::filesystem::canonical(path, error);
    if (error)
    {
        throw RuntimeError(RuntimeErrorCode::FileIo,
                           "model artifact path cannot be resolved: " + path.string());
    }
    return result;
}

void require_request(const ModelLoadRequest& request)
{
    if (request.artifact_format.empty())
    {
        throw RuntimeError(RuntimeErrorCode::InvalidArgument,
                           "model artifact format must not be empty");
    }
    if (request.device_id.empty())
    {
        throw RuntimeError(RuntimeErrorCode::InvalidArgument,
                           "model device id must not be empty");
    }
}

} // namespace

struct BackendPlugin::State final
{
    std::shared_ptr<detail::DynamicLibrary> module;
    kf_backend_api_v1 api {};
    kf_backend_handle_v1 backend = nullptr;
    std::string backend_id;
    std::string backend_name;
    std::uint64_t backend_capabilities = 0U;

    ~State()
    {
        if (backend != nullptr && api.destroy_backend != nullptr)
        {
            api.destroy_backend(backend);
            backend = nullptr;
        }
    }

    [[nodiscard]] std::string diagnostic() const
    {
        if (api.format_last_error == nullptr)
        {
            return {};
        }
        std::uint64_t required = 0U;
        if (api.format_last_error(backend, nullptr, 0U, &required) != KF_STATUS_V1_OK || required == 0U ||
            required > static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)()))
        {
            return {};
        }
        std::string buffer(static_cast<std::size_t>(required), '\0');
        if (api.format_last_error(backend, buffer.data(), required, &required) != KF_STATUS_V1_OK)
        {
            return {};
        }
        if (!buffer.empty() && buffer.back() == '\0')
        {
            buffer.pop_back();
        }
        return buffer;
    }

    void check(kf_status_v1 status, const char* operation) const
    {
        if (status == KF_STATUS_V1_OK)
        {
            return;
        }
        std::string message = std::string("runtime backend ") + operation + " failed";
        const std::string detail = diagnostic();
        if (!detail.empty())
        {
            message += ": " + detail;
        }
        throw RuntimeError(RuntimeErrorCode::BackendFailure, std::move(message));
    }
};

struct ExecutableModel::State final
{
    std::shared_ptr<BackendPlugin::State> backend;
    kf_model_handle_v1 model = nullptr;

    ~State()
    {
        if (model != nullptr && backend && backend->api.destroy_model != nullptr)
        {
            backend->api.destroy_model(model);
            model = nullptr;
        }
    }
};

struct ExecutionContext::Impl final
{
    std::shared_ptr<ExecutableModel::State> model;
    kf_context_handle_v1 context = nullptr;

    ~Impl()
    {
        if (context != nullptr && model && model->backend && model->backend->api.destroy_context != nullptr)
        {
            model->backend->api.destroy_context(context);
            context = nullptr;
        }
    }
};

BackendPlugin::BackendPlugin(std::shared_ptr<State> state)
    : state_(std::move(state))
{
}

BackendPlugin::~BackendPlugin() = default;

std::shared_ptr<BackendPlugin> BackendPlugin::load(const std::filesystem::path& explicit_path)
{
    auto module = detail::DynamicLibrary::open(explicit_path);
    const auto query = reinterpret_cast<kfcore_backend_query_v1_fn>(
        module->symbol(KFCORE_BACKEND_QUERY_V1_SYMBOL));

    kf_backend_api_v1 api {};
    api.struct_size = sizeof(api);
    const kf_status_v1 query_status = query(KFCORE_BACKEND_ABI_V1_MAJOR,
                                            KFCORE_BACKEND_ABI_V1_MINOR,
                                            &api);
    if (query_status != KF_STATUS_V1_OK)
    {
        throw RuntimeError(RuntimeErrorCode::AbiMismatch,
                           "runtime backend rejected ABI v1 query: " + module->path().string());
    }
    if (api.struct_size < sizeof(kf_backend_api_v1) || api.abi_major != KFCORE_BACKEND_ABI_V1_MAJOR)
    {
        throw RuntimeError(RuntimeErrorCode::AbiMismatch,
                           "runtime backend returned an incompatible ABI table");
    }
    if (api.create_backend == nullptr || api.destroy_backend == nullptr ||
        api.get_backend_info == nullptr || api.get_device_count == nullptr ||
        api.get_device_info == nullptr || api.probe_artifact == nullptr ||
        api.load_model == nullptr || api.destroy_model == nullptr ||
        api.get_tensor_count == nullptr || api.get_tensor_info == nullptr ||
        api.create_context == nullptr || api.destroy_context == nullptr ||
        api.run == nullptr || api.format_last_error == nullptr)
    {
        throw RuntimeError(RuntimeErrorCode::AbiMismatch,
                           "runtime backend ABI table is incomplete");
    }

    auto state = std::make_shared<State>();
    state->module = std::move(module);
    state->api = api;

    const kf_backend_create_info_v1 create_info { sizeof(kf_backend_create_info_v1) };
    const kf_status_v1 create_status = state->api.create_backend(&create_info, &state->backend);
    if (create_status != KF_STATUS_V1_OK || state->backend == nullptr)
    {
        throw RuntimeError(RuntimeErrorCode::BackendFailure,
                           "runtime backend creation failed");
    }

    kf_backend_info_v1 info {};
    info.struct_size = sizeof(info);
    state->check(state->api.get_backend_info(state->backend, &info), "get_backend_info");
    state->backend_id = copy_string(info.backend_id);
    state->backend_name = copy_string(info.backend_name);
    state->backend_capabilities = info.capabilities;
    if (state->backend_id.empty())
    {
        throw RuntimeError(RuntimeErrorCode::AbiMismatch,
                           "runtime backend id must not be empty");
    }
    return std::shared_ptr<BackendPlugin>(new BackendPlugin(std::move(state)));
}

const std::string& BackendPlugin::id() const noexcept
{
    return state_->backend_id;
}

const std::string& BackendPlugin::name() const noexcept
{
    return state_->backend_name;
}

std::uint64_t BackendPlugin::capabilities() const noexcept
{
    return state_->backend_capabilities;
}

std::vector<BackendDevice> BackendPlugin::devices() const
{
    std::uint64_t count = 0U;
    state_->check(state_->api.get_device_count(state_->backend, &count), "get_device_count");
    if (count > static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)()))
    {
        throw RuntimeError(RuntimeErrorCode::BackendFailure,
                           "runtime backend device count exceeds host capacity");
    }
    std::vector<BackendDevice> result;
    result.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0U; index < count; ++index)
    {
        kf_device_info_v1 info {};
        info.struct_size = sizeof(info);
        state_->check(state_->api.get_device_info(state_->backend, index, &info), "get_device_info");
        BackendDevice device;
        device.id = copy_string(info.device_id);
        device.name = copy_string(info.device_name);
        device.capabilities = info.capabilities;
        if (device.id.empty())
        {
            throw RuntimeError(RuntimeErrorCode::BackendFailure,
                               "runtime backend returned an empty device id");
        }
        result.push_back(std::move(device));
    }
    return result;
}

bool BackendPlugin::can_load(const ModelLoadRequest& request) const
{
    require_request(request);
    const std::string path_utf8 = canonical_artifact_path(request.artifact_path).u8string();
    const kf_artifact_desc_v1 artifact {
        sizeof(kf_artifact_desc_v1),
        abi_string(path_utf8),
        abi_string(request.artifact_format),
        abi_string(request.artifact_flavor),
    };
    kf_artifact_probe_v1 probe {};
    probe.struct_size = sizeof(probe);
    state_->check(state_->api.probe_artifact(state_->backend, &artifact,
                                             abi_string(request.device_id), &probe),
                  "probe_artifact");
    return probe.support == KF_ARTIFACT_SUPPORT_V1_SUPPORTED;
}

std::shared_ptr<ExecutableModel> BackendPlugin::load_model(const ModelLoadRequest& request) const
{
    require_request(request);
    const std::string path_utf8 = canonical_artifact_path(request.artifact_path).u8string();
    const kf_model_load_info_v1 info {
        sizeof(kf_model_load_info_v1),
        {
            sizeof(kf_artifact_desc_v1),
            abi_string(path_utf8),
            abi_string(request.artifact_format),
            abi_string(request.artifact_flavor),
        },
        abi_string(request.device_id),
    };

    auto model_state = std::make_shared<ExecutableModel::State>();
    model_state->backend = state_;
    state_->check(state_->api.load_model(state_->backend, &info, &model_state->model), "load_model");
    if (model_state->model == nullptr)
    {
        throw RuntimeError(RuntimeErrorCode::BackendFailure,
                           "runtime backend returned a null model handle");
    }
    return std::shared_ptr<ExecutableModel>(new ExecutableModel(std::move(model_state)));
}

ExecutableModel::ExecutableModel(std::shared_ptr<State> state)
    : state_(std::move(state))
{
}

ExecutableModel::~ExecutableModel() = default;

std::vector<TensorDescriptor> ExecutableModel::tensors() const
{
    std::uint64_t count = 0U;
    state_->backend->check(state_->backend->api.get_tensor_count(state_->model, &count),
                           "get_tensor_count");
    if (count > static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)()))
    {
        throw RuntimeError(RuntimeErrorCode::BackendFailure,
                           "runtime backend tensor count exceeds host capacity");
    }

    std::vector<TensorDescriptor> result;
    result.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0U; index < count; ++index)
    {
        kf_tensor_desc_v1 info {};
        info.struct_size = sizeof(info);
        state_->backend->check(state_->backend->api.get_tensor_info(state_->model, index, &info),
                               "get_tensor_info");
        if (info.rank != 0U && info.dimensions == nullptr)
        {
            throw RuntimeError(RuntimeErrorCode::BackendFailure,
                               "runtime backend returned a null tensor shape");
        }
        TensorDescriptor tensor;
        tensor.name = copy_string(info.name);
        tensor.data_type = from_abi(info.data_type);
        if (info.rank != 0U)
        {
            tensor.shape.assign(info.dimensions, info.dimensions + info.rank);
        }
        tensor.is_input = info.io == KF_TENSOR_IO_V1_INPUT;
        result.push_back(std::move(tensor));
    }
    return result;
}

std::unique_ptr<ExecutionContext> ExecutableModel::create_context() const
{
    auto impl = std::make_unique<ExecutionContext::Impl>();
    impl->model = state_;
    const kf_context_create_info_v1 create_info { sizeof(kf_context_create_info_v1) };
    state_->backend->check(
        state_->backend->api.create_context(state_->model, &create_info, &impl->context),
        "create_context");
    if (impl->context == nullptr)
    {
        throw RuntimeError(RuntimeErrorCode::BackendFailure,
                           "runtime backend returned a null execution context");
    }
    return std::unique_ptr<ExecutionContext>(new ExecutionContext(std::move(impl)));
}

ExecutionContext::ExecutionContext(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

ExecutionContext::~ExecutionContext() = default;
ExecutionContext::ExecutionContext(ExecutionContext&&) noexcept = default;
ExecutionContext& ExecutionContext::operator=(ExecutionContext&&) noexcept = default;

void ExecutionContext::run(const std::vector<TensorView>& inputs,
                           const std::vector<MutableTensorView>& outputs)
{
    std::vector<kf_tensor_view_v1> abi_inputs;
    abi_inputs.reserve(inputs.size());
    for (const TensorView& input : inputs)
    {
        abi_inputs.push_back({
            sizeof(kf_tensor_view_v1), abi_string(input.name), to_abi(input.data_type),
            input.shape.data(), checked_rank(input.shape), input.data,
            static_cast<std::uint64_t>(input.byte_size), to_abi(input.memory_kind),
            abi_string(input.device_id),
        });
    }

    std::vector<kf_mutable_tensor_view_v1> abi_outputs;
    abi_outputs.reserve(outputs.size());
    for (const MutableTensorView& output : outputs)
    {
        abi_outputs.push_back({
            sizeof(kf_mutable_tensor_view_v1), abi_string(output.name), to_abi(output.data_type),
            output.shape.data(), checked_rank(output.shape), output.data,
            static_cast<std::uint64_t>(output.byte_size), to_abi(output.memory_kind),
            abi_string(output.device_id),
        });
    }

    impl_->model->backend->check(
        impl_->model->backend->api.run(
            impl_->context,
            abi_inputs.empty() ? nullptr : abi_inputs.data(),
            static_cast<std::uint64_t>(abi_inputs.size()),
            abi_outputs.empty() ? nullptr : abi_outputs.data(),
            static_cast<std::uint64_t>(abi_outputs.size())),
        "run");
}

std::shared_ptr<BackendPlugin> BackendRegistry::load(const std::filesystem::path& explicit_path)
{
    std::shared_ptr<BackendPlugin> plugin = BackendPlugin::load(explicit_path);
    const auto [iterator, inserted] = backends_.emplace(plugin->id(), plugin);
    if (!inserted)
    {
        throw RuntimeError(RuntimeErrorCode::DuplicateBackend,
                           "runtime backend is already loaded: " + plugin->id());
    }
    return iterator->second;
}

std::shared_ptr<BackendPlugin> BackendRegistry::find(std::string_view backend_id) const
{
    const auto iterator = backends_.find(std::string(backend_id));
    if (iterator == backends_.end())
    {
        throw RuntimeError(RuntimeErrorCode::NotFound,
                           "runtime backend is not loaded: " + std::string(backend_id));
    }
    return iterator->second;
}

std::vector<std::string> BackendRegistry::backend_ids() const
{
    std::vector<std::string> result;
    result.reserve(backends_.size());
    for (const auto& entry : backends_)
    {
        result.push_back(entry.first);
    }
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace kfcore::runtime
