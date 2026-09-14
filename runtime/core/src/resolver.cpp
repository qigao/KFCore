#include "kfcore/runtime/resolver.hpp"

#include "kfcore/runtime/error.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <sstream>
#include <string_view>
#include <utility>

namespace kfcore::runtime
{
namespace
{

bool device_constraint_matches(std::string_view constraint, std::string_view device_id)
{
    if (constraint == "any")
    {
        return true;
    }
    if (constraint == "cpu")
    {
        return device_id == "cpu" || device_id.rfind("cpu:", 0U) == 0U;
    }
    if (constraint == "cuda")
    {
        return device_id == "cuda" || device_id.rfind("cuda:", 0U) == 0U;
    }
    return constraint == device_id;
}

std::shared_ptr<BackendPlugin> find_policy_backend(const BackendRegistry& registry,
                                                   std::string_view backend_id)
{
    try
    {
        return registry.find(backend_id);
    }
    catch (const RuntimeError& error)
    {
        if (error.code() == RuntimeErrorCode::NotFound)
        {
            return {};
        }
        throw;
    }
}

const BackendDevice* find_device(const std::vector<BackendDevice>& devices,
                                 std::string_view device_id)
{
    const auto iterator = std::find_if(devices.begin(), devices.end(),
                                       [&](const BackendDevice& device) {
                                           return device.id == device_id;
                                       });
    return iterator == devices.end() ? nullptr : &*iterator;
}

std::pair<std::uint32_t, std::uint32_t>
parse_compute_capability(std::string_view value, std::string_view artifact_id)
{
    const std::size_t dot = value.find('.');
    if (dot == std::string_view::npos || dot == 0U || dot + 1U >= value.size() ||
        value.find('.', dot + 1U) != std::string_view::npos)
    {
        throw RuntimeError(RuntimeErrorCode::InvalidModelPackage,
                           "artifact '" + std::string(artifact_id) +
                               "' compute_capability must use major.minor form");
    }

    std::uint32_t major = 0U;
    std::uint32_t minor = 0U;
    const char* begin = value.data();
    const char* end = value.data() + value.size();
    const auto major_result = std::from_chars(begin, begin + dot, major);
    const auto minor_result = std::from_chars(begin + dot + 1U, end, minor);
    if (major_result.ec != std::errc{} || major_result.ptr != begin + dot ||
        minor_result.ec != std::errc{} || minor_result.ptr != end || major == 0U)
    {
        throw RuntimeError(RuntimeErrorCode::InvalidModelPackage,
                           "artifact '" + std::string(artifact_id) +
                               "' compute_capability is invalid");
    }
    return {major, minor};
}

RuntimeVersion parse_runtime_version(std::string_view value,
                                     std::string_view artifact_id)
{
    RuntimeVersion result;
    std::array<std::uint32_t*, 4U> fields{
        &result.major, &result.minor, &result.patch, &result.build};
    std::size_t begin = 0U;
    for (std::size_t index = 0U; index < fields.size(); ++index)
    {
        const std::size_t end = index + 1U == fields.size()
                                    ? value.size()
                                    : value.find('.', begin);
        if (end == std::string_view::npos || end == begin)
        {
            throw RuntimeError(RuntimeErrorCode::InvalidModelPackage,
                               "artifact '" + std::string(artifact_id) +
                                   "' runtime_version must use major.minor.patch.build form");
        }
        const char* first = value.data() + begin;
        const char* last = value.data() + end;
        const auto parsed = std::from_chars(first, last, *fields[index]);
        if (parsed.ec != std::errc{} || parsed.ptr != last)
        {
            throw RuntimeError(RuntimeErrorCode::InvalidModelPackage,
                               "artifact '" + std::string(artifact_id) +
                                   "' runtime_version is invalid");
        }
        begin = end + 1U;
    }
    if (begin != value.size() + 1U || result.major == 0U)
    {
        throw RuntimeError(RuntimeErrorCode::InvalidModelPackage,
                           "artifact '" + std::string(artifact_id) +
                               "' runtime_version must use major.minor.patch.build form");
    }
    return result;
}

std::string_view current_platform() noexcept
{
#if defined(_WIN32)
#  if defined(_M_X64) || defined(__x86_64__)
    return "windows-x86_64";
#  elif defined(_M_ARM64) || defined(__aarch64__)
    return "windows-aarch64";
#  else
    return "windows-unknown";
#  endif
#elif defined(__linux__)
#  if defined(__x86_64__)
    return "linux-x86_64";
#  elif defined(__aarch64__)
    return "linux-aarch64";
#  else
    return "linux-unknown";
#  endif
#else
    return "unsupported-unknown";
#endif
}

bool runtime_version_matches(const RuntimeVersion& required,
                             const RuntimeVersion& actual) noexcept
{
    return required.major == actual.major &&
           required.minor == actual.minor &&
           required.patch == actual.patch &&
           required.build == actual.build;
}

bool artifact_runtime_matches(const ModelArtifact& artifact,
                              const BackendPlugin& backend,
                              const BackendDevice& device)
{
    if (artifact.format != "tensorrt-engine")
    {
        return true;
    }

    if (artifact.platform != current_platform())
    {
        return false;
    }

    const RuntimeVersion required =
        parse_runtime_version(artifact.runtime_version, artifact.id);
    if (!runtime_version_matches(required, backend.execution_runtime_version()))
    {
        return false;
    }

    const auto required_cc =
        parse_compute_capability(artifact.compute_capability, artifact.id);
    if (device.compute_capability_major == 0U ||
        device.compute_capability_major != required_cc.first ||
        device.compute_capability_minor != required_cc.second)
    {
        return false;
    }

    if (artifact.hardware_compatibility == "same-compute-capability")
    {
        return true;
    }
    if (artifact.hardware_compatibility == "exact-device")
    {
        return !artifact.device_name.empty() && artifact.device_name == device.name;
    }
    return false;
}

std::string describe_policy(const ExecutionPolicy& policy)
{
    std::ostringstream stream;
    bool first = true;
    for (const auto& preference : policy.preferences())
    {
        if (!first)
        {
            stream << ", ";
        }
        first = false;
        stream << preference.backend_id << '/' << preference.device_id;
    }
    return stream.str();
}

} // namespace

ExecutionPolicy::ExecutionPolicy(std::vector<ExecutionPreference> preferences)
    : preferences_(std::move(preferences))
{
    if (preferences_.empty())
    {
        throw RuntimeError(RuntimeErrorCode::InvalidArgument,
                           "execution policy must contain at least one preference");
    }
    for (const auto& preference : preferences_)
    {
        if (preference.backend_id.empty() || preference.device_id.empty())
        {
            throw RuntimeError(RuntimeErrorCode::InvalidArgument,
                               "execution policy backend and device must be explicit");
        }
    }
}

ExecutionPolicy ExecutionPolicy::exact(std::string backend_id, std::string device_id)
{
    return ExecutionPolicy({ExecutionPreference{std::move(backend_id), std::move(device_id)}});
}

ExecutionPolicy ExecutionPolicy::ordered(std::vector<ExecutionPreference> preferences)
{
    return ExecutionPolicy(std::move(preferences));
}

const std::vector<ExecutionPreference>& ExecutionPolicy::preferences() const noexcept
{
    return preferences_;
}

ResolvedModel ModelResolver::load(const ModelPackage& package,
                                  const BackendRegistry& registry,
                                  const ExecutionPolicy& policy)
{
    for (const auto& preference : policy.preferences())
    {
        auto backend = find_policy_backend(registry, preference.backend_id);
        if (!backend)
        {
            continue;
        }
        const std::vector<BackendDevice> devices = backend->devices();
        const BackendDevice* selected_device = find_device(devices, preference.device_id);
        if (selected_device == nullptr)
        {
            continue;
        }

        for (const auto& artifact : package.artifacts())
        {
            if (artifact.backend != preference.backend_id ||
                !device_constraint_matches(artifact.device, preference.device_id) ||
                !artifact_runtime_matches(artifact, *backend, *selected_device))
            {
                continue;
            }

            verify_model_artifact(package, artifact);
            ModelLoadRequest request;
            request.artifact_path = package.artifact_path(artifact);
            request.artifact_format = artifact.format;
            request.artifact_flavor = artifact.flavor;
            request.device_id = preference.device_id;

            if (!backend->can_load(request))
            {
                continue;
            }

            ResolvedModel result;
            result.route.backend_id = preference.backend_id;
            result.route.device_id = preference.device_id;
            result.route.artifact = artifact;
            result.model = backend->load_model(request);
            if (!result.model)
            {
                throw RuntimeError(RuntimeErrorCode::BackendFailure,
                                   "backend returned no executable model for package '" +
                                       package.id() + "'");
            }
            return result;
        }
    }

    throw RuntimeError(RuntimeErrorCode::NoCompatibleExecution,
                       "no compatible execution route for model package '" + package.id() +
                           "' using policy [" + describe_policy(policy) + "]");
}

} // namespace kfcore::runtime
