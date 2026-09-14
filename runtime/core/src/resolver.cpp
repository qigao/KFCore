#include "kfcore/runtime/resolver.hpp"

#include "kfcore/runtime/error.hpp"

#include <algorithm>
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

bool artifact_runtime_matches(const ModelArtifact& artifact,
                              const BackendPlugin& backend,
                              const BackendDevice& device)
{
    if (artifact.runtime_major != 0U)
    {
        const RuntimeVersion& runtime = backend.execution_runtime_version();
        if (runtime.major == 0U || runtime.major != artifact.runtime_major)
        {
            return false;
        }
    }

    if (!artifact.compute_capability.empty())
    {
        const auto required = parse_compute_capability(artifact.compute_capability,
                                                       artifact.id);
        if (device.compute_capability_major == 0U ||
            device.compute_capability_major != required.first ||
            device.compute_capability_minor != required.second)
        {
            return false;
        }
    }
    return true;
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
