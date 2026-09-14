#include "kfcore/runtime/resolver.hpp"

#include "kfcore/runtime/error.hpp"

#include <algorithm>
#include <sstream>
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

bool backend_has_device(const BackendPlugin& backend, std::string_view device_id)
{
    const auto devices = backend.devices();
    return std::any_of(devices.begin(), devices.end(), [&](const BackendDevice& device) {
        return device.id == device_id;
    });
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
        if (!backend || !backend_has_device(*backend, preference.device_id))
        {
            continue;
        }

        for (const auto& artifact : package.artifacts())
        {
            if (artifact.backend != preference.backend_id ||
                !device_constraint_matches(artifact.device, preference.device_id))
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
