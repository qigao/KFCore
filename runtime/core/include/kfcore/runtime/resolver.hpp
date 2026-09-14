#pragma once

#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/plugin.hpp"

#include <memory>
#include <string>
#include <vector>

namespace kfcore::runtime
{

struct ExecutionPreference
{
    std::string backend_id;
    std::string device_id;
};

class ExecutionPolicy final
{
public:
    static ExecutionPolicy exact(std::string backend_id, std::string device_id);
    static ExecutionPolicy ordered(std::vector<ExecutionPreference> preferences);

    [[nodiscard]] const std::vector<ExecutionPreference>& preferences() const noexcept;

private:
    explicit ExecutionPolicy(std::vector<ExecutionPreference> preferences);
    std::vector<ExecutionPreference> preferences_;
};

struct ExecutionRoute
{
    std::string backend_id;
    std::string device_id;
    ModelArtifact artifact;
};

struct ResolvedModel
{
    ExecutionRoute route;
    std::shared_ptr<ExecutableModel> model;
};

class ModelResolver final
{
public:
    [[nodiscard]] static ResolvedModel load(const ModelPackage& package,
                                            const BackendRegistry& registry,
                                            const ExecutionPolicy& policy);
};

} // namespace kfcore::runtime
