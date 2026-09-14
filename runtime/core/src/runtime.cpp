#include "kfcore/runtime/runtime.hpp"

namespace kfcore::runtime
{

std::shared_ptr<BackendPlugin> Runtime::load_backend(const std::filesystem::path& explicit_path)
{
    return backends_.load(explicit_path);
}

std::shared_ptr<BackendPlugin> Runtime::backend(std::string_view backend_id) const
{
    return backends_.find(backend_id);
}

std::vector<std::string> Runtime::backend_ids() const
{
    return backends_.backend_ids();
}

ResolvedModel Runtime::load_model(const ModelPackage& package,
                                  const ExecutionPolicy& policy) const
{
    return ModelResolver::load(package, backends_, policy);
}

BackendRegistry& Runtime::backends() noexcept
{
    return backends_;
}

const BackendRegistry& Runtime::backends() const noexcept
{
    return backends_;
}

} // namespace kfcore::runtime
