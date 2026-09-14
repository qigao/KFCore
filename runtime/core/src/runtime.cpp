#include "kfcore/runtime/runtime.hpp"

#include "kfcore/runtime/error.hpp"

#include <algorithm>
#include <system_error>

namespace kfcore::runtime
{
namespace
{

bool unversioned_backend_stem(const std::filesystem::path& path,
                              std::string_view required_prefix)
{
    const std::string stem = path.stem().string();
    return stem.rfind(required_prefix, 0U) == 0U &&
           stem.find('.', required_prefix.size()) == std::string::npos;
}

bool backend_library_name(const std::filesystem::path& path)
{
#if defined(_WIN32)
    return path.extension() == ".dll" &&
           unversioned_backend_stem(path, "kfcore_backend_");
#elif defined(__APPLE__)
    return path.extension() == ".dylib" &&
           unversioned_backend_stem(path, "libkfcore_backend_");
#else
    return path.extension() == ".so" &&
           unversioned_backend_stem(path, "libkfcore_backend_");
#endif
}

} // namespace

std::shared_ptr<BackendPlugin> Runtime::load_backend(const std::filesystem::path& explicit_path)
{
    return backends_.load(explicit_path);
}

std::vector<std::shared_ptr<BackendPlugin>>
Runtime::load_backends_from(const std::filesystem::path& controlled_directory)
{
    if (controlled_directory.empty())
    {
        throw RuntimeError(RuntimeErrorCode::InvalidArgument,
                           "runtime plugin directory must not be empty");
    }

    std::error_code error;
    const std::filesystem::path directory =
        std::filesystem::canonical(controlled_directory, error);
    if (error || !std::filesystem::is_directory(directory, error) || error)
    {
        throw RuntimeError(RuntimeErrorCode::FileIo,
                           "runtime plugin directory is invalid: " +
                               controlled_directory.string());
    }

    std::vector<std::filesystem::path> candidates;
    std::filesystem::directory_iterator iterator(directory, error);
    if (error)
    {
        throw RuntimeError(RuntimeErrorCode::FileIo,
                           "runtime plugin directory cannot be enumerated: " +
                               directory.string());
    }
    for (const auto& entry : iterator)
    {
        const bool regular = entry.is_regular_file(error);
        if (error)
        {
            throw RuntimeError(RuntimeErrorCode::FileIo,
                               "runtime plugin entry cannot be inspected: " +
                                   entry.path().string());
        }
        if (regular && backend_library_name(entry.path()))
        {
            candidates.push_back(entry.path());
        }
    }
    std::sort(candidates.begin(), candidates.end());

    std::vector<std::shared_ptr<BackendPlugin>> loaded;
    loaded.reserve(candidates.size());
    for (const auto& candidate : candidates)
    {
        loaded.push_back(load_backend(candidate));
    }
    return loaded;
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
