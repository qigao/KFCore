#pragma once

#include "kfcore/runtime/resolver.hpp"

#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

namespace kfcore::runtime
{

class Runtime final
{
public:
    Runtime() = default;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&&) noexcept = default;
    Runtime& operator=(Runtime&&) noexcept = default;

    [[nodiscard]] std::shared_ptr<BackendPlugin>
    load_backend(const std::filesystem::path& explicit_path);

    [[nodiscard]] std::vector<std::shared_ptr<BackendPlugin>>
    load_backends_from(const std::filesystem::path& controlled_directory);

    [[nodiscard]] std::shared_ptr<BackendPlugin> backend(std::string_view backend_id) const;
    [[nodiscard]] std::vector<std::string> backend_ids() const;

    [[nodiscard]] ResolvedModel load_model(const ModelPackage& package,
                                           const ExecutionPolicy& policy) const;

    [[nodiscard]] BackendRegistry& backends() noexcept;
    [[nodiscard]] const BackendRegistry& backends() const noexcept;

private:
    BackendRegistry backends_;
};

} // namespace kfcore::runtime
