#pragma once

#include "kfcore/runtime/types.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace kfcore::runtime
{

struct ModelLoadRequest
{
    std::filesystem::path artifact_path;
    std::string artifact_format;
    std::string artifact_flavor;
    std::string device_id;
};

class ExecutionContext;

class ExecutableModel final
{
public:
    ~ExecutableModel();

    ExecutableModel(const ExecutableModel&) = delete;
    ExecutableModel& operator=(const ExecutableModel&) = delete;

    [[nodiscard]] std::vector<TensorDescriptor> tensors() const;
    [[nodiscard]] std::unique_ptr<ExecutionContext> create_context() const;

private:
    friend class BackendPlugin;
    friend class ExecutionContext;
    struct State;

    explicit ExecutableModel(std::shared_ptr<State> state);
    std::shared_ptr<State> state_;
};

class ExecutionContext final
{
public:
    ~ExecutionContext();

    ExecutionContext(ExecutionContext&&) noexcept;
    ExecutionContext& operator=(ExecutionContext&&) noexcept;
    ExecutionContext(const ExecutionContext&) = delete;
    ExecutionContext& operator=(const ExecutionContext&) = delete;

    void run(const std::vector<TensorView>& inputs,
             const std::vector<MutableTensorView>& outputs);

private:
    friend class ExecutableModel;
    struct Impl;

    explicit ExecutionContext(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class BackendPlugin final : public std::enable_shared_from_this<BackendPlugin>
{
public:
    static std::shared_ptr<BackendPlugin> load(const std::filesystem::path& explicit_path);

    ~BackendPlugin();
    BackendPlugin(const BackendPlugin&) = delete;
    BackendPlugin& operator=(const BackendPlugin&) = delete;

    [[nodiscard]] const std::string& id() const noexcept;
    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] std::uint64_t capabilities() const noexcept;
    [[nodiscard]] std::vector<BackendDevice> devices() const;
    [[nodiscard]] bool can_load(const ModelLoadRequest& request) const;
    [[nodiscard]] std::shared_ptr<ExecutableModel> load_model(const ModelLoadRequest& request) const;

private:
    friend class ExecutableModel;
    struct State;

    explicit BackendPlugin(std::shared_ptr<State> state);
    std::shared_ptr<State> state_;
};

class BackendRegistry final
{
public:
    [[nodiscard]] std::shared_ptr<BackendPlugin> load(const std::filesystem::path& explicit_path);
    [[nodiscard]] std::shared_ptr<BackendPlugin> find(std::string_view backend_id) const;
    [[nodiscard]] std::vector<std::string> backend_ids() const;

private:
    std::unordered_map<std::string, std::shared_ptr<BackendPlugin>> backends_;
};

} // namespace kfcore::runtime
