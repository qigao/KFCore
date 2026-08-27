#pragma once

#include "kfcore/tensorrt/error.hpp"
#include "kfcore/tensorrt/types.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace kfcore::tensorrt
{

class Executor;

class Engine final
{
public:
    ~Engine();
    Engine(const Engine& other);
    Engine& operator=(const Engine& other);

    static std::shared_ptr<const Engine> load(const std::filesystem::path& engine_path,
                                              const EngineOptions&         options = {});

    std::unique_ptr<Executor>            create_executor() const;
    const std::vector<TensorDescriptor>& tensors() const noexcept;

private:
    friend class Executor;
    struct Impl;

    explicit Engine(std::shared_ptr<const Impl> impl);

    // The heap anchor can be relinquished without decrementing the shared count when CUDA device
    // selection fails during noexcept destruction.
    std::unique_ptr<std::shared_ptr<const Impl>> impl_;
};

class Executor final
{
public:
    ~Executor();

    Executor(const Executor&)            = delete;
    Executor& operator=(const Executor&) = delete;

    void run(const std::vector<TensorView>& inputs, const std::vector<MutableTensorView>& outputs);

    std::vector<HostTensor> run_dynamic(
        const std::vector<TensorView>& inputs,
        const std::vector<DynamicOutputRequest>& outputs);

private:
    friend class Engine;
    struct Impl;

    explicit Executor(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::tensorrt
