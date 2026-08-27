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

    static std::shared_ptr<const Engine> load(const std::filesystem::path& engine_path,
                                              const EngineOptions&         options = {});

    std::unique_ptr<Executor>            create_executor() const;
    const std::vector<TensorDescriptor>& tensors() const noexcept;

private:
    struct Impl;

    explicit Engine(std::shared_ptr<const Impl> impl);

    std::shared_ptr<const Impl> impl_;
};

class Executor final
{
public:
    ~Executor();

    Executor(const Executor&)            = delete;
    Executor& operator=(const Executor&) = delete;

    void run(const std::vector<TensorView>& inputs, const std::vector<MutableTensorView>& outputs);

private:
    friend class Engine;
    struct Impl;

    explicit Executor(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::tensorrt
