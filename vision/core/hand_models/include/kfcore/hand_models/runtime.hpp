#pragma once

#include "kfcore/hand_models/core.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/resolver.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <cctype>
#include <cstddef>
#include <memory>

namespace kfcore::hand_models
{

struct HandRuntimeOptions
{
    std::size_t max_source_bytes = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes = 64U * 1024U * 1024U;
    std::size_t max_output_bytes = 32U * 1024U * 1024U;
    std::size_t max_palm_candidates = 2016U;
    std::size_t max_hands = 8U;
    float palm_score_threshold = 0.52F;
    float hand_score_threshold = 0.50F;
};

class HandBackend final : public HandInferenceBackend
{
public:
    ~HandBackend() override;

    HandBackend(const HandBackend&) = delete;
    HandBackend& operator=(const HandBackend&) = delete;

    [[nodiscard]] static std::unique_ptr<HandBackend> load(
        runtime::Runtime& runtime,
        const runtime::ModelPackage& palm_package,
        const runtime::ExecutionPolicy& palm_policy,
        const runtime::ModelPackage& landmark_package,
        const runtime::ExecutionPolicy& landmark_policy,
        const runtime::ModelPackage& classifier_package,
        const runtime::ExecutionPolicy& classifier_policy,
        const HandRuntimeOptions& options = {});

    HandFrame infer(const image::ImageView& image) override;

    [[nodiscard]] const runtime::ExecutionRoute& palm_execution_route() const noexcept;
    [[nodiscard]] const runtime::ExecutionRoute& landmark_execution_route() const noexcept;
    [[nodiscard]] const runtime::ExecutionRoute& classifier_execution_route() const noexcept;

private:
    struct Impl;
    explicit HandBackend(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::hand_models
