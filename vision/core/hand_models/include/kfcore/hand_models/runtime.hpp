#pragma once

#include "kfcore/hand_models/error.hpp"
#include "kfcore/hand_models/types.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/resolver.hpp"
#include "kfcore/runtime/runtime.hpp"

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

class HandDetector final
{
public:
    ~HandDetector();

    HandDetector(const HandDetector&) = delete;
    HandDetector& operator=(const HandDetector&) = delete;

    [[nodiscard]] static std::unique_ptr<HandDetector> load(
        runtime::Runtime& runtime,
        const runtime::ModelPackage& palm_package,
        const runtime::ExecutionPolicy& palm_policy,
        const runtime::ModelPackage& landmark_package,
        const runtime::ExecutionPolicy& landmark_policy,
        const runtime::ModelPackage& classifier_package,
        const runtime::ExecutionPolicy& classifier_policy,
        const HandRuntimeOptions& options = {});

    [[nodiscard]] HandFrame infer(const image::ImageView& image);

    /** Explicit two-model mode. Does not load or execute a gesture classifier.
     * Results retain Gesture::Unknown and classifier timing is zero. This is
     * never selected as recovery from a failed three-model load.
     */
    [[nodiscard]] static std::unique_ptr<HandDetector> load_landmarks(
        runtime::Runtime& runtime,
        const runtime::ModelPackage& palm_package,
        const runtime::ExecutionPolicy& palm_policy,
        const runtime::ModelPackage& landmark_package,
        const runtime::ExecutionPolicy& landmark_policy,
        const HandRuntimeOptions& options = {});

    [[nodiscard]] const runtime::ExecutionRoute& palm_execution_route() const noexcept;
    [[nodiscard]] const runtime::ExecutionRoute& landmark_execution_route() const noexcept;
    [[nodiscard]] const runtime::ExecutionRoute& classifier_execution_route() const noexcept;

private:
    struct Impl;
    explicit HandDetector(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::hand_models
