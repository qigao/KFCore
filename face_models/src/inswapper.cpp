#include "kfcore/face_models/tensorrt.hpp"

#include "contracts.hpp"

#include <algorithm>
#include <atomic>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kfcore::face_models
{
namespace
{

constexpr char kModelName[] = "InSwapper";

} // namespace

struct TensorRtInSwapper::Impl final
{
    Impl(std::shared_ptr<const kfcore::tensorrt::Engine> engine_in,
         std::unique_ptr<kfcore::tensorrt::Executor> executor_in,
         detail::InSwapperContract contract_in)
        : engine(std::move(engine_in))
        , executor(std::move(executor_in))
        , contract(std::move(contract_in))
        , output(contract.output_float_capacity)
        , inputs(2)
        , outputs(1)
    {
        inputs[0] = { contract.target_input_name,
                      kfcore::tensorrt::DataType::Float32,
                      { 1, kFaceModelInputChannels, kInSwapperInputExtent,
                        kInSwapperInputExtent },
                      nullptr,
                      0,
                      kfcore::tensorrt::MemoryKind::Host };
        inputs[1] = { contract.source_input_name,
                      kfcore::tensorrt::DataType::Float32,
                      { 1, static_cast<std::int64_t>(kInSwapperEmbeddingLength) },
                      nullptr,
                      0,
                      kfcore::tensorrt::MemoryKind::Host };
        outputs[0] = { contract.output_name,
                       kfcore::tensorrt::DataType::Float32,
                       { 1, kFaceModelInputChannels, kInSwapperInputExtent,
                         kInSwapperInputExtent },
                       output.data(),
                       output.size() * sizeof(float),
                       kfcore::tensorrt::MemoryKind::Host };
    }

    std::shared_ptr<const kfcore::tensorrt::Engine> engine;
    std::unique_ptr<kfcore::tensorrt::Executor> executor;
    detail::InSwapperContract contract;
    std::vector<float> output;
    std::vector<kfcore::tensorrt::TensorView> inputs;
    std::vector<kfcore::tensorrt::MutableTensorView> outputs;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

TensorRtInSwapper::TensorRtInSwapper(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

TensorRtInSwapper::~TensorRtInSwapper() = default;

std::unique_ptr<TensorRtInSwapper>
TensorRtInSwapper::load(const std::filesystem::path& engine_path,
                        const InSwapperOptions& options)
{
    try
    {
        detail::validate_inswapper_options(options);
        std::shared_ptr<const kfcore::tensorrt::Engine> engine =
            kfcore::tensorrt::Engine::load(engine_path, options.engine);
        detail::InSwapperContract contract =
            detail::validate_inswapper_contract(engine->tensors(), options);
        auto executor = engine->create_executor();
        auto impl = std::make_unique<Impl>(engine, std::move(executor), std::move(contract));
        return std::unique_ptr<TensorRtInSwapper>(new TensorRtInSwapper(std::move(impl)));
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const kfcore::tensorrt::TensorRtError& error)
    {
        detail::rethrow_tensorrt(error, kModelName, "load");
    }
    catch (const std::bad_alloc&)
    {
        detail::throw_allocation_failure(kModelName, "load");
    }
    catch (const std::length_error&)
    {
        detail::throw_capacity_failure(kModelName, "load");
    }
}

InSwapperResult TensorRtInSwapper::infer(
    const kfcore::tensorrt::TensorView& prepared_target,
    const kfcore::tensorrt::TensorView& projected_source)
{
    try
    {
        detail::AdapterCallGuard guard(impl_->in_use, kModelName);
        detail::validate_prepared_input(
            prepared_target, impl_->contract.target_input_name, impl_->contract.batch,
            { kFaceModelInputChannels, kInSwapperInputExtent, kInSwapperInputExtent },
            kModelName);
        detail::validate_prepared_vector_input(
            projected_source, impl_->contract.source_input_name, impl_->contract.batch,
            { static_cast<std::int64_t>(kInSwapperEmbeddingLength) }, kModelName);
        if (prepared_target.shape[0] != projected_source.shape[0])
        {
            throw FaceModelError(FaceModelErrorCode::InvalidTensorView,
                                 "InSwapper input validation stage: target and source batch "
                                 "dimensions must match");
        }

        detail::BorrowedInputGuard target_guard(impl_->inputs[0], prepared_target);
        detail::BorrowedInputGuard source_guard(impl_->inputs[1], projected_source);
        impl_->executor->run(impl_->inputs, impl_->outputs);

        InSwapperResult result;
        result.values.assign(impl_->output.begin(), impl_->output.end());
        return result;
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const kfcore::tensorrt::TensorRtError& error)
    {
        detail::rethrow_tensorrt(error, kModelName, "inference");
    }
    catch (const std::bad_alloc&)
    {
        detail::throw_allocation_failure(kModelName, "inference result");
    }
    catch (const std::length_error&)
    {
        detail::throw_capacity_failure(kModelName, "inference result");
    }
}

} // namespace kfcore::face_models
