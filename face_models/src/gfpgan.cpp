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

constexpr char kModelName[] = "GFPGAN";

} // namespace

struct TensorRtGfpGan::Impl final
{
    Impl(std::shared_ptr<const kfcore::tensorrt::Engine> engine_in,
         std::unique_ptr<kfcore::tensorrt::Executor> executor_in,
         detail::SingleOutputContract contract_in)
        : engine(std::move(engine_in))
        , executor(std::move(executor_in))
        , contract(std::move(contract_in))
        , output(contract.output_float_capacity)
        , inputs(1)
        , outputs(1)
    {
        inputs[0] = { contract.input_name,
                      kfcore::tensorrt::DataType::Float32,
                      { 1, kFaceModelInputChannels, kGfpGanInputExtent, kGfpGanInputExtent },
                      nullptr,
                      0,
                      kfcore::tensorrt::MemoryKind::Host };
        outputs[0] = { contract.output_name,
                       kfcore::tensorrt::DataType::Float32,
                       { 1, kFaceModelInputChannels, kGfpGanInputExtent, kGfpGanInputExtent },
                       output.data(),
                       output.size() * sizeof(float),
                       kfcore::tensorrt::MemoryKind::Host };
    }

    std::shared_ptr<const kfcore::tensorrt::Engine> engine;
    std::unique_ptr<kfcore::tensorrt::Executor> executor;
    detail::SingleOutputContract contract;
    std::vector<float> output;
    std::vector<kfcore::tensorrt::TensorView> inputs;
    std::vector<kfcore::tensorrt::MutableTensorView> outputs;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

TensorRtGfpGan::TensorRtGfpGan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

TensorRtGfpGan::~TensorRtGfpGan() = default;

std::unique_ptr<TensorRtGfpGan>
TensorRtGfpGan::load(const std::filesystem::path& engine_path, const GfpGanOptions& options)
{
    try
    {
        detail::validate_gfpgan_options(options);
        std::shared_ptr<const kfcore::tensorrt::Engine> engine =
            kfcore::tensorrt::Engine::load(engine_path, options.engine);
        detail::SingleOutputContract contract =
            detail::validate_gfpgan_contract(engine->tensors(), options);
        auto executor = engine->create_executor();
        auto impl = std::make_unique<Impl>(engine, std::move(executor), std::move(contract));
        return std::unique_ptr<TensorRtGfpGan>(new TensorRtGfpGan(std::move(impl)));
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

GfpGanResult TensorRtGfpGan::infer(const kfcore::tensorrt::TensorView& prepared_input)
{
    try
    {
        detail::AdapterCallGuard guard(impl_->in_use, kModelName);
        detail::validate_prepared_input(
            prepared_input, impl_->contract.input_name, impl_->contract.batch,
            { kFaceModelInputChannels, kGfpGanInputExtent, kGfpGanInputExtent }, kModelName);
        detail::BorrowedInputGuard input_guard(impl_->inputs[0], prepared_input);
        impl_->executor->run(impl_->inputs, impl_->outputs);

        GfpGanResult result;
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
