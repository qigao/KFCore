#include "kfcore/face_models/tensorrt.hpp"

#include "contracts.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kfcore::face_models
{
namespace
{

    constexpr char kModelName[] = "ArcFace";

} // namespace

struct TensorRtArcFace::Impl final
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
                      { 1, kFaceModelInputChannels, kArcFaceInputExtent, kArcFaceInputExtent },
                      nullptr,
                      0,
                      kfcore::tensorrt::MemoryKind::Host };
        outputs[0] = { contract.output_name,
                       kfcore::tensorrt::DataType::Float32,
                       { 1, static_cast<std::int64_t>(kArcFaceEmbeddingLength) },
                       output.data(),
                       output.size() * sizeof(float),
                       kfcore::tensorrt::MemoryKind::Host };
    }

    std::shared_ptr<const kfcore::tensorrt::Engine> engine;
    std::unique_ptr<kfcore::tensorrt::Executor>     executor;
    detail::SingleOutputContract                    contract;
    std::vector<float>                              output;
    std::vector<kfcore::tensorrt::TensorView>        inputs;
    std::vector<kfcore::tensorrt::MutableTensorView> outputs;
    std::atomic_flag                                 in_use = ATOMIC_FLAG_INIT;
};

TensorRtArcFace::TensorRtArcFace(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

TensorRtArcFace::~TensorRtArcFace() = default;

std::unique_ptr<TensorRtArcFace>
TensorRtArcFace::load(const std::filesystem::path& engine_path, const ArcFaceOptions& options)
{
    try
    {
        detail::validate_arcface_options(options);
        std::shared_ptr<const kfcore::tensorrt::Engine> engine =
            kfcore::tensorrt::Engine::load(engine_path, options.engine);
        detail::SingleOutputContract contract =
            detail::validate_arcface_contract(engine->tensors(), options);
        std::unique_ptr<kfcore::tensorrt::Executor> executor = engine->create_executor();
        auto impl = std::make_unique<Impl>(engine, std::move(executor), std::move(contract));
        return std::unique_ptr<TensorRtArcFace>(new TensorRtArcFace(std::move(impl)));
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

std::vector<ArcFaceResult>
TensorRtArcFace::infer(const kfcore::tensorrt::TensorView& prepared_input)
{
    try
    {
        detail::AdapterCallGuard guard(impl_->in_use, kModelName);
        detail::validate_prepared_input(prepared_input, impl_->contract.input_name,
                                        impl_->contract.batch,
                                        { kFaceModelInputChannels, kArcFaceInputExtent,
                                          kArcFaceInputExtent },
                                        kModelName);
        const std::size_t batch = static_cast<std::size_t>(prepared_input.shape[0]);
        impl_->inputs[0].shape[0] = static_cast<std::int64_t>(batch);
        impl_->inputs[0].data = prepared_input.data;
        impl_->inputs[0].byte_size = prepared_input.byte_size;
        impl_->inputs[0].memory_kind = prepared_input.memory_kind;
        impl_->outputs[0].shape[0] = static_cast<std::int64_t>(batch);
        impl_->executor->run(impl_->inputs, impl_->outputs);
        return detail::decode_arcface(impl_->output.data(), batch * kArcFaceEmbeddingLength,
                                      batch);
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
