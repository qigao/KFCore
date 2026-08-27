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

    constexpr char kModelName[] = "Face68";
    constexpr std::size_t kValuesPerLandmarkOutput =
        kFace68LandmarkCount * static_cast<std::size_t>(kFace68LandmarkWidth);

} // namespace

struct TensorRtFace68::Impl final
{
    Impl(std::shared_ptr<const kfcore::tensorrt::Engine> engine_in,
         std::unique_ptr<kfcore::tensorrt::Executor> executor_in,
         detail::Face68Contract contract_in)
        : engine(std::move(engine_in))
        , executor(std::move(executor_in))
        , contract(std::move(contract_in))
        , landmarks(contract.landmark_float_capacity)
        , heatmaps(contract.heatmap_float_capacity)
        , inputs(1)
        , outputs(contract.has_heatmaps ? 2U : 1U)
    {
        inputs[0] = { contract.input_name,
                      kfcore::tensorrt::DataType::Float32,
                      { 1, kFaceModelInputChannels, kFace68InputExtent, kFace68InputExtent },
                      nullptr,
                      0,
                      kfcore::tensorrt::MemoryKind::Host };
        outputs[0] = { contract.landmark_output_name,
                       kfcore::tensorrt::DataType::Float32,
                       { 1, static_cast<std::int64_t>(kFace68LandmarkCount),
                         kFace68LandmarkWidth },
                       landmarks.data(),
                       landmarks.size() * sizeof(float),
                       kfcore::tensorrt::MemoryKind::Host };
        if (contract.has_heatmaps)
        {
            outputs[1] = { contract.heatmap_output_name,
                           kfcore::tensorrt::DataType::Float32,
                           { 1, static_cast<std::int64_t>(kFace68LandmarkCount),
                             kFace68HeatmapExtent, kFace68HeatmapExtent },
                           heatmaps.data(),
                           heatmaps.size() * sizeof(float),
                           kfcore::tensorrt::MemoryKind::Host };
        }
    }

    std::shared_ptr<const kfcore::tensorrt::Engine> engine;
    std::unique_ptr<kfcore::tensorrt::Executor>     executor;
    detail::Face68Contract                         contract;
    std::vector<float>                             landmarks;
    std::vector<float>                             heatmaps;
    std::vector<kfcore::tensorrt::TensorView>       inputs;
    std::vector<kfcore::tensorrt::MutableTensorView> outputs;
    std::atomic_flag                                in_use = ATOMIC_FLAG_INIT;
};

TensorRtFace68::TensorRtFace68(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

TensorRtFace68::~TensorRtFace68() = default;

std::unique_ptr<TensorRtFace68> TensorRtFace68::load(const std::filesystem::path& engine_path,
                                                     const Face68Options& options)
{
    try
    {
        detail::validate_face68_options(options);
        std::shared_ptr<const kfcore::tensorrt::Engine> engine =
            kfcore::tensorrt::Engine::load(engine_path, options.engine);
        detail::Face68Contract contract =
            detail::validate_face68_contract(engine->tensors(), options);
        std::unique_ptr<kfcore::tensorrt::Executor> executor = engine->create_executor();
        auto impl = std::make_unique<Impl>(engine, std::move(executor), std::move(contract));
        return std::unique_ptr<TensorRtFace68>(new TensorRtFace68(std::move(impl)));
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

std::vector<Face68Result>
TensorRtFace68::infer(const kfcore::tensorrt::TensorView& prepared_input)
{
    try
    {
        detail::AdapterCallGuard guard(impl_->in_use, kModelName);
        detail::validate_prepared_input(prepared_input, impl_->contract.input_name,
                                        impl_->contract.batch,
                                        { kFaceModelInputChannels, kFace68InputExtent,
                                          kFace68InputExtent },
                                        kModelName);
        const std::size_t batch = static_cast<std::size_t>(prepared_input.shape[0]);
        impl_->inputs[0].shape[0] = static_cast<std::int64_t>(batch);
        impl_->inputs[0].data = prepared_input.data;
        impl_->inputs[0].byte_size = prepared_input.byte_size;
        impl_->inputs[0].memory_kind = prepared_input.memory_kind;
        impl_->outputs[0].shape[0] = static_cast<std::int64_t>(batch);
        if (impl_->contract.has_heatmaps)
        {
            impl_->outputs[1].shape[0] = static_cast<std::int64_t>(batch);
        }
        impl_->executor->run(impl_->inputs, impl_->outputs);
        return detail::decode_face68(impl_->landmarks.data(),
                                     batch * kValuesPerLandmarkOutput, batch);
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
