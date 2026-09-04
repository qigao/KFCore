#include "kfcore/face_models/cpu.hpp"

#include "decode.hpp"
#include "support.hpp"

#include <atomic>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace kfcore::face_models
{
namespace
{

constexpr char kModelName[] = "ArcFace";

detail::CpuModelContract model_contract()
{
    return { kModelName,
             { detail::fp32_contract("input.1", { -1, 3, 112, 112 }) },
             { detail::fp32_contract("683", { 1, 512 }) } };
}

} // namespace

struct CpuArcFace::Impl final
{
    Impl(const std::filesystem::path& path, const CpuFaceModelOptions& options)
        : session(detail::make_session(kModelName, path, model_contract(), options))
    {
    }

    runtime_onnx::OwnedSession session;
    std::atomic_flag       in_use = ATOMIC_FLAG_INIT;
};

CpuArcFace::CpuArcFace(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CpuArcFace::~CpuArcFace() = default;

std::unique_ptr<CpuArcFace> CpuArcFace::load(const std::filesystem::path& model_path,
                                             const CpuFaceModelOptions& options)
{
    try
    {
        return std::unique_ptr<CpuArcFace>(
            new CpuArcFace(std::make_unique<Impl>(model_path, options)));
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        detail::throw_allocation(kModelName, "load");
    }
}

ArcFaceResult CpuArcFace::infer(const CpuTensorView& prepared_input)
{
    try
    {
        detail::CpuCallGuard guard(impl_->in_use, kModelName);
        const std::vector<detail::CpuSessionOutput> outputs =
            detail::run(impl_->session,
                        { detail::image_input(prepared_input, kArcFaceInputExtent) });
        return detail::decode_arcface(outputs.at(0).float_values.data(),
                                      outputs.at(0).float_values.size(), 1).front();
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        detail::throw_allocation(kModelName, "inference");
    }
}

} // namespace kfcore::face_models
