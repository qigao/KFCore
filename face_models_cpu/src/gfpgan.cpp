#include "kfcore/face_models/cpu.hpp"

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

constexpr char kModelName[] = "GFPGAN";

detail::CpuModelContract model_contract()
{
    return { kModelName,
             { detail::fp32_contract("input", { 1, 3, 512, 512 }) },
             { detail::fp32_contract("output", { 1, 3, 512, 512 }) } };
}

} // namespace

struct CpuGfpGan::Impl final
{
    Impl(const std::filesystem::path& path, const CpuFaceModelOptions& options)
        : session(detail::make_session(kModelName, path, model_contract(), options))
    {
    }

    runtime_onnx::OwnedSession session;
    std::atomic_flag       in_use = ATOMIC_FLAG_INIT;
};

CpuGfpGan::CpuGfpGan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CpuGfpGan::~CpuGfpGan() = default;

std::unique_ptr<CpuGfpGan> CpuGfpGan::load(const std::filesystem::path& model_path,
                                           const CpuFaceModelOptions& options)
{
    try
    {
        return std::unique_ptr<CpuGfpGan>(
            new CpuGfpGan(std::make_unique<Impl>(model_path, options)));
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

GfpGanResult CpuGfpGan::infer(const CpuTensorView& prepared_input)
{
    try
    {
        detail::CpuCallGuard guard(impl_->in_use, kModelName);
        std::vector<detail::CpuSessionOutput> outputs =
            detail::run(impl_->session,
                        { detail::image_input(prepared_input, kGfpGanInputExtent) });
        GfpGanResult result;
        result.values = std::move(outputs.at(0).float_values);
        return result;
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
