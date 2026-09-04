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

constexpr char kModelName[] = "Face68";

detail::CpuModelContract model_contract()
{
    return { kModelName,
             { detail::fp32_contract("input", { 1, 3, 256, 256 }) },
             { detail::fp32_contract("landmarks_xyscore", { 1, 68, 3 }),
               detail::fp32_contract("heatmaps", { 1, 68, 64, 64 }) } };
}

} // namespace

struct CpuFace68::Impl final
{
    Impl(const std::filesystem::path& path, const CpuFaceModelOptions& options)
        : session(detail::make_session(kModelName, path, model_contract(), options))
    {
    }

    runtime_onnx::OwnedSession session;
    std::atomic_flag       in_use = ATOMIC_FLAG_INIT;
};

CpuFace68::CpuFace68(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CpuFace68::~CpuFace68() = default;

std::unique_ptr<CpuFace68> CpuFace68::load(const std::filesystem::path& model_path,
                                           const CpuFaceModelOptions& options)
{
    try
    {
        return std::unique_ptr<CpuFace68>(
            new CpuFace68(std::make_unique<Impl>(model_path, options)));
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

Face68Result CpuFace68::infer(const CpuTensorView& prepared_input)
{
    try
    {
        detail::CpuCallGuard guard(impl_->in_use, kModelName);
        const std::vector<detail::CpuSessionOutput> outputs =
            detail::run(impl_->session,
                        { detail::image_input(prepared_input, kFace68InputExtent) });
        return detail::decode_face68(outputs.at(0).float_values.data(),
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
