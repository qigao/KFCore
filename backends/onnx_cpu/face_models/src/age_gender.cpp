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

constexpr char kModelName[] = "AgeGender";

detail::CpuModelContract model_contract()
{
    return { kModelName,
             { detail::fp32_contract("pixel_values", { -1, 3, 224, 224 }) },
             { detail::fp32_contract("logits", { -1, 2 }) } };
}

} // namespace

struct CpuAgeGender::Impl final
{
    Impl(const std::filesystem::path& path, const CpuFaceModelOptions& options)
        : session(detail::make_session(kModelName, path, model_contract(), options))
    {
    }

    runtime_onnx::OwnedSession session;
    std::atomic_flag       in_use = ATOMIC_FLAG_INIT;
};

CpuAgeGender::CpuAgeGender(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CpuAgeGender::~CpuAgeGender() = default;

std::unique_ptr<CpuAgeGender> CpuAgeGender::load(
    const std::filesystem::path& model_path, const CpuFaceModelOptions& options)
{
    try
    {
        return std::unique_ptr<CpuAgeGender>(
            new CpuAgeGender(std::make_unique<Impl>(model_path, options)));
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

AgeGenderResult CpuAgeGender::infer(const CpuTensorView& prepared_input)
{
    try
    {
        detail::CpuCallGuard guard(impl_->in_use, kModelName);
        const std::vector<detail::CpuSessionOutput> outputs =
            detail::run(impl_->session,
                        { detail::image_input(prepared_input, kAgeGenderInputExtent) });
        return detail::decode_age_gender(outputs.at(0).float_values.data(),
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
