#include "kfcore/face_models/error.hpp"
#include "kfcore/face_models/tensorrt.hpp"
#include "tinytest.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <type_traits>

using namespace kfcore::face_models;

namespace
{

void expect_error(const std::function<void()>& operation, FaceModelErrorCode expected,
                  const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const FaceModelError& error)
    {
        threw = true;
        check(error.code() == expected);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

spec("TensorRT face swap adapter API")
{
    it("exposes move-only owning adapters")
    {
        check_false(std::is_copy_constructible_v<TensorRtInSwapper>);
        check_false(std::is_copy_assignable_v<TensorRtInSwapper>);
        check_false(std::is_copy_constructible_v<TensorRtGfpGan>);
        check_false(std::is_copy_assignable_v<TensorRtGfpGan>);
    }

    it("exposes caller-owned output inference for device-resident pipelines")
    {
        using InSwapperInferInto = void (TensorRtInSwapper::*)(
            const kfcore::tensorrt::TensorView&, const kfcore::tensorrt::TensorView&,
            const kfcore::tensorrt::MutableTensorView&);
        using GfpGanInferInto = void (TensorRtGfpGan::*)(
            const kfcore::tensorrt::TensorView&,
            const kfcore::tensorrt::MutableTensorView&);

        constexpr bool has_inswapper_infer_into = std::is_same_v<
            decltype(static_cast<InSwapperInferInto>(&TensorRtInSwapper::infer_into)),
            InSwapperInferInto>;
        constexpr bool has_gfpgan_infer_into = std::is_same_v<
            decltype(static_cast<GfpGanInferInto>(&TensorRtGfpGan::infer_into)), GfpGanInferInto>;
        check_true(has_inswapper_infer_into);
        check_true(has_gfpgan_infer_into);
    }

    it("validates InSwapper options before engine loading")
    {
        InSwapperOptions options;
        options.max_batch = 0;
        expect_error(
            [&] { (void)TensorRtInSwapper::load(std::filesystem::path {}, options); },
            FaceModelErrorCode::ResourceLimitExceeded, "max_batch");
    }

    it("validates GFPGAN options before engine loading")
    {
        GfpGanOptions options;
        options.input_name.clear();
        expect_error([&] { (void)TensorRtGfpGan::load(std::filesystem::path {}, options); },
                     FaceModelErrorCode::InvalidArgument, "empty");
    }
}
