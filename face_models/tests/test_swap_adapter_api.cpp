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
