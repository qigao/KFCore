#include "kfcore/hand_models/cpu.hpp"
#include "tinytest.hpp"

#include <functional>
#include <string>
#include <type_traits>

using namespace kfcore::hand_models;

namespace
{

void check_error(const std::function<void()>& operation, HandModelErrorCode code,
                 const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const HandModelError& error)
    {
        threw = true;
        check(error.code() == code);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

static_assert(!std::is_copy_constructible_v<CpuHandBackend>);
static_assert(!std::is_copy_assignable_v<CpuHandBackend>);

spec("CPU hand model public validation")
{
    it("rejects missing hand model paths before creating sessions")
    {
        check_error([&] { (void)CpuHandBackend::load({}, {}); },
                    HandModelErrorCode::InvalidModelAsset, "Palm");
    }

    it("rejects zero CPU resource limits before reading model files")
    {
        HandOnnxModelPaths paths { "palm.onnx", "hand.onnx", "gesture.onnx" };
        CpuHandOptions options;
        options.max_model_bytes = 0;
        check_error([&] { (void)CpuHandBackend::load(paths, options); },
                    HandModelErrorCode::ResourceLimitExceeded, "positive");
    }
}
