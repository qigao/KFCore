#include "kfcore/vision_models/cpu.hpp"
#include "tinytest.hpp"

#include <functional>
#include <string>
#include <type_traits>

using namespace kfcore::vision_models;

namespace
{

void check_error(const std::function<void()>& operation, VisionModelErrorCode code,
                 const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const VisionModelError& error)
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
static_assert(!std::is_copy_constructible_v<CpuFaceLandmarker>);
static_assert(!std::is_copy_assignable_v<CpuFaceLandmarker>);

spec("CPU vision model public validation")
{
    it("rejects missing hand model paths before creating sessions")
    {
        check_error([&] { (void)CpuHandBackend::load({}, {}); },
                    VisionModelErrorCode::InvalidModelAsset, "Palm");
    }

    it("rejects missing face landmark model paths")
    {
        check_error([&] { (void)CpuFaceLandmarker::load({}, {}); },
                    VisionModelErrorCode::InvalidModelAsset, "face landmark");
    }

    it("rejects zero CPU resource limits before reading model files")
    {
        HandOnnxModelPaths paths { "palm.onnx", "hand.onnx", "gesture.onnx" };
        CpuVisionOptions options;
        options.max_model_bytes = 0;
        check_error([&] { (void)CpuHandBackend::load(paths, options); },
                    VisionModelErrorCode::ResourceLimitExceeded, "positive");
    }
}
