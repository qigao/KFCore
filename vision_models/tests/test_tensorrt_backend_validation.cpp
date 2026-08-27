#include "kfcore/vision_models/tensorrt.hpp"
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

static_assert(!std::is_copy_constructible_v<TensorRtHandBackend>);
static_assert(!std::is_copy_assignable_v<TensorRtHandBackend>);
static_assert(!std::is_copy_constructible_v<TensorRtFaceLandmarker>);
static_assert(!std::is_copy_assignable_v<TensorRtFaceLandmarker>);

spec("TensorRT vision model public validation")
{
    it("rejects missing hand engine paths")
    {
        check_error([&] { (void)TensorRtHandBackend::load({}, {}); },
                    VisionModelErrorCode::InvalidModelAsset, "Palm");
    }

    it("rejects missing face landmark engine paths")
    {
        check_error([&] { (void)TensorRtFaceLandmarker::load({}, {}); },
                    VisionModelErrorCode::InvalidModelAsset, "face landmark");
    }

    it("rejects zero CUDA resource limits before reading engines")
    {
        HandTensorRtEnginePaths paths { "palm.engine", "hand.engine", "gesture.engine" };
        TensorRtVisionOptions options;
        options.max_engine_bytes = 0;
        check_error([&] { (void)TensorRtHandBackend::load(paths, options); },
                    VisionModelErrorCode::ResourceLimitExceeded, "positive");
    }
}
