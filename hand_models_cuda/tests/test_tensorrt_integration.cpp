#include "kfcore/hand_models/tensorrt.hpp"

#include "tinytest.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace kfcore::hand_models;

spec("TensorRT hand model integration")
{
    it("executes the public CUDA backend with trusted engines")
    {
        const HandTensorRtEnginePaths paths {
            KFCORE_HAND_TRT_TEST_PALM,
            KFCORE_HAND_TRT_TEST_LANDMARK,
            KFCORE_HAND_TRT_TEST_CLASSIFIER,
        };
        TensorRtHandOptions options;
        auto backend = TensorRtHandBackend::load(paths, options);
        auto input = TensorRtHandInput::create(options);

        constexpr std::int32_t kExtent = 256;
        std::vector<std::uint8_t> pixels(
            static_cast<std::size_t>(kExtent) * kExtent * 3U, 0U);
        const kfcore::image::ImageView host {
            pixels.data(), pixels.size(), kExtent, kExtent,
            static_cast<std::size_t>(kExtent) * 3U,
            kfcore::image::PixelFormat::Bgr8,
            kfcore::image::MemoryKind::Host,
        };
        const kfcore::image::FrameView prepared = input->prepare(host);
        check(prepared.compute.memory_kind == kfcore::image::MemoryKind::CudaDevice);

        const HandFrame frame = backend->infer(prepared.compute);
        check_true(frame.hands.size() <= options.max_hands);
        check_true(std::isfinite(frame.timings.total_ms));
        check_true(frame.timings.preprocess_ms >= 0.0);
        check_true(frame.timings.palm_inference_ms >= 0.0);
    }
}
