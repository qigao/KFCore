#include "kfcore/hand_models/cpu.hpp"

#include "tinytest.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace kfcore::hand_models;

spec("CPU hand model integration")
{
    it("executes the public hand backend with trusted ONNX models")
    {
        const HandOnnxModelPaths paths {
            KFCORE_HAND_CPU_TEST_PALM,
            KFCORE_HAND_CPU_TEST_LANDMARK,
            KFCORE_HAND_CPU_TEST_CLASSIFIER,
        };
        CpuHandOptions options;
        options.intra_op_threads = 1;
        options.inter_op_threads = 1;
        auto backend = CpuHandBackend::load(paths, options);

        constexpr std::int32_t kExtent = 256;
        std::vector<std::uint8_t> pixels(
            static_cast<std::size_t>(kExtent) * kExtent * 3U, 0U);
        const kfcore::image::ImageView image {
            pixels.data(), pixels.size(), kExtent, kExtent,
            static_cast<std::size_t>(kExtent) * 3U,
            kfcore::image::PixelFormat::Bgr8,
            kfcore::image::MemoryKind::Host,
        };
        const HandFrame frame = backend->infer(image);

        check_true(frame.hands.size() <= options.max_hands);
        check_true(std::isfinite(frame.timings.total_ms));
        check_true(frame.timings.preprocess_ms >= 0.0);
        check_true(frame.timings.palm_inference_ms >= 0.0);
    }
}
