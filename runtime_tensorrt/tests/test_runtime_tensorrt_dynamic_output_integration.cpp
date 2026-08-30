#include "kfcore/tensorrt/runtime.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kfcore::tensorrt;

namespace
{

constexpr std::int64_t kBatch                  = 1;
constexpr std::int64_t kChannels               = 3;
constexpr std::int64_t kInputSize              = 192;
constexpr std::int64_t kDetectionWidth         = 8;
constexpr std::size_t  kMaximumPalmCandidates  = 2016;
constexpr std::size_t  kMaximumPalmOutputBytes =
    kMaximumPalmCandidates * static_cast<std::size_t>(kDetectionWidth) * sizeof(float);

std::filesystem::path required_engine_path()
{
    constexpr char kVariable[] = "KFCORE_RUNTIME_TENSORRT_TEST_ENGINE_PALM";
    const char* value = std::getenv(kVariable);
    if (value == nullptr || *value == '\0')
    {
        throw std::runtime_error(std::string(kVariable) +
                                 " must name a trusted Palm TensorRT engine");
    }
    return std::filesystem::path(value);
}

const TensorDescriptor& tensor_named(const std::vector<TensorDescriptor>& tensors,
                                     const char* name, TensorIoMode mode)
{
    for (const TensorDescriptor& tensor : tensors)
    {
        if (tensor.name == name && tensor.mode == mode)
        {
            return tensor;
        }
    }
    throw std::runtime_error(std::string("trusted Palm engine lacks tensor: ") + name);
}

void check_error(const std::function<void()>& operation, TensorRtErrorCode code,
                 const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const TensorRtError& error)
    {
        threw = true;
        check(error.code() == code);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

spec("TensorRT runtime data-dependent Palm output integration")
{
    it("executes bounded NMS output and preserves the executor for another call")
    {
        auto engine = Engine::load(required_engine_path());
        const TensorDescriptor& input = tensor_named(engine->tensors(), "input",
                                                     TensorIoMode::Input);
        const TensorDescriptor& output = tensor_named(
            engine->tensors(),
            "pdscore_boxx_boxy_boxsize_kp0x_kp0y_kp2x_kp2y", TensorIoMode::Output);
        check(input.data_type == DataType::Float32);
        check(output.data_type == DataType::Float32);

        auto executor = engine->create_executor();
        const TensorShape input_shape = { kBatch, kChannels, kInputSize, kInputSize };
        std::vector<float> input_values(
            static_cast<std::size_t>(kBatch * kChannels * kInputSize * kInputSize), 0.0F);
        const TensorView input_view { input.name, DataType::Float32, input_shape,
                                      input_values.data(), input_values.size() * sizeof(float),
                                      MemoryKind::Host };
        const DynamicOutputRequest output_request { output.name, DataType::Float32,
                                                    kMaximumPalmOutputBytes };

        for (int attempt = 0; attempt < 2; ++attempt)
        {
            const std::vector<HostTensor> results =
                executor->run_dynamic({ input_view }, { output_request });
            check(results.size() == 1);
            check(results[0].shape.size() == 2);
            check(results[0].shape[0] >= 0);
            check(results[0].shape[0] <=
                  static_cast<std::int64_t>(kMaximumPalmCandidates));
            check(results[0].shape[1] == kDetectionWidth);
            const std::vector<float> values = results[0].float32_values();
            check(values.size() == static_cast<std::size_t>(results[0].shape[0] *
                                                            kDetectionWidth));
            for (float value : values)
            {
                check_true(std::isfinite(value));
            }
        }

        check_error(
            [&]
            {
                (void)executor->run_dynamic(
                    { input_view },
                    { { output.name, DataType::Float32, sizeof(float) } });
            },
            TensorRtErrorCode::ResourceLimitExceeded, "engine upper bound");
    }
}
