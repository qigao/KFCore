#include "kfcore/runtime_onnx/runtime.hpp"
#include "tinytest.hpp"

#include <array>
#include <filesystem>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif

using namespace kfcore::runtime_onnx;

namespace
{
ModelContract contract()
{
    return { "provider-test", { { "input", ElementType::Float32, { 1, 4 } } },
             { { "output", ElementType::Float32, { 1, 4 } } } };
}
}

spec("ONNX execution provider contract")
{
    it("executes a tensor operation and returns host output across repeated runs")
    {
        Environment environment("KFCoreOnnxExecutionTest");
        Session session(environment,
                        std::filesystem::path(KFCORE_ONNX_FIXTURES) / "relu.onnx", contract());
        const std::array<float, 4> input { -2.0F, 0.0F, 3.0F, -1.0F };
        const std::vector<FloatTensorView> views { { input.data(), input.size(), { 1, 4 } } };
        const auto first = session.run(views);
        const auto second = session.run(views);
        check(first.size() == 1U);
        check(first.at(0).float_values.size() == input.size());
        check(first.at(0).float_values.at(0) == 0.0F);
        check(first.at(0).float_values.at(1) == 0.0F);
        check(first.at(0).float_values.at(2) == 3.0F);
        check(first.at(0).float_values.at(3) == 0.0F);
        check_true(first.at(0).float_values == second.at(0).float_values);
#if defined(_WIN32) && KFCORE_TEST_ONNX_CUDA
        check_true(GetModuleHandleW(L"onnxruntime_providers_cuda.dll") != nullptr);
        check_true(GetModuleHandleW(L"onnxruntime_providers_tensorrt.dll") == nullptr);
#endif
    }

    it("honors the configured CPU node policy")
    {
        Environment environment("KFCoreOnnxProviderTest");
        bool rejected = false;
        try
        {
            Session session(environment,
                            std::filesystem::path(KFCORE_ONNX_FIXTURES) / "binarizer.onnx",
                            contract());
            const std::array<float, 4> input { -2.0F, 0.0F, 3.0F, -1.0F };
            const auto output = session.run({ { input.data(), input.size(), { 1, 4 } } });
            check(output.at(0).float_values.at(2) == 1.0F);
        }
        catch (const Error& error)
        {
            rejected = true;
            check_true(error.code() == ErrorCode::RuntimeFailure);
            check_true(std::string(error.what()).find("CPU") != std::string::npos);
        }
        check(rejected == (KFCORE_TEST_ONNX_CUDA != 0 && KFCORE_TEST_ALLOW_CPU_NODES == 0));
    }
}
