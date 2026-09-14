#include "kfcore/runtime_onnx/runtime.hpp"
#include "tinytest.hpp"

#include <string>

spec("ONNX library loading boundary")
{
    it("reports an invalid SDK root without loading the Windows system runtime")
    {
        bool rejected = false;
        try
        {
            kfcore::runtime_onnx::Environment environment("KFCoreInvalidOnnxRoot");
        }
        catch (const kfcore::runtime_onnx::Error& error)
        {
            rejected = true;
            check_true(error.code() == kfcore::runtime_onnx::ErrorCode::RuntimeFailure);
            check_true(std::string(error.what()).find("ONNX load stage:") != std::string::npos);
        }
        check_true(rejected);
    }
}
