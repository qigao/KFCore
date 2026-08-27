#include "kfcore/face_applications/cpu.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <filesystem>
#include <limits>

using namespace kfcore::face_applications;

namespace
{

CpuFaceApplicationModelPaths valid_paths()
{
    CpuFaceApplicationModelPaths paths;
    paths.detector_model   = "detector.onnx";
    paths.face68_model     = "face68.onnx";
    paths.arcface_model    = "arcface.onnx";
    paths.inswapper_model  = "inswapper.onnx";
    paths.inswapper_matrix = "matrix.bin";
    return paths;
}

} // namespace

spec("ONNX CPU face application validation")
{
    it("rejects empty required model paths before opening assets")
    {
        CpuFaceApplicationModelPaths paths = valid_paths();
        paths.arcface_model.clear();
        check_throws_as(OnnxFaceSwapApplication::load(paths), CpuFaceApplicationError);
    }

    it("rejects invalid score blend and resource limits before loading models")
    {
        const CpuFaceApplicationModelPaths paths = valid_paths();
        CpuFaceSwapOptions options;
        options.detector_score_threshold =
            (std::numeric_limits<float>::quiet_NaN)();
        check_throws_as(OnnxFaceSwapApplication::load(paths, options),
                        CpuFaceApplicationError);

        options = {};
        options.enhancer_blend = 1.1F;
        check_throws_as(OnnxFaceSwapApplication::load(paths, options),
                        CpuFaceApplicationError);

        options = {};
        options.max_image_bytes = 0;
        check_throws_as(OnnxFaceSwapApplication::load(paths, options),
                        CpuFaceApplicationError);
    }
}
