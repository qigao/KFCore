#include "kfcore/face_applications/cpu.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <filesystem>
#include <limits>
#include <type_traits>
#include <vector>

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
    it("offers KFCore-owned default model discovery")
    {
        using DefaultLoadSignature = std::unique_ptr<OnnxFaceSwapApplication> (*)(
            const CpuFaceSwapOptions&);
        check_true((std::is_same_v<decltype(static_cast<DefaultLoadSignature>(
                                      &OnnxFaceSwapApplication::load)),
                                   DefaultLoadSignature>));
    }

    it("rejects empty required model paths before opening assets")
    {
        CpuFaceApplicationModelPaths paths = valid_paths();
        paths.arcface_model.clear();
        check_throws_as(OnnxFaceSwapApplication::load(paths), CpuFaceApplicationError);
    }

    it("requires the InSwapper model and matrix assets")
    {
        CpuFaceApplicationModelPaths paths = valid_paths();
        paths.inswapper_model.clear();
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

    it("exposes both owned-BGR and borrowed-ImageView entry points")
    {
        using AnalyzeView = CpuFaceAnalysis (OnnxFaceSwapApplication::*)(
            const kfcore::image::ImageView&);
        using SwapView = kfcore::image::BgrImage (OnnxFaceSwapApplication::*)(
            const kfcore::image::ImageView&, const kfcore::image::ImageView&);
        check_true((std::is_same_v<decltype(static_cast<AnalyzeView>(
                                      &OnnxFaceSwapApplication::analyze)), AnalyzeView>));
        check_true((std::is_same_v<decltype(static_cast<SwapView>(
                                      &OnnxFaceSwapApplication::swap)), SwapView>));
    }

    it("exposes all-face analysis and prepared swap entry points")
    {
        using AnalyzeAllOwned = std::vector<CpuFaceAnalysis> (
            OnnxFaceSwapApplication::*)(const kfcore::image::BgrImage&);
        using AnalyzeAllView = std::vector<CpuFaceAnalysis> (
            OnnxFaceSwapApplication::*)(const kfcore::image::ImageView&);
        using PreparedOwned = kfcore::image::BgrImage (
            OnnxFaceSwapApplication::*)(
                const kfcore::image::BgrImage&,
                const kfcore::face_models::ArcFaceResult&,
                const CpuFiveLandmarks&, bool);
        using PreparedView = kfcore::image::BgrImage (
            OnnxFaceSwapApplication::*)(
                const kfcore::image::ImageView&,
                const kfcore::face_models::ArcFaceResult&,
                const CpuFiveLandmarks&, bool);
        using PreparedBlendView = kfcore::image::BgrImage (
            OnnxFaceSwapApplication::*)(
                const kfcore::image::ImageView&,
                const kfcore::face_models::ArcFaceResult&,
                const CpuFiveLandmarks&, bool, float);

        check_true((std::is_same_v<decltype(static_cast<AnalyzeAllOwned>(
                                      &OnnxFaceSwapApplication::analyze_all)),
                                   AnalyzeAllOwned>));
        check_true((std::is_same_v<decltype(static_cast<AnalyzeAllView>(
                                      &OnnxFaceSwapApplication::analyze_all)),
                                   AnalyzeAllView>));
        check_true((std::is_same_v<decltype(static_cast<PreparedOwned>(
                                      &OnnxFaceSwapApplication::swap_prepared)),
                                   PreparedOwned>));
        check_true((std::is_same_v<decltype(static_cast<PreparedView>(
                                      &OnnxFaceSwapApplication::swap_prepared)),
                                   PreparedView>));
        check_true((std::is_same_v<decltype(static_cast<PreparedBlendView>(
                                      &OnnxFaceSwapApplication::swap_prepared)),
                                   PreparedBlendView>));
    }
}
