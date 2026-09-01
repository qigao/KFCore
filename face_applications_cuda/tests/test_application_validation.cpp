#include "kfcore/face_applications/error.hpp"
#include "kfcore/face_applications/tensorrt.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace kfcore::face_applications;

namespace
{

void expect_invalid(const std::function<void()>& operation, const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const FaceApplicationError& error)
    {
        threw = true;
        check(error.code() == FaceApplicationErrorCode::InvalidArgument);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

FaceApplicationModelPaths empty_paths()
{
    return {};
}

FaceApplicationModelPaths complete_dummy_paths()
{
    return { "detector.engine", "face68.engine", "arcface.engine", "inswapper.engine",
             "matrix.bin", "gfpgan.engine", std::nullopt };
}

} // namespace

spec("TensorRT face application validation")
{
    it("offers KFCore-owned default engine discovery")
    {
        using DefaultLoadSignature = std::unique_ptr<TensorRtFaceSwapApplication> (*)(
            const FaceSwapOptions&);
        check_true((std::is_same_v<decltype(static_cast<DefaultLoadSignature>(
                                      &TensorRtFaceSwapApplication::load)),
                                   DefaultLoadSignature>));
    }

    it("selects only the highest-score configured 12face class with stable ties")
    {
        const std::vector<kfcore::yolo::Detection> detections = {
            { { 0.0F, 0.0F, 10.0F, 10.0F }, 0.7F, 0 },
            { { 1.0F, 1.0F, 11.0F, 11.0F }, 0.9F, 1 },
            { { 2.0F, 2.0F, 12.0F, 12.0F }, 0.8F, 0 },
            { { 3.0F, 3.0F, 13.0F, 13.0F }, 0.8F, 0 },
        };
        const auto selected = select_highest_score_face(detections, 0, 0.5F);
        check_true(selected.has_value());
        check(selected->box.left == 2.0F);
        check(selected->score == 0.8F);
        check_false(select_highest_score_face(detections, 0, 0.81F).has_value());
    }

    it("ignores non-finite scores instead of selecting them")
    {
        std::vector<kfcore::yolo::Detection> detections = {
            { { 0.0F, 0.0F, 10.0F, 10.0F },
              (std::numeric_limits<float>::quiet_NaN)(), 0 }
        };
        check_false(select_highest_score_face(detections, 0, 0.0F).has_value());
    }

    it("rejects invalid options before loading any engine")
    {
        FaceSwapOptions options;
        options.detector_score_threshold = -0.1F;
        expect_invalid(
            [&] { (void)TensorRtFaceSwapApplication::load(empty_paths(), options); },
            "threshold");

        options.detector_score_threshold = 0.5F;
        options.enhancer_blend = 1.1F;
        expect_invalid(
            [&] { (void)TensorRtFaceSwapApplication::load(empty_paths(), options); },
            "blend");

        options.enhancer_blend = 0.8F;
        options.face_class_id = -1;
        expect_invalid(
            [&] { (void)TensorRtFaceSwapApplication::load(empty_paths(), options); },
            "class");
    }

    it("rejects missing required model paths before runtime loading")
    {
        expect_invalid(
            [&] { (void)TensorRtFaceSwapApplication::load(empty_paths(), FaceSwapOptions {}); },
            "detector");
    }

    it("requires the InSwapper engine and matrix assets")
    {
        FaceApplicationModelPaths paths = complete_dummy_paths();
        paths.inswapper_engine.clear();
        expect_invalid(
            [&] { (void)TensorRtFaceSwapApplication::load(paths, FaceSwapOptions {}); },
            "InSwapper engine");
    }

    it("requires the final TensorRT chain to share one CUDA device")
    {
        FaceSwapOptions options;
        options.inswapper.engine.device_id = 0;
        options.gfpgan.engine.device_id = 1;
        expect_invalid(
            [&]
            {
                (void)TensorRtFaceSwapApplication::load(complete_dummy_paths(), options);
            },
            "same CUDA device");
    }

    it("exposes a move-only non-copying facade")
    {
        check_false(std::is_copy_constructible_v<TensorRtFaceSwapApplication>);
        check_false(std::is_copy_assignable_v<TensorRtFaceSwapApplication>);
        using AnalyzeSignature = FaceAnalysis (TensorRtFaceSwapApplication::*)(
            const kfcore::image::ImageView&);
        using SwapSignature = kfcore::image::BgrImage (TensorRtFaceSwapApplication::*)(
            const kfcore::image::ImageView&, const kfcore::image::ImageView&);
        check_true((std::is_same_v<decltype(&TensorRtFaceSwapApplication::analyze),
                                   AnalyzeSignature>));
        check_true((std::is_same_v<decltype(&TensorRtFaceSwapApplication::swap),
                                   SwapSignature>));
    }

    it("exposes all-face analysis and prepared swap entry points")
    {
        using AnalyzeAllSignature = std::vector<FaceAnalysis> (
            TensorRtFaceSwapApplication::*)(const kfcore::image::ImageView&);
        using PreparedSignature = kfcore::image::BgrImage (
            TensorRtFaceSwapApplication::*)(
                const kfcore::image::ImageView&,
                const kfcore::face_models::ArcFaceResult&,
                const FiveLandmarks&, bool);
        using PreparedBlendSignature = kfcore::image::BgrImage (
            TensorRtFaceSwapApplication::*)(
                const kfcore::image::ImageView&,
                const kfcore::face_models::ArcFaceResult&,
                const FiveLandmarks&, bool, float);

        check_true((std::is_same_v<decltype(
                                      &TensorRtFaceSwapApplication::analyze_all),
                                   AnalyzeAllSignature>));
        check_true((std::is_same_v<decltype(static_cast<PreparedSignature>(
                                      &TensorRtFaceSwapApplication::swap_prepared)),
                                   PreparedSignature>));
        check_true((std::is_same_v<decltype(static_cast<PreparedBlendSignature>(
                                      &TensorRtFaceSwapApplication::swap_prepared)),
                                   PreparedBlendSignature>));
    }
}
