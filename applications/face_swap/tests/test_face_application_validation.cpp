#include "kfcore/face_applications/application.hpp"

#include "tinytest.hpp"

#include <limits>
#include <type_traits>

using namespace kfcore::face_applications;

namespace
{

FaceApplicationModelPackages valid_packages()
{
    FaceApplicationModelPackages packages;
    packages.detector = "detector-package";
    packages.face68 = "face68-package";
    packages.arcface = "arcface-package";
    packages.inswapper = "inswapper-package";
    packages.inswapper_matrix = "matrix.bin";
    return packages;
}

} // namespace

spec("face application validation")
{
    it("rejects empty required package paths before loading a backend")
    {
        kfcore::runtime::Runtime runtime;
        FaceApplicationModelPackages packages = valid_packages();
        packages.arcface.clear();
        const FaceApplicationPolicies policies = FaceApplicationPolicies::onnx_cuda(2U);

        check_throws_as(FaceSwapApplication::load(runtime, packages, policies),
                        FaceApplicationError);
        check_true(runtime.backend_ids().empty());
    }

    it("rejects invalid options before opening model packages")
    {
        kfcore::runtime::Runtime runtime;
        const FaceApplicationModelPackages packages = valid_packages();
        const FaceApplicationPolicies policies = FaceApplicationPolicies::onnx_cuda(2U);
        FaceSwapOptions options;
        options.detector_score_threshold =
            (std::numeric_limits<float>::quiet_NaN)();
        check_throws_as(FaceSwapApplication::load(runtime, packages, policies, options),
                        FaceApplicationError);

        options = {};
        options.enhancer_blend = 1.1F;
        check_throws_as(FaceSwapApplication::load(runtime, packages, policies, options),
                        FaceApplicationError);

        options = {};
        options.max_image_bytes = 0U;
        check_throws_as(FaceSwapApplication::load(runtime, packages, policies, options),
                        FaceApplicationError);
    }

    it("classifies an unreadable package as an invalid model asset")
    {
        kfcore::runtime::Runtime runtime;
        const FaceApplicationModelPackages packages = valid_packages();
        const FaceApplicationPolicies policies = FaceApplicationPolicies::onnx_cuda(2U);
        try
        {
            (void)FaceSwapApplication::load(runtime, packages, policies);
            check_true(false);
        }
        catch (const FaceApplicationError& error)
        {
            check_true(error.code() == FaceApplicationErrorCode::InvalidModelAsset);
        }
    }

    it("exposes borrowed image and prepared swap entry points")
    {
        using AnalyzeView = FaceAnalysis (FaceSwapApplication::*)(
            const kfcore::image::ImageView&);
        using AnalyzeAllView = std::vector<FaceAnalysis> (FaceSwapApplication::*)(
            const kfcore::image::ImageView&);
        using PreparedBlendView = kfcore::image::BgrImage (FaceSwapApplication::*)(
            const kfcore::image::ImageView&,
            const kfcore::face_models::ArcFaceResult&,
            const FiveLandmarks&, bool, float);

        check_true((std::is_same_v<decltype(static_cast<AnalyzeView>(
                                      &FaceSwapApplication::analyze)), AnalyzeView>));
        check_true((std::is_same_v<decltype(static_cast<AnalyzeAllView>(
                                      &FaceSwapApplication::analyze_all)),
                                   AnalyzeAllView>));
        check_true((std::is_same_v<decltype(static_cast<PreparedBlendView>(
                                      &FaceSwapApplication::swap_prepared)),
                                   PreparedBlendView>));
    }
}
