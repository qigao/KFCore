#include "kfcore/face_models/cpu.hpp"
#include "kfcore/face_models/error.hpp"
#include "tinytest.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <type_traits>

using namespace kfcore::face_models;

namespace
{

void expect_invalid_asset(const std::function<void()>& operation, const char* model_name)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const FaceModelError& error)
    {
        threw = true;
        check(error.code() == FaceModelErrorCode::InvalidModelAsset);
        check(std::string(error.what()).find(model_name) != std::string::npos);
    }
    check_true(threw);
}

void expect_error(const std::function<void()>& operation, FaceModelErrorCode code,
                  const char* detail)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const FaceModelError& error)
    {
        threw = true;
        check(error.code() == code);
        check(std::string(error.what()).find(detail) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

spec("CPU face model adapter API")
{
    it("exposes move-only owning adapters")
    {
        check_false(std::is_copy_constructible_v<CpuFace68>);
        check_false(std::is_copy_assignable_v<CpuFace68>);
        check_false(std::is_copy_constructible_v<CpuArcFace>);
        check_false(std::is_copy_assignable_v<CpuArcFace>);
        check_false(std::is_copy_constructible_v<CpuAgeGender>);
        check_false(std::is_copy_assignable_v<CpuAgeGender>);
        check_false(std::is_copy_constructible_v<CpuInSwapper>);
        check_false(std::is_copy_assignable_v<CpuInSwapper>);
        check_false(std::is_copy_constructible_v<CpuGfpGan>);
        check_false(std::is_copy_assignable_v<CpuGfpGan>);
        check_false(std::is_copy_constructible_v<CpuFaceDetector>);
        check_false(std::is_copy_assignable_v<CpuFaceDetector>);
        check_false(std::is_copy_constructible_v<CpuFaceLandmarker>);
        check_false(std::is_copy_assignable_v<CpuFaceLandmarker>);
    }

    it("exposes all-face detector inference without changing the legacy result")
    {
        using InferAllSignature = FaceDetectionsResult (CpuFaceDetector::*)(
            const kfcore::image::ImageView&);
        using InferSignature = FaceDetectionResult (CpuFaceDetector::*)(
            const kfcore::image::ImageView&);
        check_true((std::is_same_v<decltype(&CpuFaceDetector::infer_all),
                                   InferAllSignature>));
        check_true((std::is_same_v<decltype(&CpuFaceDetector::infer),
                                   InferSignature>));
    }

    it("rejects empty model assets before creating ONNX Runtime sessions")
    {
        const std::filesystem::path empty;
        expect_invalid_asset([&] { (void)CpuFace68::load(empty); }, "Face68");
        expect_invalid_asset([&] { (void)CpuArcFace::load(empty); }, "ArcFace");
        expect_invalid_asset([&] { (void)CpuAgeGender::load(empty); }, "AgeGender");
        expect_invalid_asset([&] { (void)CpuInSwapper::load(empty); }, "InSwapper");
        expect_invalid_asset([&] { (void)CpuGfpGan::load(empty); }, "GFPGAN");
        expect_invalid_asset([&] { (void)CpuFaceDetector::load(empty); },
                             "YOLOv12 face detector");
        expect_invalid_asset([&] { (void)CpuFaceLandmarker::load(empty); },
                             "MediaPipe face landmark");
    }

    it("rejects invalid FaceMesh options before reading model assets")
    {
        CpuFaceMeshOptions options;
        options.face_detection_score_threshold = 1.01F;
        expect_error([&] { (void)CpuFaceDetector::load("detector.onnx", options); },
                     FaceModelErrorCode::InvalidArgument, "score threshold");
        options.face_detection_score_threshold = 0.50F;
        options.face_class_id = -1;
        expect_error([&] { (void)CpuFaceDetector::load("detector.onnx", options); },
                     FaceModelErrorCode::InvalidArgument, "class id");
    }
}
