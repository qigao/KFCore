#include "kfcore/face_applications/application.hpp"

#include "tinytest.hpp"

#include <type_traits>

using namespace kfcore::face_applications;

spec("face application public API")
{
    it("routes every face model to the requested CUDA ordinal")
    {
        const FaceApplicationPolicies policies = FaceApplicationPolicies::onnx_cuda(3U);

        check_true(policies.detector.preferences().size() == 1U);
        check_true(policies.detector.preferences().front().backend_id == "onnxruntime");
        check_true(policies.detector.preferences().front().device_id == "cuda:3");
        check_true(policies.face68.preferences().front().device_id == "cuda:3");
        check_true(policies.arcface.preferences().front().device_id == "cuda:3");
        check_true(policies.inswapper.preferences().front().device_id == "cuda:3");
        check_true(policies.gfpgan.preferences().front().device_id == "cuda:3");
        check_true(policies.age_gender.preferences().front().backend_id == "onnxruntime");
        check_true(policies.age_gender.preferences().front().device_id == "cuda:3");
    }

    it("keeps the stateful application non-copyable")
    {
        check_false(std::is_copy_constructible_v<FaceSwapApplication>);
        check_false(std::is_copy_assignable_v<FaceSwapApplication>);
    }
}
