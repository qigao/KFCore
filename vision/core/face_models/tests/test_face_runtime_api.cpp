#include "kfcore/face_models/runtime.hpp"

#include "tinytest.hpp"

#include <type_traits>

using namespace kfcore::face_models;

spec("face runtime public API")
{
    it("exposes concrete backend-neutral typed models without pipeline interfaces")
    {
        check_false(std::is_abstract_v<FaceDetector>);
        check_false(std::is_abstract_v<FaceLandmarker>);
        check_false(std::is_copy_constructible_v<FaceMesh>);
        check_false(std::is_copy_assignable_v<FaceMesh>);
    }
}
