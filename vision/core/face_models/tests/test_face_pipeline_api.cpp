#include "kfcore/face_models/core.hpp"

#include <tinytest.h>

#include <type_traits>

using namespace kfcore::face_models;

spec("face model core public pipeline API")
{
    it("exposes backend interfaces and a non-copyable FaceMesh pipeline")
    {
        check_true(std::is_abstract_v<FaceDetectorBackend>);
        check_true(std::is_abstract_v<FaceLandmarkBackend>);
        check_false(std::is_copy_constructible_v<FaceMeshPipeline>);
        check_false(std::is_copy_assignable_v<FaceMeshPipeline>);
    }
}
