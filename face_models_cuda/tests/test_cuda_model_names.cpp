#include "kfcore/face_models/cuda_model_names.hpp"
#include "tinytest.hpp"

#include <string_view>

namespace names = kfcore::face_models::cuda_model_names;

spec("CUDA face model asset names")
{
    it("maps semantic roles to the fixed TensorRT engine assets")
    {
        check(names::engine_profile_directory == std::string_view("tensorrt"));
        check(names::face_detector == std::string_view("yolov12n-face.engine"));
        check(names::face_mesh_landmarker == std::string_view("face_landmark.engine"));
        check(names::face_68_landmarker == std::string_view("2dfan4.engine"));
        check(names::face_embedding == std::string_view("arcface_w600k_r50.engine"));
        check(names::age_gender_estimator == std::string_view("age-gender.engine"));
        check(names::face_swapper == std::string_view("inswapper_128.engine"));
        check(names::face_restorer == std::string_view("gfpgan_1.4.engine"));
        check(names::face_swap_projection == std::string_view("model_matrix.bin"));
    }
}
