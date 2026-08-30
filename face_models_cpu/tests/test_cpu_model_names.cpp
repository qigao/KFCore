#include "kfcore/face_models/cpu_model_names.hpp"
#include "tinytest.hpp"

#include <string_view>

namespace names = kfcore::face_models::cpu_model_names;

spec("CPU face model asset names")
{
    it("maps semantic roles to the fixed yolo-models ONNX assets")
    {
        check(names::face_detector == std::string_view("yolov12n-face.onnx"));
        check(names::face_mesh_landmarker ==
              std::string_view("MediaPipeFaceLandmarkDetector.onnx"));
        check(names::face_68_landmarker == std::string_view("2dfan4.onnx"));
        check(names::face_embedding == std::string_view("arcface_w600k_r50.onnx"));
        check(names::age_gender_estimator == std::string_view("age-gender.onnx"));
        check(names::face_swapper == std::string_view("inswapper_128.onnx"));
        check(names::face_restorer == std::string_view("gfpgan_1.4.onnx"));
        check(names::face_swap_projection == std::string_view("model_matrix.bin"));
    }
}
