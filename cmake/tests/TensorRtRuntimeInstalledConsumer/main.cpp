#include <kfcore/face_models/tensorrt.hpp>
#include <kfcore/tensorrt/runtime.hpp>

int main()
{
    kfcore::tensorrt::TensorView input;
    input.data_type = kfcore::tensorrt::DataType::Float32;
    input.memory_kind = kfcore::tensorrt::MemoryKind::Host;

    const kfcore::tensorrt::TensorRtError runtime_error(
        kfcore::tensorrt::TensorRtErrorCode::InvalidArgument, "runtime consumer");
    const kfcore::face_models::FaceModelError model_error(
        kfcore::face_models::FaceModelErrorCode::InvalidArgument, "face consumer");

    return input.data_type == kfcore::tensorrt::DataType::Float32 &&
                   input.memory_kind == kfcore::tensorrt::MemoryKind::Host &&
                   kfcore::face_models::kFace68LandmarkCount == 68 &&
                   runtime_error.code() ==
                       kfcore::tensorrt::TensorRtErrorCode::InvalidArgument &&
                   model_error.code() ==
                       kfcore::face_models::FaceModelErrorCode::InvalidArgument
               ? 0
               : 1;
}
