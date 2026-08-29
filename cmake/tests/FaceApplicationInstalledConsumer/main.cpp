#include <kfcore/face_applications/error.hpp>
#include <kfcore/face_applications/tensorrt.hpp>

#include <type_traits>

int main()
{
    using kfcore::face_applications::FaceApplicationError;
    using kfcore::face_applications::FaceApplicationErrorCode;
    using kfcore::face_applications::FaceAnalysis;
    using kfcore::face_applications::TensorRtFaceSwapApplication;
    using AnalyzeSignature = FaceAnalysis (TensorRtFaceSwapApplication::*)(
        const kfcore::image::ImageView&);

    const FaceApplicationError error(FaceApplicationErrorCode::InvalidArgument,
                                     "installed consumer");
    return !std::is_copy_constructible_v<TensorRtFaceSwapApplication> &&
                   std::is_same_v<decltype(&TensorRtFaceSwapApplication::analyze),
                                  AnalyzeSignature> &&
                   error.code() == FaceApplicationErrorCode::InvalidArgument
               ? 0
               : 1;
}
