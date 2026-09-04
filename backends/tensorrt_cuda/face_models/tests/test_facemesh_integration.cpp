#include "kfcore/face_models/tensorrt.hpp"

#include "tinytest.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace kfcore::face_models;

spec("TensorRT FaceMesh integration")
{
    it("executes detector and landmarker through the public CUDA face API")
    {
        TensorRtFaceMeshOptions options;
        options.face_detection_score_threshold = 0.25F;
        auto detector = TensorRtFaceDetector::load(
            KFCORE_FACE_TRT_TEST_DETECTOR, options);
        auto landmarker = TensorRtFaceLandmarker::load(
            KFCORE_FACE_TRT_TEST_LANDMARKER, options);

        constexpr std::int32_t kExtent = 256;
        std::vector<std::uint8_t> pixels(
            static_cast<std::size_t>(kExtent) * kExtent * 3U, 0U);
        const kfcore::image::ImageView image {
            pixels.data(), pixels.size(), kExtent, kExtent,
            static_cast<std::size_t>(kExtent) * 3U,
            kfcore::image::PixelFormat::Bgr8,
            kfcore::image::MemoryKind::Host,
        };

        const FaceDetectionResult detection = detector->infer(image);
        check_true(std::isfinite(detection.total_ms));
        const FaceLandmarkResult landmarks = landmarker->infer(
            image, { 0.0F, 0.0F, static_cast<float>(kExtent),
                     static_cast<float>(kExtent) });
        check_true(std::isfinite(landmarks.confidence));
        check_true(std::isfinite(landmarks.total_ms));
        for (const Point3f& point : landmarks.landmarks)
        {
            check_true(std::isfinite(point.x));
            check_true(std::isfinite(point.y));
            check_true(std::isfinite(point.z));
        }
    }
}
