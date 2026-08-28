#include "decode.hpp"
#include "geometry.hpp"
#include "kfcore/vision_models/error.hpp"
#include "kfcore/vision_models/types.hpp"
#include "tinytest.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

using namespace kfcore::vision_models;

namespace
{

constexpr float kTolerance = 1.0e-4F;

void check_close(float actual, float expected)
{
    check_true(std::fabs(actual - expected) <= kTolerance);
}

void check_error(const std::function<void()>& operation, VisionModelErrorCode code,
                 const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const VisionModelError& error)
    {
        threw = true;
        check(error.code() == code);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

spec("vision model output decoding")
{
    it("maps hand landmarks and z scale through the shared ROI transform")
    {
        const RotatedRoi roi { { 100.0F, 80.0F }, 56.0F, 0.0F };
        std::array<float, kHandLandmarkCount * 3U> xyz {};
        for (std::size_t point = 0; point < kHandLandmarkCount; ++point)
        {
            xyz[point * 3U]     = 111.5F;
            xyz[point * 3U + 1] = 111.5F;
            xyz[point * 3U + 2] = 8.0F;
        }

        const auto landmarks = detail::decode_hand_landmarks(xyz.data(), xyz.size(), roi, 224);

        check_close(landmarks[0].x, 100.0F);
        check_close(landmarks[0].y, 80.0F);
        check_close(landmarks[0].z, 2.0F);
    }

    it("normalizes classifier features relative to the wrist and maximum magnitude")
    {
        std::array<HandLandmark, kHandLandmarkCount> landmarks {};
        for (HandLandmark& landmark : landmarks)
        {
            landmark = { 10.0F, 20.0F, 0.0F };
        }
        landmarks[1] = { 14.0F, 18.0F, 0.0F };

        const auto features = detail::make_keypoint_features(landmarks);

        check_close(features[0], 0.0F);
        check_close(features[1], 0.0F);
        check_close(features[2], 1.0F);
        check_close(features[3], -0.5F);
    }

    it("maps known classifier ids and keeps unknown ids bounded")
    {
        check(detail::decode_gesture(0) == Gesture::Open);
        check(detail::decode_gesture(1) == Gesture::Closed);
        check(detail::decode_gesture(2) == Gesture::Pointer);
        check(detail::decode_gesture(99) == Gesture::Unknown);
    }

    it("sorts Palm candidates by confidence and applies independent hand limits")
    {
        const std::array<float, kPalmRowWidth * 2U> rows {
            0.7F, 0.25F, 0.5F, 0.1F, 0.25F, 0.55F, 0.25F, 0.45F,
            0.9F, 0.75F, 0.5F, 0.1F, 0.75F, 0.55F, 0.75F, 0.45F,
        };
        const kfcore::image::LetterboxTransform letterbox { 1.0F, 0.0F, 0.0F,
                                                            192, 192 };

        const auto palms = detail::decode_palms(rows.data(), rows.size(), 0.5F, 2, 1,
                                                letterbox, 192);

        check(palms.size() == 1);
        check_close(palms[0].confidence, 0.9F);
    }

    it("rejects malformed and over-capacity Palm outputs")
    {
        const std::array<float, kPalmRowWidth * 2U> rows {};
        const kfcore::image::LetterboxTransform letterbox { 1.0F, 0.0F, 0.0F,
                                                            192, 192 };
        check_error(
            [&]
            {
                (void)detail::decode_palms(rows.data(), rows.size() - 1U, 0.5F, 2, 1,
                                           letterbox, 192);
            },
            VisionModelErrorCode::ModelContractMismatch, "rows of eight");
        check_error(
            [&]
            {
                (void)detail::decode_palms(rows.data(), rows.size(), 0.5F, 1, 1,
                                           letterbox, 192);
            },
            VisionModelErrorCode::ResourceLimitExceeded, "candidates");
    }

    it("maps all 468 face landmarks with the same face ROI")
    {
        const detail::FaceRoi roi = detail::make_face_roi({ 100.0F, 50.0F, 100.0F, 100.0F },
                                                          192);
        std::vector<float> values(kFaceLandmarkCount * 3U, 0.0F);
        values[0] = 95.5F;
        values[1] = 95.5F;
        values[2] = 10.0F;

        const auto landmarks = detail::decode_face_landmarks(
            values.data(), values.size(), roi, 192, false);

        check_close(landmarks[0].x, 150.0F);
        check_close(landmarks[0].y, 95.0F);
        check_close(landmarks[0].z, 135.0F * 10.0F / 192.0F);
    }

    it("selects the highest-score YOLOv12 face and restores letterbox coordinates")
    {
        const std::array<float, 12> rows {
            100.0F, 120.0F, 300.0F, 360.0F, 0.75F, 0.0F,
             90.0F, 110.0F, 310.0F, 370.0F, 0.90F, 0.0F,
        };
        const kfcore::image::LetterboxTransform letterbox {
            0.5F, 10.0F, 20.0F, 640, 480
        };

        const auto face = detail::decode_yolo12_face(
            rows.data(), rows.size(), 0, 0.50F, letterbox, 640, 480);

        check_true(face.has_value());
        check_close(face->box.x, 160.0F);
        check_close(face->box.y, 180.0F);
        check_close(face->box.width, 440.0F);
        check_close(face->box.height, 300.0F);
        check_close(face->confidence, 0.90F);
    }

    it("returns no YOLOv12 face when class and score filters reject every row")
    {
        const std::array<float, 12> rows {
            10.0F, 10.0F, 20.0F, 20.0F, 0.49F, 0.0F,
            30.0F, 30.0F, 50.0F, 50.0F, 0.99F, 1.0F,
        };
        const kfcore::image::LetterboxTransform letterbox {
            1.0F, 0.0F, 0.0F, 640, 480
        };

        const auto face = detail::decode_yolo12_face(
            rows.data(), rows.size(), 0, 0.50F, letterbox, 640, 480);

        check_false(face.has_value());
    }

    it("rejects malformed and non-finite YOLOv12 face rows")
    {
        std::array<float, 6> row {
            10.0F, 10.0F, 20.0F, 20.0F, 0.90F, 0.0F
        };
        const kfcore::image::LetterboxTransform letterbox {
            1.0F, 0.0F, 0.0F, 640, 480
        };
        check_error(
            [&] {
                (void)detail::decode_yolo12_face(
                    row.data(), row.size() - 1U, 0, 0.50F, letterbox, 640, 480);
            },
            VisionModelErrorCode::ModelContractMismatch, "rows of six");

        row[4] = (std::numeric_limits<float>::quiet_NaN)();
        check_error(
            [&] {
                (void)detail::decode_yolo12_face(
                    row.data(), row.size(), 0, 0.50F, letterbox, 640, 480);
            },
            VisionModelErrorCode::ModelContractMismatch, "finite");
    }

    it("rejects invalid YOLOv12 face boxes before filtering")
    {
        const std::array<float, 6> row {
            20.0F, 10.0F, 10.0F, 20.0F, 0.90F, 0.0F
        };
        const kfcore::image::LetterboxTransform letterbox {
            1.0F, 0.0F, 0.0F, 640, 480
        };
        check_error(
            [&] {
                (void)detail::decode_yolo12_face(
                    row.data(), row.size(), 0, 0.50F, letterbox, 640, 480);
            },
            VisionModelErrorCode::ModelContractMismatch, "box");
    }

    it("rejects a YOLOv12 face letterbox transform from another source image")
    {
        const std::array<float, 6> row {
            10.0F, 10.0F, 20.0F, 20.0F, 0.90F, 0.0F
        };
        const kfcore::image::LetterboxTransform letterbox {
            1.0F, 0.0F, 0.0F, 320, 240
        };
        check_error(
            [&] {
                (void)detail::decode_yolo12_face(
                    row.data(), row.size(), 0, 0.50F, letterbox, 640, 480);
            },
            VisionModelErrorCode::ModelContractMismatch, "source dimensions");
    }
}
