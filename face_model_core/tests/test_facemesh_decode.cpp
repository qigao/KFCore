#include "facemesh_decode.hpp"
#include "facemesh_geometry.hpp"

#include "kfcore/face_models/error.hpp"
#include "tinytest.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

using namespace kfcore::face_models;

namespace
{

constexpr float kTolerance = 1.0e-4F;

void check_close(float actual, float expected)
{
    check_true(std::fabs(actual - expected) <= kTolerance);
}

void check_error(const std::function<void()>& operation, const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const FaceModelError& error)
    {
        threw = true;
        check(error.code() == FaceModelErrorCode::ModelContractMismatch);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

spec("face mesh output decoding")
{
    it("maps all 468 landmarks with the same face ROI")
    {
        const detail::FaceRoi roi = detail::make_face_roi(
            { 100.0F, 50.0F, 100.0F, 100.0F }, 192);
        std::vector<float> values(kFaceMeshLandmarkCount * 3U, 0.0F);
        values[0] = 95.5F;
        values[1] = 95.5F;
        values[2] = 10.0F;

        const auto landmarks = detail::decode_face_landmarks(
            values.data(), values.size(), roi, 192, false);

        check_close(landmarks[0].x, 150.0F);
        check_close(landmarks[0].y, 95.0F);
        check_close(landmarks[0].z, 135.0F * 10.0F / 192.0F);
    }

    it("selects the highest-score YOLOv12 face and restores coordinates")
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

    it("returns every qualifying YOLOv12 face in stable model order")
    {
        const std::array<float, 18> rows {
            100.0F, 120.0F, 300.0F, 360.0F, 0.75F, 0.0F,
             90.0F, 110.0F, 310.0F, 370.0F, 0.90F, 0.0F,
             10.0F,  20.0F,  30.0F,  40.0F, 0.99F, 1.0F,
        };
        const kfcore::image::LetterboxTransform letterbox {
            0.5F, 10.0F, 20.0F, 640, 480
        };

        const auto faces = detail::decode_yolo12_faces(
            rows.data(), rows.size(), 0, 0.50F, letterbox, 640, 480);

        check(faces.size() == 2U);
        check_close(faces[0].confidence, 0.75F);
        check_close(faces[1].confidence, 0.90F);
        check_close(faces[0].box.x, 180.0F);
        check_close(faces[1].box.x, 160.0F);
    }

    it("filters unmatched face rows")
    {
        const std::array<float, 12> rows {
            10.0F, 10.0F, 20.0F, 20.0F, 0.49F, 0.0F,
            30.0F, 30.0F, 50.0F, 50.0F, 0.99F, 1.0F,
        };
        const kfcore::image::LetterboxTransform letterbox {
            1.0F, 0.0F, 0.0F, 640, 480
        };
        check_false(detail::decode_yolo12_face(
            rows.data(), rows.size(), 0, 0.50F, letterbox, 640, 480).has_value());
    }

    it("rejects malformed, non-finite, and invalid boxes")
    {
        std::array<float, 6> row {
            10.0F, 10.0F, 20.0F, 20.0F, 0.90F, 0.0F
        };
        const kfcore::image::LetterboxTransform letterbox {
            1.0F, 0.0F, 0.0F, 640, 480
        };
        check_error([&] {
            (void)detail::decode_yolo12_face(
                row.data(), row.size() - 1U, 0, 0.50F, letterbox, 640, 480);
        }, "rows of six");

        row[4] = (std::numeric_limits<float>::quiet_NaN)();
        check_error([&] {
            (void)detail::decode_yolo12_face(
                row.data(), row.size(), 0, 0.50F, letterbox, 640, 480);
        }, "finite");

        row = { 20.0F, 10.0F, 10.0F, 20.0F, 0.90F, 0.0F };
        check_error([&] {
            (void)detail::decode_yolo12_face(
                row.data(), row.size(), 0, 0.50F, letterbox, 640, 480);
        }, "box");
    }

    it("rejects a letterbox transform from another source image")
    {
        const std::array<float, 6> row {
            10.0F, 10.0F, 20.0F, 20.0F, 0.90F, 0.0F
        };
        const kfcore::image::LetterboxTransform letterbox {
            1.0F, 0.0F, 0.0F, 320, 240
        };
        check_error([&] {
            (void)detail::decode_yolo12_face(
                row.data(), row.size(), 0, 0.50F, letterbox, 640, 480);
        }, "source dimensions");
    }
}
