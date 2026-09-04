#include "decode.hpp"
#include "geometry.hpp"
#include "kfcore/hand_models/error.hpp"
#include "kfcore/hand_models/types.hpp"
#include "tinytest.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

using namespace kfcore::hand_models;

namespace
{

constexpr float kTolerance = 1.0e-4F;

void check_close(float actual, float expected)
{
    check_true(std::fabs(actual - expected) <= kTolerance);
}

void check_error(const std::function<void()>& operation, HandModelErrorCode code,
                 const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const HandModelError& error)
    {
        threw = true;
        check(error.code() == code);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

spec("hand model output decoding")
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
            HandModelErrorCode::ModelContractMismatch, "rows of eight");
        check_error(
            [&]
            {
                (void)detail::decode_palms(rows.data(), rows.size(), 0.5F, 1, 1,
                                           letterbox, 192);
            },
            HandModelErrorCode::ResourceLimitExceeded, "candidates");
    }

}
