#include "decode.hpp"

#include "kfcore/face_models/error.hpp"
#include "kfcore/face_models/types.hpp"
#include "tinytest.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace kfcore::face_models;

namespace
{

template <typename Callable>
void expect_decode_size_error(Callable&& callable)
{
    bool threw = false;
    try
    {
        callable();
    }
    catch (const FaceModelError& error)
    {
        threw = true;
        check(error.code() == FaceModelErrorCode::RuntimeFailure);
        check(std::string(error.what()).find("element count") != std::string::npos);
    }
    check_true(threw);
}

void expect_face68_finite_error(const std::vector<float>& values)
{
    bool threw = false;
    try
    {
        (void)detail::decode_face68(values.data(), values.size(), 1);
    }
    catch (const FaceModelError& error)
    {
        threw = true;
        const std::string message(error.what());
        check(error.code() == FaceModelErrorCode::RuntimeFailure);
        check(message.find("Face68") != std::string::npos);
        check(message.find("result decoding") != std::string::npos);
        check(message.find("finite") != std::string::npos);
    }
    check_true(threw);
}

template <typename Callable>
void expect_finite_error(Callable&& callable, const char* model_name)
{
    bool threw = false;
    try
    {
        callable();
    }
    catch (const FaceModelError& error)
    {
        threw = true;
        const std::string message(error.what());
        check(error.code() == FaceModelErrorCode::RuntimeFailure);
        check(message.find(model_name) != std::string::npos);
        check(message.find("finite") != std::string::npos);
    }
    check_true(threw);
}

} // namespace

spec("typed face model results")
{
    it("uses fixed arrays for every per-image result")
    {
        static_assert(std::is_same_v<Face68Result,
                                     std::array<Face68Landmark, kFace68LandmarkCount>>);
        static_assert(std::is_same_v<ArcFaceResult,
                                     std::array<float, kArcFaceEmbeddingLength>>);
        static_assert(std::is_same_v<AgeGenderResult,
                                     std::array<float, kAgeGenderLogitCount>>);
        check(kFace68LandmarkCount == std::size_t { 68 });
        check(kArcFaceEmbeddingLength == std::size_t { 512 });
        check(kAgeGenderLogitCount == std::size_t { 2 });
    }

    it("decodes Face68 coordinates into the 256 pixel model system and preserves score")
    {
        std::vector<float> values(kFace68LandmarkCount * 3U, 0.0f);
        values[0] = 10.0f;
        values[1] = 20.0f;
        values[2] = 0.75f;
        const std::vector<Face68Result> results =
            detail::decode_face68(values.data(), values.size(), 1);

        check(results.size() == std::size_t { 1 });
        check(results[0][0].x == 40.0f);
        check(results[0][0].y == 80.0f);
        check(results[0][0].score == 0.75f);
    }

    it("rejects NaN in every Face68 landmark triple field")
    {
        for (std::size_t field = 0; field < 3U; ++field)
        {
            std::vector<float> values(kFace68LandmarkCount * 3U, 0.0f);
            values[field] = (std::numeric_limits<float>::quiet_NaN)();
            expect_face68_finite_error(values);
        }
    }

    it("rejects infinity in every Face68 landmark triple field")
    {
        for (std::size_t field = 0; field < 3U; ++field)
        {
            std::vector<float> values(kFace68LandmarkCount * 3U, 0.0f);
            values[field] = (std::numeric_limits<float>::infinity)();
            expect_face68_finite_error(values);
        }
    }

    it("rejects finite Face68 coordinates that overflow during scaling")
    {
        std::vector<float> values(kFace68LandmarkCount * 3U, 0.0f);
        values[0] = (std::numeric_limits<float>::max)();
        expect_face68_finite_error(values);
    }

    it("copies raw ArcFace embeddings without implicit normalization")
    {
        std::array<float, kArcFaceEmbeddingLength> values {};
        values[0] = 3.0f;
        values[1] = 4.0f;
        const std::vector<ArcFaceResult> results =
            detail::decode_arcface(values.data(), values.size(), 1);

        check(results[0][0] == 3.0f);
        check(results[0][1] == 4.0f);
    }

    it("copies two raw age gender logits without naming their order")
    {
        const std::array<float, 4> values = { -2.0f, 3.0f, 7.0f, 11.0f };
        const std::vector<AgeGenderResult> results =
            detail::decode_age_gender(values.data(), values.size(), 2);

        check(results.size() == std::size_t { 2 });
        check(results[0][0] == -2.0f);
        check(results[0][1] == 3.0f);
        check(results[1][0] == 7.0f);
        check(results[1][1] == 11.0f);
    }

    it("rejects non-finite ArcFace embeddings and age gender logits")
    {
        std::array<float, kArcFaceEmbeddingLength> embedding {};
        embedding[17] = (std::numeric_limits<float>::quiet_NaN)();
        expect_finite_error(
            [&] { (void)detail::decode_arcface(embedding.data(), embedding.size(), 1); },
            "ArcFace");

        std::array<float, kAgeGenderLogitCount> logits {};
        logits[1] = (std::numeric_limits<float>::infinity)();
        expect_finite_error(
            [&] { (void)detail::decode_age_gender(logits.data(), logits.size(), 1); },
            "AgeGender");
    }

    it("rejects short and trailing output storage for every decoder")
    {
        float value = 0.0f;
        expect_decode_size_error(
            [&] { (void)detail::decode_face68(&value, 1, 1); });
        expect_decode_size_error(
            [&] { (void)detail::decode_arcface(&value, kArcFaceEmbeddingLength + 1U, 1); });
        expect_decode_size_error(
            [&] { (void)detail::decode_age_gender(&value, 1, 1); });
    }

    it("reports stable typed face model errors")
    {
        const FaceModelError error(FaceModelErrorCode::InvalidTensorView,
                                   "ArcFace input validation stage: wrong shape");
        check(error.code() == FaceModelErrorCode::InvalidTensorView);
        check(std::string(error.what()) ==
              "ArcFace input validation stage: wrong shape");
    }
}
