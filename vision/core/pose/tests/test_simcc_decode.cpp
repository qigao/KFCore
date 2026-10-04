#include "simcc_decode.hpp"

#include "kfcore/pose/error.hpp"
#include "tinytest.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

using namespace kfcore::pose;
using namespace kfcore::pose::detail;

spec("SimCC decode")
{
    it("decodes argmax coordinates and uses the smaller axis maximum as confidence")
    {
        const std::array<float, 8> x {{
            -1.0F, 0.5F, 3.0F, 1.0F,
            0.0F, 2.0F, 1.0F, 0.0F,
        }};
        const std::array<float, 10> y {{
            0.0F, 4.0F, 1.0F, 0.0F, -1.0F,
            0.0F, 1.0F, 5.0F, 2.0F, 0.0F,
        }};
        std::array<DecodedSimccKeypoint, 2> output {};

        decode_simcc(
            {x.data(), SimccElementType::Float32, 2U, 4U},
            {y.data(), SimccElementType::Float32, 2U, 5U},
            2.0F, output.data(), output.size());

        check(std::fabs(output[0].x - 1.0F) < 1.0e-6F);
        check(std::fabs(output[0].y - 0.5F) < 1.0e-6F);
        check(std::fabs(output[0].confidence - 3.0F) < 1.0e-6F);

        check(std::fabs(output[1].x - 0.5F) < 1.0e-6F);
        check(std::fabs(output[1].y - 1.0F) < 1.0e-6F);
        check(std::fabs(output[1].confidence - 2.0F) < 1.0e-6F);
    }

    it("keeps coordinates invalid when the canonical confidence is non-positive")
    {
        const std::array<float, 3> x {{-2.0F, 0.0F, -1.0F}};
        const std::array<float, 3> y {{1.0F, 2.0F, 3.0F}};
        DecodedSimccKeypoint output;

        decode_simcc(
            {x.data(), SimccElementType::Float32, 1U, 3U},
            {y.data(), SimccElementType::Float32, 1U, 3U},
            2.0F, &output, 1U);

        check(output.confidence == 0.0F);
        check(output.x == -1.0F);
        check(output.y == -1.0F);
    }

    it("decodes FP16 inputs without changing score semantics")
    {
        const std::array<std::uint16_t, 4> x {{
            UINT16_C(0x0000),
            UINT16_C(0x3c00),
            UINT16_C(0x4000),
            UINT16_C(0x3c00),
        }};
        const std::array<std::uint16_t, 4> y {{
            UINT16_C(0x0000),
            UINT16_C(0x3c00),
            UINT16_C(0x3c00),
            UINT16_C(0x4200),
        }};
        DecodedSimccKeypoint output;

        decode_simcc(
            {x.data(), SimccElementType::Float16, 1U, 4U},
            {y.data(), SimccElementType::Float16, 1U, 4U},
            2.0F, &output, 1U);

        check(std::fabs(output.x - 1.0F) < 1.0e-6F);
        check(std::fabs(output.y - 1.5F) < 1.0e-6F);
        check(std::fabs(output.confidence - 2.0F) < 1.0e-6F);
    }

    it("decodes visibility from independently scaled softmax peaks")
    {
        const std::array<float, 3> x {{0.0F, 1.0F, 2.0F}};
        const std::array<float, 3> y {{0.0F, 0.0F, 1.0F}};
        float visibility = 0.0F;

        decode_simcc_visibility(
            {x.data(), SimccElementType::Float32, 1U, 3U},
            {y.data(), SimccElementType::Float32, 1U, 3U},
            {2.0F, 1.0F, 0.5F},
            &visibility, 1U);

        check(std::fabs(visibility - 0.57611686F) < 1.0e-6F);
    }

    it("matches the released RTMW visibility scaling parameters")
    {
        const std::array<float, 3> x {{0.0F, 0.001F, 0.002F}};
        const std::array<float, 3> y {{0.0F, 0.0015F, 0.0025F}};
        float visibility = 0.0F;

        decode_simcc_visibility(
            {x.data(), SimccElementType::Float32, 1U, 3U},
            {y.data(), SimccElementType::Float32, 1U, 3U},
            {150.0F, 6.0F, 6.93F},
            &visibility, 1U);

        check(std::fabs(visibility - 0.6361855F) < 1.0e-5F);
    }

    it("keeps visibility numerically separate from raw confidence")
    {
        const std::array<float, 3> x {{0.0F, 1.0F, 2.0F}};
        const std::array<float, 3> y {{0.0F, 0.0F, 1.0F}};
        DecodedSimccKeypoint decoded;
        float visibility = 0.0F;

        decode_simcc(
            {x.data(), SimccElementType::Float32, 1U, 3U},
            {y.data(), SimccElementType::Float32, 1U, 3U},
            2.0F, &decoded, 1U);
        decode_simcc_visibility(
            {x.data(), SimccElementType::Float32, 1U, 3U},
            {y.data(), SimccElementType::Float32, 1U, 3U},
            {2.0F, 1.0F, 0.5F},
            &visibility, 1U);

        check(decoded.confidence == 1.0F);
        check(std::fabs(visibility - decoded.confidence) > 0.4F);
    }

    it("rejects invalid visibility parameters and non-finite responses")
    {
        const std::array<float, 2> values {{1.0F, 2.0F}};
        float visibility = 0.0F;

        check_throws_as(
            decode_simcc_visibility(
                {values.data(), SimccElementType::Float32, 1U, 2U},
                {values.data(), SimccElementType::Float32, 1U, 2U},
                {0.0F, 1.0F, 1.0F},
                &visibility, 1U),
            PoseError);

        const float nan = (std::numeric_limits<float>::quiet_NaN)();
        const std::array<float, 2> invalid {{nan, 1.0F}};
        check_throws_as(
            decode_simcc_visibility(
                {invalid.data(), SimccElementType::Float32, 1U, 2U},
                {values.data(), SimccElementType::Float32, 1U, 2U},
                {150.0F, 6.0F, 6.93F},
                &visibility, 1U),
            PoseError);
    }

    it("rejects malformed axis and output contracts")
    {
        const std::array<float, 2> values {{1.0F, 2.0F}};
        DecodedSimccKeypoint output;

        check_throws_as(
            decode_simcc(
                {values.data(), SimccElementType::Float32, 1U, 2U},
                {values.data(), SimccElementType::Float32, 2U, 1U},
                2.0F, &output, 1U),
            PoseError);

        check_throws_as(
            decode_simcc(
                {values.data(), SimccElementType::Float32, 1U, 2U},
                {values.data(), SimccElementType::Float32, 1U, 2U},
                0.0F, &output, 1U),
            PoseError);

        check_throws_as(
            decode_simcc(
                {values.data(), SimccElementType::Float32, 1U, 2U},
                {values.data(), SimccElementType::Float32, 1U, 2U},
                2.0F, nullptr, 0U),
            PoseError);
    }

    it("fails closed on non-finite maximum responses")
    {
        const float nan = (std::numeric_limits<float>::quiet_NaN)();
        const std::array<float, 2> x {{nan, nan}};
        const std::array<float, 2> y {{1.0F, 2.0F}};
        DecodedSimccKeypoint output;

        check_throws_as(
            decode_simcc(
                {x.data(), SimccElementType::Float32, 1U, 2U},
                {y.data(), SimccElementType::Float32, 1U, 2U},
                2.0F, &output, 1U),
            PoseError);
    }
}
