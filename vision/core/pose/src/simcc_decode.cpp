#include "simcc_decode.hpp"

#include "kfcore/pose/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace kfcore::pose::detail
{
namespace
{

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw PoseError(PoseErrorCode::InvalidArgument,
                    "SimCC decode: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw PoseError(PoseErrorCode::ModelContractMismatch,
                    "SimCC decode: " + detail);
}

float half_to_float(std::uint16_t bits) noexcept
{
    const bool negative = (bits & UINT16_C(0x8000)) != 0U;
    const std::uint16_t exponent =
        static_cast<std::uint16_t>((bits >> 10U) & 0x1fU);
    const std::uint16_t fraction =
        static_cast<std::uint16_t>(bits & 0x03ffU);

    float value = 0.0F;
    if (exponent == 0U)
    {
        value = std::ldexp(static_cast<float>(fraction), -24);
    }
    else if (exponent == 0x1fU)
    {
        value = fraction == 0U
                    ? (std::numeric_limits<float>::infinity)()
                    : (std::numeric_limits<float>::quiet_NaN)();
    }
    else
    {
        value = std::ldexp(
            static_cast<float>(UINT16_C(0x0400) + fraction),
            static_cast<int>(exponent) - 25);
    }
    return negative ? -value : value;
}

float tensor_value(const SimccAxisView& axis, std::size_t index)
{
    if (axis.data_type == SimccElementType::Float32)
    {
        return static_cast<const float*>(axis.data)[index];
    }
    return half_to_float(static_cast<const std::uint16_t*>(axis.data)[index]);
}

void validate_axis(const SimccAxisView& axis, const char* name)
{
    if (axis.data == nullptr)
    {
        throw_invalid(std::string(name) + " data must not be null");
    }
    if (axis.keypoints == 0U)
    {
        throw_invalid(std::string(name) + " keypoint count must be positive");
    }
    if (axis.bins == 0U)
    {
        throw_invalid(std::string(name) + " bin count must be positive");
    }
}

} // namespace

void decode_simcc(const SimccAxisView& x_axis,
                  const SimccAxisView& y_axis,
                  float split_ratio,
                  DecodedSimccKeypoint* output,
                  std::size_t output_count)
{
    validate_axis(x_axis, "X");
    validate_axis(y_axis, "Y");

    if (!std::isfinite(split_ratio) || split_ratio <= 0.0F)
    {
        throw_invalid("split ratio must be finite and positive");
    }
    if (x_axis.keypoints != y_axis.keypoints)
    {
        throw_invalid("X/Y keypoint counts must match");
    }
    if (output == nullptr)
    {
        throw_invalid("output storage must not be null");
    }
    if (output_count < x_axis.keypoints)
    {
        throw_invalid("output storage is smaller than the keypoint count");
    }

    for (std::size_t keypoint = 0U; keypoint < x_axis.keypoints; ++keypoint)
    {
        const std::size_t x_base = keypoint * x_axis.bins;
        const std::size_t y_base = keypoint * y_axis.bins;

        float max_x = -(std::numeric_limits<float>::infinity)();
        float max_y = -(std::numeric_limits<float>::infinity)();
        std::size_t x_index = 0U;
        std::size_t y_index = 0U;

        for (std::size_t index = 0U; index < x_axis.bins; ++index)
        {
            const float value = tensor_value(x_axis, x_base + index);
            if (value > max_x)
            {
                max_x = value;
                x_index = index;
            }
        }
        for (std::size_t index = 0U; index < y_axis.bins; ++index)
        {
            const float value = tensor_value(y_axis, y_base + index);
            if (value > max_y)
            {
                max_y = value;
                y_index = index;
            }
        }

        if (!std::isfinite(max_x) || !std::isfinite(max_y))
        {
            throw_contract("output contains a non-finite maximum response");
        }

        DecodedSimccKeypoint decoded;
        decoded.confidence = (std::min)(max_x, max_y);
        if (decoded.confidence > 0.0F)
        {
            decoded.x = static_cast<float>(x_index) / split_ratio;
            decoded.y = static_cast<float>(y_index) / split_ratio;
        }
        output[keypoint] = decoded;
    }
}

} // namespace kfcore::pose::detail
