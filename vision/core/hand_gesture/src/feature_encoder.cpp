#include "kfcore/hand_gesture/feature_encoder.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace kfcore::hand_gesture
{
namespace
{

constexpr std::size_t kIndexMcp = 5U;
constexpr std::size_t kMiddleMcp = 9U;
constexpr std::size_t kRingMcp = 13U;
constexpr std::size_t kPinkyMcp = 17U;
constexpr std::array<std::size_t, 4U> kPalmMcpIndices = {
    kIndexMcp, kMiddleMcp, kRingMcp, kPinkyMcp};
constexpr double kNanosecondsPerSecond = 1'000'000'000.0;

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw std::invalid_argument("temporal gesture feature encoding: " + detail);
}

void require_finite(float value, const char* name)
{
    if (!std::isfinite(value))
    {
        throw_invalid(std::string(name) + " must be finite");
    }
}

void require_unit(float value, const char* name)
{
    require_finite(value, name);
    if (value < 0.0F || value > 1.0F)
    {
        throw_invalid(std::string(name) + " must be within [0,1]");
    }
}

float distance_xy(const hand_models::HandLandmark& first,
                  const hand_models::HandLandmark& second)
{
    const float dx = second.x - first.x;
    const float dy = second.y - first.y;
    return std::sqrt(dx * dx + dy * dy);
}

float palm_scale(const hand_models::HandResult& hand)
{
    const auto& wrist = hand.landmarks[0];
    float total = 0.0F;
    for (const std::size_t index : kPalmMcpIndices)
    {
        const float value = distance_xy(wrist, hand.landmarks[index]);
        require_finite(value, "palm scale component");
        total += value;
    }
    const float scale = total / static_cast<float>(kPalmMcpIndices.size());
    if (!std::isfinite(scale) || scale <= 0.0F)
    {
        throw_invalid("palm scale must be positive");
    }
    return scale;
}

float handedness_value(hand_models::Handedness handedness) noexcept
{
    switch (handedness)
    {
    case hand_models::Handedness::Left: return -1.0F;
    case hand_models::Handedness::Right: return 1.0F;
    default: return 0.0F;
    }
}

std::size_t static_pose_index(hand_models::Gesture gesture) noexcept
{
    switch (gesture)
    {
    case hand_models::Gesture::Open: return 1U;
    case hand_models::Gesture::Closed: return 2U;
    case hand_models::Gesture::Pointer: return 3U;
    default: return 0U;
    }
}

} // namespace

EncodedGestureFeatures GestureFeatureEncoder::encode(
    const hand_models::HandResult& hand,
    const GestureFrameMetadata& metadata,
    const std::optional<GestureFeatureState>& previous)
{
    if (metadata.image_width <= 0 || metadata.image_height <= 0)
    {
        throw_invalid("image dimensions must be positive");
    }

    require_unit(hand.landmark_confidence, "landmark confidence");
    require_unit(hand.palm.confidence, "palm confidence");
    require_finite(hand.palm.roi.rotation_radians, "palm ROI rotation");

    for (const auto& landmark : hand.landmarks)
    {
        require_finite(landmark.x, "landmark x");
        require_finite(landmark.y, "landmark y");
        require_finite(landmark.z, "landmark z");
    }

    const float scale = palm_scale(hand);
    const auto& wrist = hand.landmarks[0];
    const auto& middle_mcp = hand.landmarks[kMiddleMcp];
    const float orientation_x = middle_mcp.x - wrist.x;
    const float orientation_y = middle_mcp.y - wrist.y;
    if (orientation_x == 0.0F && orientation_y == 0.0F)
    {
        throw_invalid("palm orientation vector must be non-zero");
    }
    const float orientation = std::atan2(orientation_y, orientation_x);

    const float width = static_cast<float>(metadata.image_width);
    const float height = static_cast<float>(metadata.image_height);
    const float image_diagonal = std::sqrt(width * width + height * height);
    if (!std::isfinite(image_diagonal) || image_diagonal <= 0.0F)
    {
        throw_invalid("image diagonal must be positive");
    }

    const float wrist_x = wrist.x / width;
    const float wrist_y = wrist.y / height;
    require_finite(wrist_x, "normalized wrist x");
    require_finite(wrist_y, "normalized wrist y");

    float delta_seconds = 0.0F;
    float velocity_x = 0.0F;
    float velocity_y = 0.0F;
    if (previous)
    {
        if (metadata.timestamp_ns <= previous->timestamp_ns)
        {
            throw_invalid("timestamp must increase when previous state is supplied");
        }
        delta_seconds = static_cast<float>(
            static_cast<double>(metadata.timestamp_ns - previous->timestamp_ns) /
            kNanosecondsPerSecond);
        if (!std::isfinite(delta_seconds) || delta_seconds <= 0.0F)
        {
            throw_invalid("delta time must be positive and finite");
        }
        velocity_x = (wrist_x - previous->wrist_x_normalized) / delta_seconds;
        velocity_y = (wrist_y - previous->wrist_y_normalized) / delta_seconds;
        require_finite(velocity_x, "wrist velocity x");
        require_finite(velocity_y, "wrist velocity y");
    }

    EncodedGestureFeatures result;
    std::size_t cursor = 0U;
    const float mirror = hand.handedness == hand_models::Handedness::Left ? -1.0F : 1.0F;
    for (const auto& landmark : hand.landmarks)
    {
        result.values[cursor++] = mirror * (landmark.x - wrist.x) / scale;
        result.values[cursor++] = (landmark.y - wrist.y) / scale;
        result.values[cursor++] = (landmark.z - wrist.z) / scale;
    }

    result.values[cursor++] = wrist_x;
    result.values[cursor++] = wrist_y;
    result.values[cursor++] = velocity_x;
    result.values[cursor++] = velocity_y;
    result.values[cursor++] = scale / image_diagonal;
    result.values[cursor++] = std::sin(orientation);
    result.values[cursor++] = std::cos(orientation);
    result.values[cursor++] = handedness_value(hand.handedness);
    result.values[cursor++] = hand.landmark_confidence;
    result.values[cursor++] = hand.palm.confidence;
    result.values[cursor++] = delta_seconds;

    const std::size_t pose = static_pose_index(hand.gesture);
    for (std::size_t index = 0U; index < 4U; ++index)
    {
        result.values[cursor++] = index == pose ? 1.0F : 0.0F;
    }

    if (cursor != kTemporalGestureFeatureCount)
    {
        throw std::logic_error("temporal gesture feature encoder produced wrong feature count");
    }
    for (float value : result.values)
    {
        require_finite(value, "encoded feature");
    }

    result.next_state.timestamp_ns = metadata.timestamp_ns;
    result.next_state.wrist_x_normalized = wrist_x;
    result.next_state.wrist_y_normalized = wrist_y;
    return result;
}

} // namespace kfcore::hand_gesture
