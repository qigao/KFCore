#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kfcore::hand_gesture
{

inline constexpr std::size_t kTemporalGestureFeatureCount = 78U;
inline constexpr std::size_t kTemporalGestureHiddenLayers = 2U;
inline constexpr std::size_t kTemporalGestureHiddenSize = 64U;
inline constexpr std::size_t kTemporalGestureHiddenElementCount =
    kTemporalGestureHiddenLayers * kTemporalGestureHiddenSize;
inline constexpr std::size_t kTemporalGestureClassCount = 8U;
inline constexpr std::size_t kTemporalGesturePhaseCount = 4U;

enum class GestureClass : std::uint8_t
{
    None = 0,
    Wave = 1,
    SwipeLeft = 2,
    SwipeRight = 3,
    Grab = 4,
    Release = 5,
    Point = 6,
    Click = 7,
};

enum class GesturePhase : std::uint8_t
{
    Idle = 0,
    Start = 1,
    Active = 2,
    End = 3,
};

struct GestureEvent
{
    int track_id = -1;
    GestureClass gesture = GestureClass::None;
    GesturePhase phase = GesturePhase::Idle;
    float confidence = 0.0F;
    std::uint64_t timestamp_ns = 0U;
};

struct GestureFrameMetadata
{
    std::uint64_t timestamp_ns = 0U;
    int image_width = 0;
    int image_height = 0;
};

struct GestureFeatureState
{
    std::uint64_t timestamp_ns = 0U;
    float wrist_x_normalized = 0.0F;
    float wrist_y_normalized = 0.0F;
};

using GestureFeatureVector = std::array<float, kTemporalGestureFeatureCount>;
using GestureHiddenState = std::array<float, kTemporalGestureHiddenElementCount>;

struct EncodedGestureFeatures
{
    GestureFeatureVector values {};
    GestureFeatureState next_state {};
};

} // namespace kfcore::hand_gesture
