#pragma once

#include "kfcore/image_processor/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace kfcore::hand_models
{

inline constexpr std::size_t  kHandLandmarkCount = 21;
inline constexpr std::size_t  kHandAppearancePartCount = 6;
inline constexpr std::size_t  kPalmAppearanceFeatureCount = 96;
inline constexpr std::size_t  kFingerAppearanceFeatureCount = 32;
inline constexpr std::size_t  kHandAppearanceFeatureCount =
    kPalmAppearanceFeatureCount + 5U * kFingerAppearanceFeatureCount;
inline constexpr std::int32_t kPalmInputExtent = 192;
inline constexpr std::int32_t kHandLandmarkInputExtent = 224;
inline constexpr std::size_t  kPalmRowWidth = 8;

struct Point2f
{
    float x = 0.0F;
    float y = 0.0F;
};

struct RectF
{
    float x      = 0.0F;
    float y      = 0.0F;
    float width  = 0.0F;
    float height = 0.0F;
};

struct RotatedRoi
{
    Point2f center;
    float   size             = 0.0F;
    float   rotation_radians = 0.0F;
};

struct PalmDetection
{
    RectF      box;
    RotatedRoi roi;
    Point2f    wrist_keypoint;
    Point2f    middle_finger_keypoint;
    float      confidence = 0.0F;
};

struct HandLandmark
{
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};

enum class Handedness
{
    Unknown,
    Left,
    Right,
};

enum class Gesture
{
    Unknown,
    Open,
    Closed,
    Pointer,
};

enum class HandAppearancePart : std::uint8_t
{
    Palm = 0,
    Thumb,
    Index,
    Middle,
    Ring,
    Pinky,
};

inline constexpr std::uint8_t kAllHandAppearanceParts =
    static_cast<std::uint8_t>((1U << kHandAppearancePartCount) - 1U);

[[nodiscard]] inline constexpr std::uint8_t
hand_appearance_part_bit(HandAppearancePart part) noexcept
{
    return static_cast<std::uint8_t>(1U << static_cast<std::uint8_t>(part));
}

[[nodiscard]] inline constexpr std::size_t
hand_appearance_feature_offset(HandAppearancePart part) noexcept
{
    const std::size_t index = static_cast<std::size_t>(part);
    return index == 0U
        ? 0U
        : kPalmAppearanceFeatureCount + (index - 1U) * kFingerAppearanceFeatureCount;
}

[[nodiscard]] inline constexpr std::size_t
hand_appearance_feature_count(HandAppearancePart part) noexcept
{
    return part == HandAppearancePart::Palm
        ? kPalmAppearanceFeatureCount
        : kFingerAppearanceFeatureCount;
}

struct HandAppearanceDescriptor
{
    std::array<float, kHandAppearanceFeatureCount> values {};
    std::uint8_t valid_parts = 0U;
    std::array<float, kHandAppearancePartCount> quality {};
};

struct HandResult
{
    PalmDetection                            palm;
    std::array<HandLandmark, kHandLandmarkCount> landmarks {};
    float                                    landmark_confidence = 0.0F;
    Handedness                               handedness = Handedness::Unknown;
    Gesture                                  gesture    = Gesture::Unknown;
    int                                      track_id   = -1;
    std::optional<HandAppearanceDescriptor>  appearance;
};

struct StageTimings
{
    double preprocess_ms           = 0.0;
    double palm_inference_ms       = 0.0;
    double landmark_inference_ms   = 0.0;
    double classifier_inference_ms = 0.0;
    double appearance_ms           = 0.0;
    double tracking_ms             = 0.0;
    double total_ms                = 0.0;
};

struct HandFrame
{
    std::vector<HandResult> hands;
    StageTimings            timings;
};

struct ByteTrackOptions
{
    int   lost_track_buffer          = 30;
    float frame_rate                 = 30.0F;
    float track_activation_threshold = 0.7F;
    int   minimum_consecutive_frames = 2;
    float minimum_iou_threshold      = 0.1F;
    float high_confidence_threshold  = 0.6F;
};

struct HandAppearanceOptions
{
    bool  enabled                            = false;
    float minimum_palm_span_pixels           = 8.0F;
    float minimum_part_in_frame_sample_ratio = 0.75F;
};

struct HandTrackingOptions
{
    std::size_t max_hands = 8;
    ByteTrackOptions tracker;
    HandAppearanceOptions appearance;
};

} // namespace kfcore::hand_models
