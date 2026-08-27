#pragma once

#include "kfcore/image_processor/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kfcore::vision_models
{

inline constexpr std::size_t  kHandLandmarkCount = 21;
inline constexpr std::size_t  kFaceLandmarkCount = 468;
inline constexpr std::int32_t kPalmInputExtent = 192;
inline constexpr std::int32_t kHandLandmarkInputExtent = 224;
inline constexpr std::int32_t kFaceLandmarkInputExtent = 192;
inline constexpr std::size_t  kPalmRowWidth = 8;

struct Point2f
{
    float x = 0.0F;
    float y = 0.0F;
};

struct Point3f
{
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
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

struct HandResult
{
    PalmDetection                            palm;
    std::array<HandLandmark, kHandLandmarkCount> landmarks {};
    float                                    landmark_confidence = 0.0F;
    Handedness                               handedness = Handedness::Unknown;
    Gesture                                  gesture    = Gesture::Unknown;
    int                                      track_id   = -1;
};

struct StageTimings
{
    double preprocess_ms          = 0.0;
    double palm_inference_ms      = 0.0;
    double landmark_inference_ms  = 0.0;
    double classifier_inference_ms = 0.0;
    double tracking_ms            = 0.0;
    double total_ms               = 0.0;
};

struct HandFrame
{
    std::vector<HandResult> hands;
    StageTimings            timings;
};

struct FaceLandmarkResult
{
    std::array<Point3f, kFaceLandmarkCount> landmarks {};
    float                                    confidence = 0.0F;
    double                                   preprocess_ms = 0.0;
    double                                   inference_ms  = 0.0;
    double                                   total_ms      = 0.0;
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

struct HandPipelineOptions
{
    std::size_t     max_hands = 8;
    ByteTrackOptions tracker;
};

} // namespace kfcore::vision_models
