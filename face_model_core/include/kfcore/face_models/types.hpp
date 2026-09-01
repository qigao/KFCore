#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace kfcore::face_models
{

inline constexpr std::size_t kFace68LandmarkCount    = 68;
inline constexpr std::size_t kArcFaceEmbeddingLength = 512;
inline constexpr std::size_t kAgeGenderLogitCount    = 2;
inline constexpr std::size_t kInSwapperEmbeddingLength = 512;
inline constexpr std::int64_t kFaceModelInputChannels = 3;
inline constexpr std::int64_t kFace68InputExtent      = 256;
inline constexpr std::int64_t kFace68LandmarkWidth    = 3;
inline constexpr std::int64_t kFace68HeatmapExtent    = 64;
inline constexpr std::int64_t kArcFaceInputExtent     = 112;
inline constexpr std::int64_t kAgeGenderInputExtent   = 224;
inline constexpr std::int64_t kInSwapperInputExtent   = 128;
inline constexpr std::int64_t kGfpGanInputExtent      = 512;
inline constexpr std::size_t  kInSwapperOutputElementCount =
    static_cast<std::size_t>(kFaceModelInputChannels * kInSwapperInputExtent *
                             kInSwapperInputExtent);
inline constexpr std::size_t kGfpGanOutputElementCount =
    static_cast<std::size_t>(kFaceModelInputChannels * kGfpGanInputExtent *
                             kGfpGanInputExtent);
inline constexpr std::size_t kLargeFaceModelMaxAssetBytes =
    1024U * 1024U * 1024U;
inline constexpr std::size_t kLargeFaceModelMaxSerializedEngineBytes =
    kLargeFaceModelMaxAssetBytes;

inline constexpr std::size_t  kFaceMeshLandmarkCount = 468;
inline constexpr std::int32_t kFaceMeshInputExtent   = 192;

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

struct FaceDetection
{
    RectF box;
    float confidence = 0.0F;
};

struct FaceDetectionResult
{
    std::optional<FaceDetection> face;
    double                       preprocess_ms = 0.0;
    double                       inference_ms  = 0.0;
    double                       total_ms      = 0.0;
};

struct FaceDetectionsResult
{
    std::vector<FaceDetection> faces;
    double                     preprocess_ms = 0.0;
    double                     inference_ms  = 0.0;
    double                     total_ms      = 0.0;
};

struct FaceLandmarkResult
{
    std::array<Point3f, kFaceMeshLandmarkCount> landmarks {};
    float                                       confidence    = 0.0F;
    double                                      preprocess_ms = 0.0;
    double                                      inference_ms  = 0.0;
    double                                      total_ms      = 0.0;
};

struct FaceMeshTimings
{
    double detection_preprocess_ms = 0.0;
    double detection_inference_ms  = 0.0;
    double landmark_preprocess_ms  = 0.0;
    double landmark_inference_ms   = 0.0;
    double total_ms                = 0.0;
};

struct FaceMeshFrame
{
    std::optional<FaceDetection>      detection;
    std::optional<FaceLandmarkResult> landmarks;
    FaceMeshTimings                   timings;
};

struct FaceMeshPipelineOptions
{
    float landmark_score_threshold = 0.50F;
};

struct Face68Landmark
{
    // x and y use the model's 256-pixel input coordinate system. score is returned unchanged.
    float x     = 0.0f;
    float y     = 0.0f;
    float score = 0.0f;
};

using Face68Result    = std::array<Face68Landmark, kFace68LandmarkCount>;
using ArcFaceResult   = std::array<float, kArcFaceEmbeddingLength>;
using AgeGenderResult = std::array<float, kAgeGenderLogitCount>;
struct InSwapperResult
{
    std::vector<float> values;
};

struct GfpGanResult
{
    std::vector<float> values;
};

} // namespace kfcore::face_models
