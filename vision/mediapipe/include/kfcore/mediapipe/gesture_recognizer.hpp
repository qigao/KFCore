#pragma once
#include "kfcore/hand_models/runtime.hpp"
#include <array>
#include <memory>

namespace kfcore::mediapipe {
enum class CannedGesture { None, ClosedFist, OpenPalm, PointingUp, ThumbDown, ThumbUp, Victory, ILoveYou };
inline constexpr std::size_t kGestureCount = 8;
inline constexpr std::size_t kEmbeddingSize = 128;
const char* gesture_name(CannedGesture gesture);
struct GesturePrediction {
    CannedGesture label = CannedGesture::None;
    std::array<float, kGestureCount> scores{};
};
struct GestureFrame {
    hand_models::HandFrame landmarks;
    // One prediction per hand, in the same order; no temporal identity implied.
    std::vector<GesturePrediction> gestures;
};
struct GestureFeatures {
    std::array<float, hand_models::kHandLandmarkCount * 3> hand{}, world{};
    float right_hand_probability = 0;
};
// Full source image, no caller-specified rotation. Throws on absent world data.
GestureFeatures gesture_features(const hand_models::HandResult& hand, int width, int height);

// Single-thread owned. Model contexts and results are owned; source is borrowed
// for infer(). Old HandLandmarker and its two-model contract remain unchanged.
class GestureRecognizer final {
public:
    static std::unique_ptr<GestureRecognizer> load(runtime::Runtime& runtime,
        const runtime::ModelPackage& palm, const runtime::ModelPackage& landmark,
        const runtime::ModelPackage& embedder, const runtime::ModelPackage& classifier,
        const runtime::ExecutionPolicy& policy, const hand_models::HandRuntimeOptions& options = {});
    ~GestureRecognizer();
    GestureRecognizer(const GestureRecognizer&) = delete;
    GestureRecognizer& operator=(const GestureRecognizer&) = delete;
    GestureFrame infer(const image::ImageView& image);
    GesturePrediction classify(const GestureFeatures& features);
private:
    struct Impl;
    explicit GestureRecognizer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
} // namespace kfcore::mediapipe
