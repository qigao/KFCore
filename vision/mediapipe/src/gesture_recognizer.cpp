#include "kfcore/mediapipe/gesture_recognizer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kfcore::mediapipe {
namespace {
constexpr float kNormalizationEpsilon = 1e-5F;
void normalize(std::array<float, hand_models::kHandLandmarkCount * 3>& values) {
    const std::array<float, 3> origin{values[0], values[1], values[2]};
    float min_x = 0, max_x = 0, min_y = 0, max_y = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) throw std::invalid_argument("non-finite gesture landmark");
        values[i] -= origin[i % 3];
        if (i % 3 == 0) { min_x = std::min(min_x, values[i]); max_x = std::max(max_x, values[i]); }
        if (i % 3 == 1) { min_y = std::min(min_y, values[i]); max_y = std::max(max_y, values[i]); }
    }
    const float scale = std::max(max_x-min_x, max_y-min_y) + kNormalizationEpsilon;
    for (auto& value : values) value /= scale;
}
void package(const runtime::ModelPackage& value, const char* type, const char* flavor) {
    if (value.model_type() != type || value.artifacts().empty()) throw std::invalid_argument("incorrect gesture model type");
    for (const auto& artifact : value.artifacts())
        if (artifact.format != "onnx" || artifact.backend != "onnxruntime" || artifact.flavor != flavor)
            throw std::invalid_argument(std::string("required gesture model flavor: ") + flavor);
}
void tensors(const runtime::ResolvedModel& model, const std::vector<runtime::TensorDescriptor>& expected) {
    const auto actual = model.model->tensors();
    if (actual.size() != expected.size()) throw std::invalid_argument("gesture tensor count mismatch");
    for (const auto& e : expected) {
        const auto found = std::find_if(actual.begin(), actual.end(), [&](const auto& a) { return a.name == e.name; });
        if (found == actual.end()) throw std::invalid_argument("missing gesture tensor: " + e.name);
        auto shape = found->shape;
        if (!shape.empty() && shape.front() == -1) shape.front() = 1;
        if (found->is_input != e.is_input || shape != e.shape || found->data_type != e.data_type)
            throw std::invalid_argument("gesture tensor contract mismatch: " + e.name);
    }
}
}
const char* gesture_name(CannedGesture gesture) {
    constexpr const char* names[] = {"None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"};
    const auto index = static_cast<std::size_t>(gesture);
    if (index >= kGestureCount) throw std::invalid_argument("invalid canned gesture");
    return names[index];
}
GestureFeatures gesture_features(const hand_models::HandResult& hand, int width, int height) {
    if (width <= 0 || height <= 0 || !hand.world_landmarks || !hand.right_hand_probability)
        throw std::invalid_argument("gesture classification requires world landmarks and handedness probability");
    GestureFeatures result;
    result.right_hand_probability = *hand.right_hand_probability;
    const float extent = float(std::max(width, height));
    for (std::size_t i = 0; i < hand_models::kHandLandmarkCount; ++i) {
        result.hand[i*3] = hand.landmarks[i].x / extent;
        result.hand[i*3+1] = hand.landmarks[i].y / extent;
        result.hand[i*3+2] = hand.landmarks[i].z / float(width);
        const auto& world = (*hand.world_landmarks)[i];
        result.world[i*3] = world.x; result.world[i*3+1] = world.y; result.world[i*3+2] = world.z;
    }
    normalize(result.hand); normalize(result.world);
    return result;
}
struct GestureRecognizer::Impl {
    std::unique_ptr<hand_models::HandDetector> detector;
    runtime::ResolvedModel embedder, classifier;
    std::unique_ptr<runtime::ExecutionContext> embedding_context, classification_context;
};
GestureRecognizer::GestureRecognizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GestureRecognizer::~GestureRecognizer() = default;
std::unique_ptr<GestureRecognizer> GestureRecognizer::load(runtime::Runtime& runtime,
    const runtime::ModelPackage& palm, const runtime::ModelPackage& landmark,
    const runtime::ModelPackage& embedder, const runtime::ModelPackage& classifier,
    const runtime::ExecutionPolicy& policy, const hand_models::HandRuntimeOptions& options) {
    if (policy.preferences().size() != 1 || policy.preferences()[0].backend_id != "onnxruntime")
        throw std::invalid_argument("gesture recognizer requires exact ONNX Runtime policy");
    package(palm, "hand.palm-detector", "mediapipe-palm-postprocess-v1");
    package(landmark, "hand.landmarker", "mediapipe-hand-world-v1");
    package(embedder, "hand.gesture-embedder", "mediapipe-gesture-embedder-v1");
    package(classifier, "hand.gesture-classifier", "mediapipe-canned-gesture-v1");
    auto impl = std::make_unique<Impl>();
    impl->detector = hand_models::HandDetector::load_landmarks(runtime, palm, policy, landmark, policy, options);
    impl->embedder = runtime.load_model(embedder, policy);
    impl->classifier = runtime.load_model(classifier, policy);
    using runtime::DataType;
    tensors(impl->embedder, {{"hand", DataType::Float32, {1,21,3}, true},
        {"handedness", DataType::Float32, {1,1}, true}, {"world_hand", DataType::Float32, {1,21,3}, true},
        {"Identity", DataType::Float32, {1,kEmbeddingSize}, false}});
    tensors(impl->classifier, {{"hand_embedding", DataType::Float32, {1,kEmbeddingSize}, true},
        {"Identity", DataType::Float32, {1,kGestureCount}, false}});
    impl->embedding_context = impl->embedder.model->create_context();
    impl->classification_context = impl->classifier.model->create_context();
    return std::unique_ptr<GestureRecognizer>(new GestureRecognizer(std::move(impl)));
}
GesturePrediction GestureRecognizer::classify(const GestureFeatures& features) {
    for (const auto* points : {&features.hand, &features.world})
        for (float value : *points) if (!std::isfinite(value)) throw std::invalid_argument("non-finite gesture features");
    const float right = features.right_hand_probability;
    if (!std::isfinite(right) || right < 0 || right > 1) throw std::invalid_argument("invalid handedness probability");
    using runtime::DataType;
    std::array<float, kEmbeddingSize> embedding{};
    impl_->embedding_context->run({
        {"hand", DataType::Float32, {1,21,3}, features.hand.data(), sizeof(features.hand)},
        {"handedness", DataType::Float32, {1,1}, &right, sizeof(right)},
        {"world_hand", DataType::Float32, {1,21,3}, features.world.data(), sizeof(features.world)}},
        {{"Identity", DataType::Float32, {1,kEmbeddingSize}, embedding.data(), sizeof(embedding)}});
    GesturePrediction result;
    impl_->classification_context->run({{"hand_embedding", DataType::Float32, {1,kEmbeddingSize}, embedding.data(), sizeof(embedding)}},
        {{"Identity", DataType::Float32, {1,kGestureCount}, result.scores.data(), sizeof(result.scores)}});
    for (float score : result.scores)
        if (!std::isfinite(score) || score < 0 || score > 1) throw std::runtime_error("invalid canned gesture score");
    result.label = CannedGesture(std::max_element(result.scores.begin(), result.scores.end()) - result.scores.begin());
    return result;
}
GestureFrame GestureRecognizer::infer(const image::ImageView& image) {
    GestureFrame result;
    result.landmarks = impl_->detector->infer(image);
    const auto started = std::chrono::steady_clock::now();
    result.gestures.reserve(result.landmarks.hands.size());
    for (const auto& hand : result.landmarks.hands)
        result.gestures.push_back(classify(gesture_features(hand, image.width, image.height)));
    result.landmarks.timings.classifier_inference_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    result.landmarks.timings.total_ms += result.landmarks.timings.classifier_inference_ms;
    return result;
}
} // namespace kfcore::mediapipe
