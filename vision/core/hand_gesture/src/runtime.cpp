#include "kfcore/hand_gesture/runtime.hpp"

#include "kfcore/runtime/error.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kfcore::hand_gesture
{
namespace
{

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw HandGestureError(HandGestureErrorCode::InvalidArgument,
                           "temporal gesture: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw HandGestureError(HandGestureErrorCode::ModelContractMismatch,
                           "temporal gesture model contract: " + detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw HandGestureError(HandGestureErrorCode::ResourceLimitExceeded,
                           "temporal gesture resource limit: " + detail);
}

[[noreturn]] void throw_runtime(const std::string& detail)
{
    throw HandGestureError(HandGestureErrorCode::RuntimeFailure,
                           "temporal gesture runtime: " + detail);
}

void validate_options(const TemporalGestureOptions& options)
{
    if (!std::isfinite(options.minimum_confidence) ||
        options.minimum_confidence < 0.0F || options.minimum_confidence > 1.0F)
    {
        throw_invalid("minimum_confidence must be finite within [0,1]");
    }
    if (options.maximum_observation_gap_ns == 0U)
    {
        throw_invalid("maximum_observation_gap_ns must be positive");
    }
    if (options.maximum_tracks == 0U)
    {
        throw_invalid("maximum_tracks must be positive");
    }
}

bool shape_equals(const runtime::TensorShape& actual,
                  std::initializer_list<std::int64_t> expected)
{
    return actual.size() == expected.size() &&
           std::equal(actual.begin(), actual.end(), expected.begin(), expected.end());
}

const runtime::TensorDescriptor& require_tensor(
    const std::vector<runtime::TensorDescriptor>& tensors,
    const char* name,
    bool input,
    std::initializer_list<std::int64_t> shape)
{
    const runtime::TensorDescriptor* found = nullptr;
    for (const auto& tensor : tensors)
    {
        if (tensor.name != name)
        {
            continue;
        }
        if (found != nullptr)
        {
            throw_contract(std::string("duplicate tensor '") + name + "'");
        }
        found = &tensor;
    }
    if (found == nullptr)
    {
        throw_contract(std::string("missing tensor '") + name + "'");
    }
    if (found->is_input != input)
    {
        throw_contract(std::string("tensor '") + name + "' has wrong direction");
    }
    if (found->data_type != runtime::DataType::Float32)
    {
        throw_contract(std::string("tensor '") + name + "' must be FP32");
    }
    if (!shape_equals(found->shape, shape))
    {
        throw_contract(std::string("tensor '") + name + "' has wrong fixed shape");
    }
    return *found;
}

template <std::size_t N>
std::array<float, N> softmax(const std::array<float, N>& logits, const char* subject)
{
    float maximum = -(std::numeric_limits<float>::infinity)();
    for (const float value : logits)
    {
        if (!std::isfinite(value))
        {
            throw_contract(std::string(subject) + " contains non-finite logits");
        }
        maximum = (std::max)(maximum, value);
    }

    std::array<float, N> probabilities {};
    double denominator = 0.0;
    for (std::size_t index = 0U; index < N; ++index)
    {
        const double value = std::exp(static_cast<double>(logits[index] - maximum));
        probabilities[index] = static_cast<float>(value);
        denominator += value;
    }
    if (!std::isfinite(denominator) || denominator <= 0.0)
    {
        throw_contract(std::string(subject) + " softmax denominator is invalid");
    }
    for (float& value : probabilities)
    {
        value = static_cast<float>(static_cast<double>(value) / denominator);
        if (!std::isfinite(value))
        {
            throw_contract(std::string(subject) + " softmax probability is non-finite");
        }
    }
    return probabilities;
}

template <std::size_t N>
std::size_t argmax(const std::array<float, N>& values) noexcept
{
    return static_cast<std::size_t>(
        std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

class UseGuard final
{
public:
    explicit UseGuard(std::atomic_flag& flag) : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw HandGestureError(HandGestureErrorCode::ConcurrentExecution,
                                   "temporal gesture recognizer is already in use");
        }
    }

    ~UseGuard()
    {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag& flag_;
};

struct TrackState
{
    GestureFeatureState encoder_state {};
    GestureHiddenState hidden {};
    bool active = false;
    GestureClass active_gesture = GestureClass::None;
};

std::optional<GestureEvent> decode_event(
    TrackState& state,
    int track_id,
    std::uint64_t timestamp_ns,
    GestureClass gesture,
    GesturePhase phase,
    float confidence,
    float minimum_confidence)
{
    if (!std::isfinite(confidence))
    {
        throw_contract("gesture confidence is non-finite");
    }

    const auto clear_active = [&state]() {
        state.active = false;
        state.active_gesture = GestureClass::None;
    };

    if (confidence < minimum_confidence || gesture == GestureClass::None)
    {
        clear_active();
        return std::nullopt;
    }

    switch (phase)
    {
    case GesturePhase::Idle:
        clear_active();
        return std::nullopt;

    case GesturePhase::Start:
        state.active = true;
        state.active_gesture = gesture;
        return GestureEvent {track_id, gesture, phase, confidence, timestamp_ns};

    case GesturePhase::Active:
        if (!state.active || state.active_gesture != gesture)
        {
            clear_active();
            return std::nullopt;
        }
        return GestureEvent {track_id, gesture, phase, confidence, timestamp_ns};

    case GesturePhase::End:
        if (!state.active || state.active_gesture != gesture)
        {
            clear_active();
            return std::nullopt;
        }
        {
            GestureEvent event {track_id, gesture, phase, confidence, timestamp_ns};
            clear_active();
            return event;
        }
    }

    clear_active();
    return std::nullopt;
}

} // namespace

struct TemporalGestureRecognizer::Impl final
{
    Impl(runtime::ResolvedModel resolved_value, TemporalGestureOptions options_value)
        : resolved(std::move(resolved_value))
        , options(options_value)
        , context(resolved.model->create_context())
    {
        if (resolved.route.backend_id != "onnxruntime" ||
            resolved.route.device_id != "cpu")
        {
            throw_contract("V1 requires the ONNX Runtime CPU route");
        }
        if (resolved.route.artifact.flavor != "causal-gru-v1")
        {
            throw_contract("artifact flavor must be 'causal-gru-v1'");
        }

        tensors = resolved.model->tensors();
        if (tensors.size() != 5U)
        {
            throw_contract("model must expose exactly two inputs and three outputs");
        }
        features = require_tensor(tensors, "features", true, {1, 78});
        hidden_in = require_tensor(tensors, "hidden_in", true, {2, 1, 64});
        gesture_logits = require_tensor(tensors, "gesture_logits", false, {1, 8});
        phase_logits = require_tensor(tensors, "phase_logits", false, {1, 4});
        hidden_out = require_tensor(tensors, "hidden_out", false, {2, 1, 64});
    }

    runtime::ResolvedModel resolved;
    TemporalGestureOptions options;
    std::unique_ptr<runtime::ExecutionContext> context;
    std::vector<runtime::TensorDescriptor> tensors;
    runtime::TensorDescriptor features;
    runtime::TensorDescriptor hidden_in;
    runtime::TensorDescriptor gesture_logits;
    runtime::TensorDescriptor phase_logits;
    runtime::TensorDescriptor hidden_out;
    std::unordered_map<int, TrackState> tracks;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

TemporalGestureRecognizer::TemporalGestureRecognizer(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

TemporalGestureRecognizer::~TemporalGestureRecognizer() = default;

std::unique_ptr<TemporalGestureRecognizer> TemporalGestureRecognizer::load(
    runtime::Runtime& runtime,
    const runtime::ModelPackage& package,
    const runtime::ExecutionPolicy& policy,
    const TemporalGestureOptions& options)
{
    validate_options(options);
    if (package.model_type() != "gesture.temporal-gru")
    {
        throw_contract("ModelPackage model_type must be 'gesture.temporal-gru'");
    }
    if (policy.preferences().size() != 1U ||
        policy.preferences().front().backend_id != "onnxruntime" ||
        policy.preferences().front().device_id != "cpu")
    {
        throw_invalid("V1 requires one explicit execution route: onnxruntime/cpu");
    }

    try
    {
        return std::unique_ptr<TemporalGestureRecognizer>(
            new TemporalGestureRecognizer(std::make_unique<Impl>(
                runtime.load_model(package, policy), options)));
    }
    catch (const HandGestureError&)
    {
        throw;
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("recognizer allocation failed");
    }
}

std::vector<GestureEvent> TemporalGestureRecognizer::update(
    const hand_models::HandFrame& frame,
    const GestureFrameMetadata& metadata)
{
    if (!impl_)
    {
        throw_invalid("recognizer state is unavailable");
    }
    UseGuard guard(impl_->in_use);

    std::optional<int> affected_track;
    try
    {
        std::unordered_set<int> seen;
        seen.reserve(frame.hands.size());
        for (const auto& hand : frame.hands)
        {
            if (hand.track_id < 0)
            {
                throw_invalid("tracked hands must have non-negative track_id values");
            }
            if (!seen.insert(hand.track_id).second)
            {
                throw_invalid("HandFrame contains duplicate track_id values");
            }
        }

        auto candidates = impl_->tracks;
        for (auto iterator = candidates.begin(); iterator != candidates.end();)
        {
            const std::uint64_t previous_timestamp = iterator->second.encoder_state.timestamp_ns;
            if (metadata.timestamp_ns > previous_timestamp &&
                metadata.timestamp_ns - previous_timestamp >
                    impl_->options.maximum_observation_gap_ns)
            {
                iterator = candidates.erase(iterator);
            }
            else
            {
                ++iterator;
            }
        }

        std::vector<GestureEvent> events;
        events.reserve(frame.hands.size());

        for (const auto& hand : frame.hands)
        {
            affected_track = hand.track_id;
            auto existing = candidates.find(hand.track_id);
            bool use_previous = existing != candidates.end();
            if (use_previous)
            {
                const std::uint64_t previous_timestamp = existing->second.encoder_state.timestamp_ns;
                if (metadata.timestamp_ns <= previous_timestamp ||
                    metadata.timestamp_ns - previous_timestamp >
                        impl_->options.maximum_observation_gap_ns)
                {
                    candidates.erase(existing);
                    existing = candidates.end();
                    use_previous = false;
                }
            }

            if (!use_previous && candidates.size() >= impl_->options.maximum_tracks)
            {
                throw_resource("maximum_tracks capacity reached");
            }

            TrackState candidate;
            if (use_previous)
            {
                candidate = existing->second;
            }

            const std::optional<GestureFeatureState> previous = use_previous
                ? std::optional<GestureFeatureState>(candidate.encoder_state)
                : std::nullopt;
            const EncodedGestureFeatures encoded = GestureFeatureEncoder::encode(
                hand, metadata, previous);

            std::array<float, kTemporalGestureClassCount> gesture_output {};
            std::array<float, kTemporalGesturePhaseCount> phase_output {};
            GestureHiddenState hidden_output {};

            const runtime::TensorView feature_view {
                impl_->features.name,
                runtime::DataType::Float32,
                {1, static_cast<std::int64_t>(kTemporalGestureFeatureCount)},
                encoded.values.data(),
                encoded.values.size() * sizeof(float),
                runtime::MemoryKind::Host,
                {},
            };
            const runtime::TensorView hidden_view {
                impl_->hidden_in.name,
                runtime::DataType::Float32,
                {static_cast<std::int64_t>(kTemporalGestureHiddenLayers), 1,
                 static_cast<std::int64_t>(kTemporalGestureHiddenSize)},
                candidate.hidden.data(),
                candidate.hidden.size() * sizeof(float),
                runtime::MemoryKind::Host,
                {},
            };

            runtime::MutableTensorView gesture_view {
                impl_->gesture_logits.name,
                runtime::DataType::Float32,
                {1, static_cast<std::int64_t>(kTemporalGestureClassCount)},
                gesture_output.data(),
                gesture_output.size() * sizeof(float),
                runtime::MemoryKind::Host,
                {},
            };
            runtime::MutableTensorView phase_view {
                impl_->phase_logits.name,
                runtime::DataType::Float32,
                {1, static_cast<std::int64_t>(kTemporalGesturePhaseCount)},
                phase_output.data(),
                phase_output.size() * sizeof(float),
                runtime::MemoryKind::Host,
                {},
            };
            runtime::MutableTensorView hidden_output_view {
                impl_->hidden_out.name,
                runtime::DataType::Float32,
                {static_cast<std::int64_t>(kTemporalGestureHiddenLayers), 1,
                 static_cast<std::int64_t>(kTemporalGestureHiddenSize)},
                hidden_output.data(),
                hidden_output.size() * sizeof(float),
                runtime::MemoryKind::Host,
                {},
            };

            impl_->context->run({feature_view, hidden_view},
                                {gesture_view, phase_view, hidden_output_view});

            for (const float value : hidden_output)
            {
                if (!std::isfinite(value))
                {
                    throw_contract("hidden_out contains non-finite values");
                }
            }

            const auto gesture_probabilities = softmax(gesture_output, "gesture head");
            const auto phase_probabilities = softmax(phase_output, "phase head");
            const std::size_t gesture_index = argmax(gesture_probabilities);
            const std::size_t phase_index = argmax(phase_probabilities);
            const GestureClass gesture = static_cast<GestureClass>(gesture_index);
            const GesturePhase phase = static_cast<GesturePhase>(phase_index);
            const float confidence = gesture_probabilities[gesture_index];

            candidate.encoder_state = encoded.next_state;
            candidate.hidden = hidden_output;
            if (const auto event = decode_event(
                    candidate, hand.track_id, metadata.timestamp_ns,
                    gesture, phase, confidence, impl_->options.minimum_confidence))
            {
                events.push_back(*event);
            }
            candidates[hand.track_id] = std::move(candidate);
            affected_track.reset();
        }

        impl_->tracks = std::move(candidates);
        return events;
    }
    catch (const HandGestureError&)
    {
        if (affected_track)
        {
            impl_->tracks.erase(*affected_track);
        }
        throw;
    }
    catch (const runtime::RuntimeError& error)
    {
        if (affected_track)
        {
            impl_->tracks.erase(*affected_track);
        }
        throw_runtime(error.what());
    }
    catch (const std::bad_alloc&)
    {
        if (affected_track)
        {
            impl_->tracks.erase(*affected_track);
        }
        throw_resource("update allocation failed");
    }
}

void TemporalGestureRecognizer::reset()
{
    if (!impl_)
    {
        throw_invalid("recognizer state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    impl_->tracks.clear();
}

void TemporalGestureRecognizer::reset_track(int track_id)
{
    if (!impl_)
    {
        throw_invalid("recognizer state is unavailable");
    }
    if (track_id < 0)
    {
        throw_invalid("track_id must be non-negative");
    }
    UseGuard guard(impl_->in_use);
    impl_->tracks.erase(track_id);
}

const runtime::ExecutionRoute& TemporalGestureRecognizer::execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty {};
    return impl_ ? impl_->resolved.route : empty;
}

} // namespace kfcore::hand_gesture
