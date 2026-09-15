#pragma once

#include "kfcore/hand_gesture/error.hpp"
#include "kfcore/hand_gesture/feature_encoder.hpp"
#include "kfcore/hand_gesture/types.hpp"
#include "kfcore/hand_models/types.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/resolver.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace kfcore::hand_gesture
{

struct TemporalGestureOptions
{
    float minimum_confidence = 0.70F;
    std::uint64_t maximum_observation_gap_ns = 350'000'000ULL;
    std::size_t maximum_tracks = 8U;
};

class TemporalGestureRecognizer final
{
public:
    ~TemporalGestureRecognizer();
    TemporalGestureRecognizer(const TemporalGestureRecognizer&) = delete;
    TemporalGestureRecognizer& operator=(const TemporalGestureRecognizer&) = delete;

    [[nodiscard]] static std::unique_ptr<TemporalGestureRecognizer> load(
        runtime::Runtime& runtime,
        const runtime::ModelPackage& package,
        const runtime::ExecutionPolicy& policy,
        const TemporalGestureOptions& options = {});

    [[nodiscard]] std::vector<GestureEvent> update(
        const hand_models::HandFrame& frame,
        const GestureFrameMetadata& metadata);

    void reset();
    void reset_track(int track_id);

    [[nodiscard]] const runtime::ExecutionRoute& execution_route() const noexcept;

private:
    struct Impl;
    explicit TemporalGestureRecognizer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::hand_gesture
