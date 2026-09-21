#pragma once
#include "composition_esn.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace kfcore::gesture_interaction {
struct ActionGateOptions {
    float minimum_score = 0.5F; // Linear ESN score, not a probability.
    float minimum_margin = 0.15F;
    double confirmation_seconds = 0.15;
    double rearm_seconds = 0.3;
};
// One gate per independently trained branch. Uncertain output cannot rearm a
// latched action; only a sustained, confident neutral prediction can do so.
class ActionGate {
public:
    explicit ActionGate(ActionGateOptions options = {}) : options_(options) {
        if (!std::isfinite(options.minimum_score) || !std::isfinite(options.minimum_margin) ||
            options.minimum_margin < 0 || !std::isfinite(options.confirmation_seconds) ||
            options.confirmation_seconds <= 0 || !std::isfinite(options.rearm_seconds) || options.rearm_seconds <= 0)
            throw std::invalid_argument("invalid event gate options");
    }
    void reset() noexcept { candidate_.reset(); latched_ = Composition::None; last_time_.reset(); }
    void interrupt() noexcept { candidate_.reset(); last_time_.reset(); }
    const ActionGateOptions& options() const noexcept { return options_; }
    std::optional<Composition> update(const CompositionPrediction& result, double seconds) {
        if (!std::isfinite(seconds) || (last_time_ && seconds <= *last_time_)) {
            reset(); throw std::invalid_argument("invalid event timestamp");
        }
        last_time_ = seconds;
        if (result.class_count < 2 || result.class_count > kCompositionClasses) {
            reset(); throw std::invalid_argument("invalid event class count");
        }
        auto sorted = result.scores;
        for (int i = 0; i < result.class_count; ++i) if (!std::isfinite(sorted[i])) {
            reset(); throw std::invalid_argument("non-finite event score");
        }
        std::sort(sorted.begin(), sorted.begin()+result.class_count, std::greater<float>());
        if (sorted[0] < options_.minimum_score || sorted[0]-sorted[1] < options_.minimum_margin) {
            candidate_.reset(); return std::nullopt;
        }
        if (!candidate_ || *candidate_ != result.label) {
            candidate_ = result.label; since_ = seconds;
        }
        const double duration = seconds-since_;
        if (result.label == Composition::None) {
            if (duration >= options_.rearm_seconds) latched_ = Composition::None;
            return std::nullopt;
        }
        if (duration < options_.confirmation_seconds || latched_ == result.label) return std::nullopt;
        latched_ = result.label;
        return result.label;
    }
private:
    ActionGateOptions options_;
    std::optional<Composition> candidate_;
    Composition latched_ = Composition::None;
    std::optional<double> last_time_;
    double since_ = 0;
};
} // namespace kfcore::gesture_interaction
