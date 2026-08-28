#include "hand_identity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kfcore::hand_interaction::detail {
namespace {

struct IdentityState {
  int canonical_id = 0;
  int raw_track_id = -1;
  HandShapeDescriptor shape;
  vision_models::Handedness handedness = vision_models::Handedness::Unknown;
  double center_x = 0.0;
  double center_y = 0.0;
  double scale = 0.0;
  double velocity_x = 0.0;
  double velocity_y = 0.0;
  int observation_count = 0;
  std::uint64_t last_seen_frame = 0;
};

struct Candidate {
  int canonical_id = 0;
  double cost = std::numeric_limits<double>::infinity();
};

double LinearScaleRatio(double lhs, double rhs) {
  const double smaller = std::min(lhs, rhs);
  return smaller > 0.0 ? std::max(lhs, rhs) / smaller
                        : std::numeric_limits<double>::infinity();
}

double PredictedDistanceRatio(const HandIdentityObservation& observation,
                              const IdentityState& state,
                              std::uint64_t frame_index,
                              int maximum_prediction_frames) {
  const double age = static_cast<double>(std::min<std::uint64_t>(
      frame_index - state.last_seen_frame,
      static_cast<std::uint64_t>(maximum_prediction_frames)));
  const double predicted_x = state.center_x + state.velocity_x * age;
  const double predicted_y = state.center_y + state.velocity_y * age;
  const double dx = static_cast<double>(observation.center_x) - predicted_x;
  const double dy = static_cast<double>(observation.center_y) - predicted_y;
  const double scale = std::max({static_cast<double>(observation.scale),
                                 state.scale, 1.0});
  return std::hypot(dx, dy) / scale;
}

double SaturatedDistanceCost(double distance_ratio, double maximum_ratio) {
  return std::min(distance_ratio / maximum_ratio, 1.0);
}

double SaturatedScaleCost(double scale_ratio, double maximum_ratio) {
  const double logarithmic_ratio = std::abs(std::log(scale_ratio));
  const double logarithmic_limit = std::log(maximum_ratio);
  if (logarithmic_limit <= 0.0) {
    return logarithmic_ratio == 0.0 ? 0.0 : 1.0;
  }
  return std::min(logarithmic_ratio / logarithmic_limit, 1.0);
}

struct PendingMatch {
  std::size_t observation_index = 0;
  std::vector<Candidate> candidates;
  double certainty = 0.0;
};

}  // namespace

class HandTrackIdentityRegistry::Impl {
public:
  explicit Impl(HandIdentityConfig config) : config_(std::move(config)) {
    config_.reacquire_frames = std::max(1, config_.reacquire_frames);
    config_.minimum_confidence = std::max(0.0f, config_.minimum_confidence);
    config_.maximum_distance_scale_ratio =
        std::max(0.01f, config_.maximum_distance_scale_ratio);
    config_.maximum_linear_scale_ratio =
        std::max(1.0f, config_.maximum_linear_scale_ratio);
    config_.ambiguity_cost_margin = std::max(0.0f, config_.ambiguity_cost_margin);
    config_.scale_cost_weight = std::max(0.0f, config_.scale_cost_weight);
    config_.age_cost_weight = std::max(0.0f, config_.age_cost_weight);
    config_.raw_id_continuity_bonus =
        std::max(0.0f, config_.raw_id_continuity_bonus);
    config_.velocity_observation_weight =
        std::clamp(config_.velocity_observation_weight, 0.0f, 1.0f);
    config_.maximum_prediction_frames =
        std::max(1, config_.maximum_prediction_frames);
    config_.maximum_identities = std::max<std::size_t>(1, config_.maximum_identities);
  }

  std::vector<int> Resolve(
      const std::vector<HandIdentityObservation>& observations) {
    ++frame_index_;
    PruneExpired();

    std::vector<int> resolved(observations.size(), 0);
    std::unordered_set<int> used_canonical_ids;
    std::vector<PendingMatch> pending_matches;
    std::vector<std::size_t> new_identity_observations;
    for (std::size_t index = 0; index < observations.size(); ++index) {
      if (!IsReliable(observations[index])) {
        continue;
      }
      const auto& observation = observations[index];
      std::vector<Candidate> candidates;
      candidates.reserve(identities_.size());
      // Bounded matching costs O(H * I * 20): each observation compares one
      // inline descriptor against each retention-limited prototype.
      for (const auto& [canonical_id, state] : identities_) {
        if (state.handedness != vision_models::Handedness::Unknown &&
            observation.handedness != vision_models::Handedness::Unknown &&
            state.handedness != observation.handedness) {
          continue;
        }
        const double shape_distance =
            HandShapeDistance(*observation.shape, state.shape);
        if (shape_distance > config_.maximum_shape_distance) {
          continue;
        }
        const std::uint64_t age = frame_index_ - state.last_seen_frame;
        const double capped_age_ratio = static_cast<double>(
            std::min<std::uint64_t>(age,
                                    static_cast<std::uint64_t>(config_.reacquire_frames))) /
            static_cast<double>(config_.reacquire_frames);
        const double raw_bonus = observation.raw_track_id >= 0 &&
                                        observation.raw_track_id == state.raw_track_id
                                    ? static_cast<double>(config_.raw_id_continuity_bonus)
                                    : 0.0;
        double cost = static_cast<double>(config_.shape_cost_weight) * shape_distance +
                      static_cast<double>(config_.age_cost_weight) * capped_age_ratio -
                      raw_bonus;
        const double scale_ratio =
            LinearScaleRatio(static_cast<double>(observation.scale), state.scale);
        cost += SaturatedDistanceCost(
                    PredictedDistanceRatio(observation, state, frame_index_,
                                           config_.maximum_prediction_frames),
                    static_cast<double>(config_.maximum_distance_scale_ratio)) +
                static_cast<double>(config_.scale_cost_weight) *
                    SaturatedScaleCost(
                        scale_ratio,
                        static_cast<double>(config_.maximum_linear_scale_ratio));
        candidates.push_back({canonical_id, cost});
      }
      std::sort(candidates.begin(), candidates.end(),
                [](const Candidate& lhs, const Candidate& rhs) {
                  if (lhs.cost != rhs.cost) {
                    return lhs.cost < rhs.cost;
                  }
                  return lhs.canonical_id < rhs.canonical_id;
                });

      if (candidates.empty()) {
        new_identity_observations.push_back(index);
        continue;
      }
      const double certainty = candidates.size() == 1
                                  ? std::numeric_limits<double>::infinity()
                                  : candidates[1].cost - candidates[0].cost;
      if (certainty < config_.ambiguity_cost_margin) {
        continue;
      }
      pending_matches.push_back({index, std::move(candidates), certainty});
    }

    std::unordered_map<int, std::vector<const PendingMatch*>> competitors;
    for (const auto& pending : pending_matches) {
      competitors[pending.candidates.front().canonical_id].push_back(&pending);
    }
    std::unordered_set<std::size_t> ambiguous_observations;
    for (auto& [canonical_id, competing] : competitors) {
      (void)canonical_id;
      if (competing.size() < 2U) {
        continue;
      }
      std::sort(competing.begin(), competing.end(),
                [](const PendingMatch* lhs, const PendingMatch* rhs) {
                  return lhs->candidates.front().cost < rhs->candidates.front().cost;
                });
      if (competing[1]->candidates.front().cost -
              competing[0]->candidates.front().cost <
          config_.ambiguity_cost_margin) {
        for (const PendingMatch* pending : competing) {
          ambiguous_observations.insert(pending->observation_index);
        }
      }
    }

    // Resolve only competitors with a clear winner. This keeps a shared top
    // canonical identity independent of observation input order.
    std::sort(pending_matches.begin(), pending_matches.end(),
              [](const PendingMatch& lhs, const PendingMatch& rhs) {
                if (lhs.certainty != rhs.certainty) {
                  return lhs.certainty > rhs.certainty;
                }
                if (lhs.candidates.front().cost != rhs.candidates.front().cost) {
                  return lhs.candidates.front().cost < rhs.candidates.front().cost;
                }
                return lhs.observation_index < rhs.observation_index;
              });
    for (const auto& pending : pending_matches) {
      if (ambiguous_observations.find(pending.observation_index) !=
          ambiguous_observations.end()) {
        continue;
      }
      const int canonical_id = pending.candidates.front().canonical_id;
      if (used_canonical_ids.find(canonical_id) != used_canonical_ids.end()) {
        continue;
      }
      auto state = identities_.find(canonical_id);
      if (state == identities_.end()) {
        continue;
      }
      resolved[pending.observation_index] = canonical_id;
      used_canonical_ids.insert(canonical_id);
    }

    const std::size_t remaining_capacity =
        config_.maximum_identities - identities_.size();
    const std::size_t remaining_id_range = static_cast<std::size_t>(
        std::numeric_limits<int>::max() - next_canonical_id_);
    if (new_identity_observations.size() > remaining_capacity ||
        new_identity_observations.size() > remaining_id_range) {
      throw std::length_error("canonical hand identity capacity exhausted");
    }
    for (const std::size_t observation_index : new_identity_observations) {
      const int canonical_id = AllocateIdentity();
      auto& state = identities_[canonical_id];
      state.canonical_id = canonical_id;
      resolved[observation_index] = canonical_id;
      used_canonical_ids.insert(canonical_id);
    }

    // Commit only after the complete assignment is known, so one update never
    // changes the candidate costs of another observation in the same frame.
    for (std::size_t index = 0; index < observations.size(); ++index) {
      if (resolved[index] <= 0) {
        continue;
      }
      UpdateState(identities_.at(resolved[index]), observations[index]);
    }
    return resolved;
  }

  void Reset() {
    identities_.clear();
    next_canonical_id_ = 1;
    frame_index_ = 0;
  }

private:
  void PruneExpired() {
    for (auto iterator = identities_.begin(); iterator != identities_.end();) {
      if (frame_index_ - iterator->second.last_seen_frame >
          static_cast<std::uint64_t>(config_.reacquire_frames)) {
        iterator = identities_.erase(iterator);
      } else {
        ++iterator;
      }
    }
  }

  bool IsReliable(const HandIdentityObservation& observation) const {
    return observation.confidence >= config_.minimum_confidence &&
           observation.scale > 0.0f && std::isfinite(observation.center_x) &&
           std::isfinite(observation.center_y) && std::isfinite(observation.scale) &&
           observation.shape.has_value();
  }

  int AllocateIdentity() {
    if (identities_.size() >= config_.maximum_identities ||
        next_canonical_id_ == std::numeric_limits<int>::max()) {
      return 0;
    }
    return next_canonical_id_++;
  }

  void UpdateState(IdentityState& state,
                   const HandIdentityObservation& observation) {
    const std::uint64_t age = frame_index_ - state.last_seen_frame;
    if (state.observation_count > 0 && age > 0) {
    const double inverse_age = 1.0 / static_cast<double>(age);
      const double observed_velocity_x =
          (static_cast<double>(observation.center_x) - state.center_x) * inverse_age;
      const double observed_velocity_y =
          (static_cast<double>(observation.center_y) - state.center_y) * inverse_age;
      const double weight = state.observation_count == 1
                                ? 1.0
                                : static_cast<double>(config_.velocity_observation_weight);
      state.velocity_x += weight * (observed_velocity_x - state.velocity_x);
      state.velocity_y += weight * (observed_velocity_y - state.velocity_y);
    }
    if (observation.raw_track_id >= 0) {
      state.raw_track_id = observation.raw_track_id;
    }
    if (state.observation_count == 0) {
      state.shape = *observation.shape;
    } else {
      for (std::size_t index = 0U; index < kHandShapeFeatureCount; ++index) {
        state.shape.values[index] += config_.shape_update_weight *
            (observation.shape->values[index] - state.shape.values[index]);
      }
      (void)NormalizeHandShapeDescriptor(state.shape);
    }
    if (state.handedness == vision_models::Handedness::Unknown &&
        observation.handedness != vision_models::Handedness::Unknown) {
      state.handedness = observation.handedness;
    }
    state.center_x = static_cast<double>(observation.center_x);
    state.center_y = static_cast<double>(observation.center_y);
    state.scale = static_cast<double>(observation.scale);
    state.last_seen_frame = frame_index_;
    ++state.observation_count;
  }

  HandIdentityConfig config_;
  std::unordered_map<int, IdentityState> identities_;
  int next_canonical_id_ = 1;
  std::uint64_t frame_index_ = 0;
};

HandTrackIdentityRegistry::HandTrackIdentityRegistry(HandIdentityConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

HandTrackIdentityRegistry::~HandTrackIdentityRegistry() = default;
HandTrackIdentityRegistry::HandTrackIdentityRegistry(
    const HandTrackIdentityRegistry& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}
HandTrackIdentityRegistry& HandTrackIdentityRegistry::operator=(
    const HandTrackIdentityRegistry& other) {
  if (this != &other) {
    impl_ = std::make_unique<Impl>(*other.impl_);
  }
  return *this;
}
HandTrackIdentityRegistry::HandTrackIdentityRegistry(
    HandTrackIdentityRegistry&&) noexcept = default;
HandTrackIdentityRegistry& HandTrackIdentityRegistry::operator=(
    HandTrackIdentityRegistry&&) noexcept = default;

std::vector<int> HandTrackIdentityRegistry::Resolve(
    const std::vector<HandIdentityObservation>& observations) {
  auto staged = std::make_unique<Impl>(*impl_);
  auto resolved = staged->Resolve(observations);
  impl_ = std::move(staged);
  return resolved;
}

void HandTrackIdentityRegistry::Reset() { impl_->Reset(); }

}  // namespace kfcore::hand_interaction::detail
