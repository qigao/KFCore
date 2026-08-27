#include "hand_identity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kfcore::hand_interaction::detail {
namespace {

struct IdentityState {
  int canonical_id = 0;
  int raw_track_id = -1;
  float center_x = 0.0f;
  float center_y = 0.0f;
  float scale = 0.0f;
  float velocity_x = 0.0f;
  float velocity_y = 0.0f;
  int observation_count = 0;
  std::uint64_t last_seen_frame = 0;
};

struct Candidate {
  int canonical_id = 0;
  float cost = std::numeric_limits<float>::infinity();
};

float LinearScaleRatio(float lhs, float rhs) {
  const float smaller = std::min(lhs, rhs);
  return smaller > 0.0f ? std::max(lhs, rhs) / smaller
                        : std::numeric_limits<float>::infinity();
}

float PredictedDistanceRatio(const HandIdentityObservation& observation,
                             const IdentityState& state,
                             std::uint64_t frame_index,
                             int maximum_prediction_frames) {
  const float age = static_cast<float>(std::min<std::uint64_t>(
      frame_index - state.last_seen_frame,
      static_cast<std::uint64_t>(maximum_prediction_frames)));
  const float predicted_x = state.center_x + state.velocity_x * age;
  const float predicted_y = state.center_y + state.velocity_y * age;
  const float dx = observation.center_x - predicted_x;
  const float dy = observation.center_y - predicted_y;
  const float scale = std::max({observation.scale, state.scale, 1.0f});
  return std::hypot(dx, dy) / scale;
}

struct PendingMatch {
  std::size_t observation_index = 0;
  std::vector<Candidate> candidates;
  float certainty = 0.0f;
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
    const std::size_t reliable_observation_count =
        static_cast<std::size_t>(std::count_if(
            observations.begin(), observations.end(),
            [&](const auto& observation) { return IsReliable(observation); }));
    for (std::size_t index = 0; index < observations.size(); ++index) {
      if (!IsReliable(observations[index])) {
        continue;
      }
      const auto& observation = observations[index];
      std::vector<Candidate> candidates;
      candidates.reserve(identities_.size());
      for (const auto& [canonical_id, state] : identities_) {
        if (!IsPlausible(observation, state)) {
          continue;
        }
        const float scale_ratio = LinearScaleRatio(observation.scale, state.scale);
        const float age_ratio = static_cast<float>(frame_index_ - state.last_seen_frame) /
                                static_cast<float>(config_.reacquire_frames);
        const float raw_bonus = observation.raw_track_id >= 0 &&
                                        observation.raw_track_id == state.raw_track_id
                                    ? config_.raw_id_continuity_bonus
                                    : 0.0f;
        candidates.push_back(
            {canonical_id,
             PredictedDistanceRatio(observation, state, frame_index_,
                                    config_.maximum_prediction_frames) +
                 config_.scale_cost_weight * std::abs(std::log(scale_ratio)) +
                 config_.age_cost_weight * age_ratio - raw_bonus});
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
      const float certainty = candidates.size() == 1
                                  ? std::numeric_limits<float>::infinity()
                                  : candidates[1].cost - candidates[0].cost;
      if (certainty < config_.ambiguity_cost_margin) {
        continue;
      }
      pending_matches.push_back({index, std::move(candidates), certainty});
    }

    // Resolve the clearest observations first. A second observation competing
    // for the same identity remains unconfirmed instead of receiving a
    // detection-order-dependent ID.
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

    std::size_t permitted_new_identities =
        reliable_observation_count > identities_.size()
            ? reliable_observation_count - identities_.size()
            : 0;
    for (const std::size_t observation_index : new_identity_observations) {
      if (permitted_new_identities == 0) {
        break;
      }
      const int canonical_id = AllocateIdentity();
      if (canonical_id <= 0) {
        continue;
      }
      auto& state = identities_[canonical_id];
      state.canonical_id = canonical_id;
      resolved[observation_index] = canonical_id;
      used_canonical_ids.insert(canonical_id);
      --permitted_new_identities;
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
  bool IsReliable(const HandIdentityObservation& observation) const {
    return observation.confidence >= config_.minimum_confidence &&
           observation.scale > 0.0f && std::isfinite(observation.center_x) &&
           std::isfinite(observation.center_y) && std::isfinite(observation.scale);
  }

  bool IsPlausible(const HandIdentityObservation& observation,
                   const IdentityState& state) const {
    return PredictedDistanceRatio(observation, state, frame_index_,
                                  config_.maximum_prediction_frames) <=
               config_.maximum_distance_scale_ratio &&
           LinearScaleRatio(observation.scale, state.scale) <=
               config_.maximum_linear_scale_ratio;
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
      const float inverse_age = 1.0f / static_cast<float>(age);
      const float observed_velocity_x =
          (observation.center_x - state.center_x) * inverse_age;
      const float observed_velocity_y =
          (observation.center_y - state.center_y) * inverse_age;
      const float weight = state.observation_count == 1
                               ? 1.0f
                               : config_.velocity_observation_weight;
      state.velocity_x += weight * (observed_velocity_x - state.velocity_x);
      state.velocity_y += weight * (observed_velocity_y - state.velocity_y);
    }
    if (observation.raw_track_id >= 0) {
      state.raw_track_id = observation.raw_track_id;
    }
    state.center_x = observation.center_x;
    state.center_y = observation.center_y;
    state.scale = observation.scale;
    state.last_seen_frame = frame_index_;
    ++state.observation_count;
  }

  void PruneExpired() {
    for (auto iterator = identities_.begin(); iterator != identities_.end();) {
      if (frame_index_ - iterator->second.last_seen_frame <=
          static_cast<std::uint64_t>(config_.reacquire_frames)) {
        ++iterator;
        continue;
      }
      iterator = identities_.erase(iterator);
    }
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
    HandTrackIdentityRegistry&&) noexcept = default;
HandTrackIdentityRegistry& HandTrackIdentityRegistry::operator=(
    HandTrackIdentityRegistry&&) noexcept = default;

std::vector<int> HandTrackIdentityRegistry::Resolve(
    const std::vector<HandIdentityObservation>& observations) {
  return impl_->Resolve(observations);
}

void HandTrackIdentityRegistry::Reset() { impl_->Reset(); }

}  // namespace kfcore::hand_interaction::detail

