#include "hand_identity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
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
  std::optional<hand_models::HandAppearanceDescriptor> appearance;
  hand_models::Handedness handedness = hand_models::Handedness::Unknown;
  double center_x = 0.0;
  double center_y = 0.0;
  double scale = 0.0;
  double velocity_x = 0.0;
  double velocity_y = 0.0;
  int observation_count = 0;
  std::uint64_t last_seen_frame = 0;
  std::uint64_t last_observed_frame = 0;
};

struct Candidate {
  int canonical_id = 0;
  double cost = std::numeric_limits<double>::infinity();
  double distance_scale_ratio = std::numeric_limits<double>::infinity();
  bool compatible_shape = false;
  bool compatible_appearance = false;
  bool raw_continuity = false;
  bool handedness_conflict = false;
};

bool IsValidAppearance(
    const hand_models::HandAppearanceDescriptor& descriptor) {
  if (descriptor.valid_parts == 0U ||
      (descriptor.valid_parts & ~hand_models::kAllHandAppearanceParts) != 0U) {
    return false;
  }
  for (std::size_t part_index = 0U;
       part_index < hand_models::kHandAppearancePartCount; ++part_index) {
    const auto part = static_cast<hand_models::HandAppearancePart>(part_index);
    const bool valid =
        (descriptor.valid_parts & hand_models::hand_appearance_part_bit(part)) != 0U;
    const float quality = descriptor.quality[part_index];
    if (!std::isfinite(quality) ||
        (valid ? quality <= 0.0F || quality > 1.0F : quality != 0.0F)) {
      return false;
    }
    const std::size_t offset = hand_models::hand_appearance_feature_offset(part);
    const std::size_t count = hand_models::hand_appearance_feature_count(part);
    const std::size_t texture_count = count / 2U;
    for (std::size_t local = 0U; local < count; ++local) {
      const float value = descriptor.values[offset + local];
      const bool texture = local < texture_count;
      if (!std::isfinite(value) ||
          (texture && (value < -1.0F || value > 1.0F)) ||
          (!texture && (value < 0.0F || value > 1.0F))) {
        return false;
      }
    }
  }
  return true;
}

struct AppearanceComparison {
  double distance = 0.0;
  double maximum_part_distance = 0.0;
  std::size_t comparable_parts = 0U;
};

std::optional<AppearanceComparison> CompareAppearance(
    const std::optional<hand_models::HandAppearanceDescriptor>& first,
    const std::optional<hand_models::HandAppearanceDescriptor>& second,
    std::size_t minimum_comparable_parts) {
  if (!first || !second) {
    return std::nullopt;
  }
  AppearanceComparison comparison;
  double weighted_distance = 0.0;
  double total_weight = 0.0;
  for (std::size_t part_index = 0U;
       part_index < hand_models::kHandAppearancePartCount; ++part_index) {
    const auto part = static_cast<hand_models::HandAppearancePart>(part_index);
    const std::uint8_t bit = hand_models::hand_appearance_part_bit(part);
    if ((first->valid_parts & bit) == 0U || (second->valid_parts & bit) == 0U) {
      continue;
    }
    const std::size_t offset = hand_models::hand_appearance_feature_offset(part);
    const std::size_t count = hand_models::hand_appearance_feature_count(part);
    double part_distance = 0.0;
    for (std::size_t local = 0U; local < count; ++local) {
      part_distance += std::abs(
          static_cast<double>(first->values[offset + local]) -
          static_cast<double>(second->values[offset + local]));
    }
    part_distance /= static_cast<double>(count);
    const double weight = static_cast<double>(
        std::min(first->quality[part_index], second->quality[part_index]));
    weighted_distance += weight * part_distance;
    total_weight += weight;
    comparison.maximum_part_distance =
        std::max(comparison.maximum_part_distance, part_distance);
    ++comparison.comparable_parts;
  }
  if (comparison.comparable_parts < minimum_comparable_parts ||
      total_weight <= 0.0) {
    return std::nullopt;
  }
  comparison.distance = weighted_distance / total_weight;
  return comparison;
}

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
    config_.handedness_conflict_cost =
        std::max(0.0f, config_.handedness_conflict_cost);
    config_.velocity_observation_weight =
        std::clamp(config_.velocity_observation_weight, 0.0f, 1.0f);
    config_.maximum_prediction_frames =
        std::max(1, config_.maximum_prediction_frames);
    config_.maximum_identities = std::max<std::size_t>(1, config_.maximum_identities);
    config_.maximum_appearance_distance =
        std::max(0.0001f, config_.maximum_appearance_distance);
    config_.maximum_appearance_part_distance =
        std::max(0.0001f, config_.maximum_appearance_part_distance);
    config_.appearance_cost_weight =
        std::max(0.0f, config_.appearance_cost_weight);
    config_.appearance_update_weight =
        std::clamp(config_.appearance_update_weight, 0.0001f, 1.0f);
    config_.minimum_comparable_appearance_parts =
        std::clamp<std::size_t>(config_.minimum_comparable_appearance_parts, 1U,
                                hand_models::kHandAppearancePartCount);
  }

  std::vector<HandIdentityResolution> Resolve(
      const std::vector<HandIdentityObservation>& observations) {
    ++frame_index_;
    PruneExpired();

    std::vector<HandIdentityResolution> resolved(observations.size());
    std::vector<std::optional<Candidate>> selected_matches(observations.size());
    std::unordered_set<int> used_canonical_ids;
    std::vector<PendingMatch> pending_matches;
    std::vector<std::size_t> new_identity_observations;
    for (std::size_t index = 0; index < observations.size(); ++index) {
      if (observations[index].appearance &&
          !IsValidAppearance(*observations[index].appearance)) {
        throw std::invalid_argument("invalid hand appearance descriptor");
      }
      if (!IsReliable(observations[index])) {
        const int raw_track_id = observations[index].raw_track_id;
        if (raw_track_id >= 0) {
          IdentityState* retained = nullptr;
          for (auto& [canonical_id, state] : identities_) {
            (void)canonical_id;
            if (state.raw_track_id != raw_track_id) {
              continue;
            }
            if (retained != nullptr) {
              retained = nullptr;
              break;
            }
            retained = &state;
          }
          if (retained != nullptr) {
            resolved[index].canonical_id = retained->canonical_id;
            resolved[index].association =
                HandIdentityAssociation::UnreliableObservation;
            retained->last_observed_frame = frame_index_;
          }
        }
        continue;
      }
      resolved[index].association = HandIdentityAssociation::Ambiguous;
      const auto& observation = observations[index];
      std::vector<Candidate> candidates;
      candidates.reserve(identities_.size());
      // Bounded matching costs O(H * I * (20 + 256)): each observation compares
      // fixed shape and, when available, appearance descriptors against each
      // retention-limited prototype.
      for (const auto& [canonical_id, state] : identities_) {
        const bool handedness_conflict =
            state.handedness != hand_models::Handedness::Unknown &&
            observation.handedness != hand_models::Handedness::Unknown &&
            state.handedness != observation.handedness;
        const double shape_distance =
            HandShapeDistance(*observation.shape, state.shape);
        const bool compatible_shape =
            shape_distance <= config_.maximum_shape_distance;
        const std::optional<AppearanceComparison> appearance = CompareAppearance(
            observation.appearance, state.appearance,
            config_.minimum_comparable_appearance_parts);
        const bool compatible_appearance =
            appearance &&
            appearance->distance <= config_.maximum_appearance_distance &&
            appearance->maximum_part_distance <=
                config_.maximum_appearance_part_distance;
        const bool raw_continuity =
            observation.raw_track_id >= 0 &&
            observation.raw_track_id == state.raw_track_id;
        // Handedness is a classifier output and may flip for a single frame.
        // Preserve the conflict as a hard rejection only when neither the raw
        // track nor compatible appearance independently identifies this state.
        if (handedness_conflict && !raw_continuity && !compatible_appearance) {
          continue;
        }
        // A single landmark outlier must not fork an otherwise continuous raw
        // track. Shape remains mandatory when neither appearance nor raw
        // continuity provides independent identity evidence.
        if (!compatible_shape && !compatible_appearance && !raw_continuity) {
          continue;
        }
        const std::uint64_t age = frame_index_ - state.last_seen_frame;
        const double distance_scale_ratio = PredictedDistanceRatio(
            observation, state, frame_index_, config_.maximum_prediction_frames);
        const double capped_age_ratio = static_cast<double>(
            std::min<std::uint64_t>(age,
                                    static_cast<std::uint64_t>(config_.reacquire_frames))) /
            static_cast<double>(config_.reacquire_frames);
        const double raw_bonus =
            raw_continuity
                ? static_cast<double>(config_.raw_id_continuity_bonus)
                : 0.0;
        double cost = static_cast<double>(config_.shape_cost_weight) * shape_distance +
                      static_cast<double>(config_.age_cost_weight) * capped_age_ratio -
                      raw_bonus;
        if (handedness_conflict) {
          cost += static_cast<double>(config_.handedness_conflict_cost);
        }
        if (appearance) {
          // A pose change can alter aligned finger pixels without changing the
          // physical hand. Keep that mismatch as bounded negative evidence;
          // shape and motion may still reacquire the identity.
          cost += static_cast<double>(config_.appearance_cost_weight) *
                  (compatible_appearance
                       ? appearance->distance
                       : std::max(appearance->distance,
                                  appearance->maximum_part_distance));
        }
        const double scale_ratio =
            LinearScaleRatio(static_cast<double>(observation.scale), state.scale);
        cost += SaturatedDistanceCost(
                    distance_scale_ratio,
                    static_cast<double>(config_.maximum_distance_scale_ratio)) +
                static_cast<double>(config_.scale_cost_weight) *
                    SaturatedScaleCost(
                        scale_ratio,
                        static_cast<double>(config_.maximum_linear_scale_ratio));
        candidates.push_back({canonical_id, cost, distance_scale_ratio,
                              compatible_shape,
                              compatible_appearance, raw_continuity,
                              handedness_conflict});
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
      bool prefer_unique_raw = false;
      auto raw_candidate = std::find_if(
          candidates.begin(), candidates.end(),
          [](const Candidate& candidate) { return candidate.raw_continuity; });
      if (raw_candidate != candidates.end()) {
        const bool another_appearance_match = std::any_of(
            candidates.begin(), candidates.end(),
            [&](const Candidate& candidate) {
              return candidate.canonical_id != raw_candidate->canonical_id &&
                     candidate.compatible_appearance;
            });
        const bool spatially_isolated =
            raw_candidate->distance_scale_ratio <=
                static_cast<double>(config_.maximum_distance_scale_ratio) &&
            std::all_of(candidates.begin(), candidates.end(),
                        [&](const Candidate& candidate) {
                          return candidate.canonical_id ==
                                     raw_candidate->canonical_id ||
                                 candidate.distance_scale_ratio >
                                     static_cast<double>(
                                         config_.maximum_distance_scale_ratio);
                        });
        const bool single_observation_continuity =
            observations.size() == 1U && !raw_candidate->handedness_conflict;
        const bool isolated_handedness_noise =
            raw_candidate->handedness_conflict && spatially_isolated;
        if ((single_observation_continuity || isolated_handedness_noise) &&
            (raw_candidate->compatible_appearance ||
             !another_appearance_match)) {
          std::rotate(candidates.begin(), raw_candidate,
                      std::next(raw_candidate));
          prefer_unique_raw = true;
        }
      }
      const double certainty =
          candidates.size() == 1 || prefer_unique_raw
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
      auto& resolution = resolved[pending.observation_index];
      resolution.canonical_id = canonical_id;
      const auto& observation = observations[pending.observation_index];
      resolution.association =
          observation.raw_track_id >= 0 &&
                  observation.raw_track_id == state->second.raw_track_id
              ? HandIdentityAssociation::RawTrackContinuity
              : (pending.candidates.front().compatible_appearance
                     ? HandIdentityAssociation::AppearanceReacquired
                     : HandIdentityAssociation::ShapeReacquired);
      selected_matches[pending.observation_index] = pending.candidates.front();
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
      resolved[observation_index].canonical_id = canonical_id;
      resolved[observation_index].association =
          HandIdentityAssociation::NewIdentity;
      used_canonical_ids.insert(canonical_id);
    }

    // A raw tracker id has one owner. Re-identification may move that binding;
    // clear the previous owner before committing any prototype updates.
    for (std::size_t index = 0; index < observations.size(); ++index) {
      if (resolved[index].canonical_id <= 0 || !IsReliable(observations[index]) ||
          observations[index].raw_track_id < 0) {
        continue;
      }
      for (auto& [canonical_id, state] : identities_) {
        if (canonical_id != resolved[index].canonical_id &&
            state.raw_track_id == observations[index].raw_track_id) {
          state.raw_track_id = -1;
        }
      }
    }

    // Commit only after the complete assignment is known, so one update never
    // changes the candidate costs of another observation in the same frame.
    for (std::size_t index = 0; index < observations.size(); ++index) {
      if (resolved[index].canonical_id <= 0 || !IsReliable(observations[index])) {
        continue;
      }
      UpdateState(identities_.at(resolved[index].canonical_id), observations[index],
                  selected_matches[index]);
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
      if (frame_index_ - iterator->second.last_observed_frame >
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
                   const HandIdentityObservation& observation,
                   const std::optional<Candidate>& evidence) {
    const bool is_new_identity = state.observation_count == 0;
    const bool trusted_geometry =
        is_new_identity || !evidence || evidence->compatible_shape ||
        evidence->compatible_appearance ||
        (evidence->raw_continuity && !evidence->handedness_conflict);
    const bool trusted_shape =
        is_new_identity || !evidence || evidence->compatible_shape;
    const bool trusted_appearance =
        is_new_identity || !evidence || evidence->compatible_appearance ||
        (!state.appearance &&
         (evidence->compatible_shape ||
          (evidence->raw_continuity && !evidence->handedness_conflict)));
    const std::uint64_t age = frame_index_ - state.last_seen_frame;
    if (trusted_geometry && state.observation_count > 0 && age > 0) {
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
    if (is_new_identity) {
      state.shape = *observation.shape;
    } else if (trusted_shape) {
      for (std::size_t index = 0U; index < kHandShapeFeatureCount; ++index) {
        state.shape.values[index] += config_.shape_update_weight *
            (observation.shape->values[index] - state.shape.values[index]);
      }
      (void)NormalizeHandShapeDescriptor(state.shape);
    }
    if (trusted_appearance && observation.appearance) {
      if (!state.appearance) {
        state.appearance.emplace();
      }
      for (std::size_t part_index = 0U;
           part_index < hand_models::kHandAppearancePartCount; ++part_index) {
        const auto part =
            static_cast<hand_models::HandAppearancePart>(part_index);
        const std::uint8_t bit = hand_models::hand_appearance_part_bit(part);
        if ((observation.appearance->valid_parts & bit) == 0U) {
          continue;
        }
        const std::size_t offset =
            hand_models::hand_appearance_feature_offset(part);
        const std::size_t count =
            hand_models::hand_appearance_feature_count(part);
        if ((state.appearance->valid_parts & bit) == 0U) {
          std::copy_n(observation.appearance->values.begin() + offset, count,
                      state.appearance->values.begin() + offset);
          state.appearance->quality[part_index] =
              observation.appearance->quality[part_index];
          state.appearance->valid_parts |= bit;
          continue;
        }
        for (std::size_t local = 0U; local < count; ++local) {
          const std::size_t index = offset + local;
          state.appearance->values[index] += config_.appearance_update_weight *
              (observation.appearance->values[index] -
               state.appearance->values[index]);
        }
        state.appearance->quality[part_index] +=
            config_.appearance_update_weight *
            (observation.appearance->quality[part_index] -
             state.appearance->quality[part_index]);
      }
    }
    if (trusted_geometry &&
        state.handedness == hand_models::Handedness::Unknown &&
        observation.handedness != hand_models::Handedness::Unknown) {
      state.handedness = observation.handedness;
    }
    if (trusted_geometry) {
      state.center_x = static_cast<double>(observation.center_x);
      state.center_y = static_cast<double>(observation.center_y);
      state.scale = static_cast<double>(observation.scale);
      state.last_seen_frame = frame_index_;
      ++state.observation_count;
    }
    state.last_observed_frame = frame_index_;
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

std::vector<HandIdentityResolution> HandTrackIdentityRegistry::Resolve(
    const std::vector<HandIdentityObservation>& observations) {
  auto staged = std::make_unique<Impl>(*impl_);
  auto resolved = staged->Resolve(observations);
  impl_ = std::move(staged);
  return resolved;
}

void HandTrackIdentityRegistry::Reset() { impl_->Reset(); }

}  // namespace kfcore::hand_interaction::detail
