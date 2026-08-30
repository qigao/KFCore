#pragma once

#include "hand_shape_descriptor.hpp"
#include "kfcore/hand_interaction/types.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace kfcore::hand_interaction::detail {

struct HandIdentityObservation {
  int raw_track_id = -1;
  double center_x = 0.0;
  double center_y = 0.0;
  double scale = 0.0;
  float confidence = 0.0f;
  std::optional<HandShapeDescriptor> shape;
  std::optional<hand_models::HandAppearanceDescriptor> appearance;
  hand_models::Handedness handedness = hand_models::Handedness::Unknown;
};

struct HandIdentityConfig {
  int reacquire_frames = 180;
  float minimum_confidence = 0.85f;
  float maximum_distance_scale_ratio = 1.75f;
  float maximum_linear_scale_ratio = 2.75f;
  float ambiguity_cost_margin = 0.20f;
  float scale_cost_weight = 0.25f;
  float age_cost_weight = 0.10f;
  float raw_id_continuity_bonus = 0.08f;
  float handedness_conflict_cost = 1.0f;
  float velocity_observation_weight = 0.65f;
  int maximum_prediction_frames = 2;
  std::size_t maximum_identities = 32;
  float maximum_shape_distance = 0.20f;
  float shape_cost_weight = 2.0f;
  float shape_update_weight = 0.20f;
  float maximum_appearance_distance = 0.18f;
  float maximum_appearance_part_distance = 0.45f;
  float appearance_cost_weight = 3.0f;
  float appearance_update_weight = 0.10f;
  std::size_t minimum_comparable_appearance_parts = 2;
};

struct HandIdentityResolution {
  int canonical_id = 0;
  HandIdentityAssociation association =
      HandIdentityAssociation::UnreliableObservation;
};

// Owns bounded-reacquisition identities exposed to temporal consumers. Shape and
// known handedness gate candidates during the retention horizon; appearance,
// ByteTrack, and normalized spatial evidence rank compatible states. Ambiguous
// observations remain unconfirmed (0).
class HandTrackIdentityRegistry {
public:
  explicit HandTrackIdentityRegistry(HandIdentityConfig config = {});
  ~HandTrackIdentityRegistry();

  HandTrackIdentityRegistry(const HandTrackIdentityRegistry&);
  HandTrackIdentityRegistry& operator=(const HandTrackIdentityRegistry&);
  HandTrackIdentityRegistry(HandTrackIdentityRegistry&&) noexcept;
  HandTrackIdentityRegistry& operator=(HandTrackIdentityRegistry&&) noexcept;

  [[nodiscard]] std::vector<HandIdentityResolution> Resolve(
      const std::vector<HandIdentityObservation>& observations);
  void Reset();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kfcore::hand_interaction::detail
