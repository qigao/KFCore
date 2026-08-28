#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace kfcore::hand_interaction::detail {

struct HandIdentityObservation {
  int raw_track_id = -1;
  float center_x = 0.0f;
  float center_y = 0.0f;
  float scale = 0.0f;
  float confidence = 0.0f;
};

struct HandIdentityConfig {
  int reacquire_frames = 45;
  float minimum_confidence = 0.85f;
  float maximum_distance_scale_ratio = 1.75f;
  float maximum_linear_scale_ratio = 2.75f;
  float ambiguity_cost_margin = 0.20f;
  float scale_cost_weight = 0.25f;
  float age_cost_weight = 0.10f;
  float raw_id_continuity_bonus = 0.08f;
  float velocity_observation_weight = 0.65f;
  int maximum_prediction_frames = 2;
  std::size_t maximum_identities = 8;
};

// Owns the stable identity exposed to temporal consumers. ByteTrack IDs are
// short-lived evidence only. This registry never predicts geometry and never
// classifies gestures; it only rekeys a recently observed identity when the
// raw tracker recreates its ID. Ambiguous observations remain unconfirmed (0).
class HandTrackIdentityRegistry {
public:
  explicit HandTrackIdentityRegistry(HandIdentityConfig config = {});
  ~HandTrackIdentityRegistry();

  HandTrackIdentityRegistry(const HandTrackIdentityRegistry&);
  HandTrackIdentityRegistry& operator=(const HandTrackIdentityRegistry&);
  HandTrackIdentityRegistry(HandTrackIdentityRegistry&&) noexcept;
  HandTrackIdentityRegistry& operator=(HandTrackIdentityRegistry&&) noexcept;

  [[nodiscard]] std::vector<int> Resolve(
      const std::vector<HandIdentityObservation>& observations);
  void Reset();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kfcore::hand_interaction::detail
