#pragma once

#include "kfcore/thig/temporal_graph.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kfcore::hand_interaction
{

struct GestureFrameContext
{
    std::uint64_t serial = 0;
    std::chrono::steady_clock::time_point observed_at {};
    int image_width  = 0;
    int image_height = 0;
};

struct HandIdentityOptions
{
    int         reacquire_frames              = 180;
    float       minimum_confidence             = 0.85F;
    float       maximum_distance_scale_ratio  = 1.75F;
    float       maximum_linear_scale_ratio    = 2.75F;
    float       ambiguity_cost_margin         = 0.20F;
    float       scale_cost_weight             = 0.25F;
    float       age_cost_weight               = 0.10F;
    float       raw_id_continuity_bonus       = 0.08F;
    float       handedness_conflict_cost      = 1.0F;
    float       velocity_observation_weight   = 0.65F;
    int         maximum_prediction_frames     = 2;
    std::size_t maximum_identities             = 32;
    float       maximum_shape_distance         = 0.20F;
    float       shape_cost_weight              = 2.0F;
    float       shape_update_weight            = 0.20F;
    float       maximum_appearance_distance    = 0.18F;
    float       maximum_appearance_part_distance = 0.45F;
    float       appearance_cost_weight         = 3.0F;
    float       appearance_update_weight       = 0.10F;
    std::size_t minimum_comparable_appearance_parts = 2;
};

struct HandPoseOptions
{
    float minimum_confidence                   = 0.85F;
    float thumb_index_contact_ratio            = 0.20F;
    float thumb_index_contact_to_mcp_ratio     = 0.65F;
    float contact_palm_distance_ratio          = 0.14F;
    float extended_tip_beyond_pip_ratio        = 0.06F;
    float extended_tip_mcp_distance_ratio      = 0.22F;
    float v_extended_tip_beyond_pip_ratio      = 0.04F;
    float v_extended_tip_mcp_distance_ratio    = 0.18F;
    float folded_tip_beyond_pip_max_ratio      = 0.04F;
    float folded_tip_mcp_distance_max_ratio    = 0.30F;
    float v_tip_separation_ratio               = 0.18F;
    float index_extended_mcp_tip_palm_ratio    = 1.05F;
    float index_pressed_mcp_tip_palm_ratio     = 0.70F;
    float index_extended_pip_angle_degrees     = 150.0F;
    float index_pressed_pip_angle_degrees      = 120.0F;
};

struct HandMotionOptions
{
    std::size_t max_history_samples                   = 16;
    int         history_window_ms                     = 1000;
    std::size_t movement_window_samples               = 6;
    std::size_t stationary_samples                    = 4;
    std::uint64_t track_state_ttl_frames              = 90;
    std::uint64_t empty_reset_frames                  = 12;
    float nominal_samples_per_second                  = 30.0F;
    float stop_speed_min_px_per_sample                = 5.0F;
    float stop_speed_hand_ratio                       = 0.045F;
    float move_distance_min_px                        = 12.0F;
    float move_distance_hand_ratio                    = 0.14F;
    float direction_distance_min_px                   = 28.0F;
    float direction_distance_hand_ratio               = 0.28F;
    float direction_dominance_ratio                   = 1.20F;
    float stationary_low_speed_px_per_sample_at_30fps = 8.5F;
    float stationary_total_step_px                    = 78.0F;
    float stationary_displacement_px                  = 46.0F;
    float safe_area_inset_ratio                       = 0.06F;
};

struct HandSpatialOptions
{
    int         window_ms                          = 350;
    std::size_t minimum_samples                    = 3;
    std::size_t max_samples_per_hand               = 16;
    std::size_t max_pair_histories                 = 28;
    float       scale_change_ratio                 = 0.12F;
    float       rotation_change_degrees            = 18.0F;
    float       two_hand_distance_change_ratio     = 0.06F;
    float       palm_axis_horizontal_max_degrees   = 30.0F;
    float       palm_axis_vertical_min_degrees     = 60.0F;
};

struct HandPrimitiveOptions
{
    std::size_t max_hands = 8;
    HandIdentityOptions identity;
    HandPoseOptions     pose;
    HandMotionOptions   motion;
    HandSpatialOptions  spatial;
};

enum class HandIdentityAssociation
{
    UnreliableObservation,
    NewIdentity,
    RawTrackContinuity,
    ShapeReacquired,
    Ambiguous,
    AppearanceReacquired,
};

struct CanonicalHand
{
    std::size_t             input_index  = 0;
    int                     raw_track_id = -1;
    int                     canonical_id = 0;
    HandIdentityAssociation association =
        HandIdentityAssociation::UnreliableObservation;
};

struct PrimitiveFrame
{
    std::vector<CanonicalHand>    hands;
    std::vector<thig::Observation> observations;
};

} // namespace kfcore::hand_interaction
