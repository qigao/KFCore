#include "kfcore/hand_interaction/primitive_extractor.hpp"

#include "hand_identity.hpp"
#include "hand_shape_descriptor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kfcore::hand_interaction
{
namespace
{

using vision_models::Gesture;
using vision_models::HandLandmark;
using vision_models::HandResult;

constexpr std::array<std::size_t, 5> kPalmLandmarkIndices = { 0, 5, 9, 13, 17 };
constexpr float kMinimumPalmSpanPixels = 8.0F;
constexpr float kRightAngleDegrees = 90.0F;
constexpr float kHalfTurnDegrees = 180.0F;
constexpr float kFullTurnDegrees = 360.0F;
constexpr float kDegreesPerRadian = kHalfTurnDegrees / 3.14159265358979323846F;
constexpr std::size_t kMaximumSupportedHands = 64;
constexpr float kMaximumHandShapeDistance = 2.0F;

struct PalmGeometry
{
    float center_x = 0.0F;
    float center_y = 0.0F;
    float scale    = 0.0F;
    float normalized_scale   = 0.0F;
    float orientation_degrees = 0.0F;
};

struct TimedPoint
{
    float x = 0.0F;
    float y = 0.0F;
    std::chrono::steady_clock::time_point observed_at {};
};

struct TimedGeometry
{
    PalmGeometry geometry;
    std::chrono::steady_clock::time_point observed_at {};
};

struct TimedScalar
{
    float value = 0.0F;
    std::chrono::steady_clock::time_point observed_at {};
};

struct IndexGeometry
{
    float mcp_tip_palm_ratio = 0.0F;
    float pip_angle_degrees  = 0.0F;
};

enum class IndexPose
{
    Unknown,
    Intermediate,
    Extended,
    Pressed,
};

float distance(const HandLandmark& first, const HandLandmark& second)
{
    return std::hypot(second.x - first.x, second.y - first.y);
}

bool finite(const HandLandmark& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) &&
           std::isfinite(point.z);
}

float hand_scale(const HandResult& hand)
{
    return std::max(1.0F, std::max(std::fabs(hand.palm.box.width),
                                   std::fabs(hand.palm.box.height)));
}

bool estimate_palm_geometry(const HandResult& hand, int image_width,
                            int image_height, PalmGeometry& output)
{
    float center_x = 0.0F;
    float center_y = 0.0F;
    for (const std::size_t index : kPalmLandmarkIndices)
    {
        if (!finite(hand.landmarks[index]))
        {
            return false;
        }
        center_x += hand.landmarks[index].x;
        center_y += hand.landmarks[index].y;
    }
    const float count = static_cast<float>(kPalmLandmarkIndices.size());
    center_x /= count;
    center_y /= count;
    const float scale = std::max(distance(hand.landmarks[5], hand.landmarks[17]),
                                 distance(hand.landmarks[0], hand.landmarks[9]));
    if (!std::isfinite(scale) || scale < kMinimumPalmSpanPixels)
    {
        return false;
    }
    const float frame_diagonal =
        std::hypot(static_cast<float>(image_width),
                   static_cast<float>(image_height));
    const float orientation =
        std::atan2(hand.landmarks[5].y - hand.landmarks[17].y,
                   hand.landmarks[5].x - hand.landmarks[17].x) *
        kDegreesPerRadian;
    if (!std::isfinite(frame_diagonal) || frame_diagonal <= 0.0F ||
        !std::isfinite(orientation))
    {
        return false;
    }
    output = { center_x, center_y, scale, scale / frame_diagonal, orientation };
    return true;
}

float angle_degrees(const HandLandmark& first, const HandLandmark& vertex,
                    const HandLandmark& last)
{
    const float first_x = first.x - vertex.x;
    const float first_y = first.y - vertex.y;
    const float last_x  = last.x - vertex.x;
    const float last_y  = last.y - vertex.y;
    const float first_length = std::hypot(first_x, first_y);
    const float last_length  = std::hypot(last_x, last_y);
    if (!std::isfinite(first_length) || !std::isfinite(last_length) ||
        first_length <= 0.0F || last_length <= 0.0F)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float cosine = std::clamp(
        (first_x * last_x + first_y * last_y) /
            (first_length * last_length),
        -1.0F, 1.0F);
    return std::acos(cosine) * kDegreesPerRadian;
}

bool estimate_index_geometry(const HandResult& hand, IndexGeometry& output)
{
    for (const std::size_t index : { 0U, 5U, 6U, 8U, 9U, 17U })
    {
        if (!finite(hand.landmarks[index]))
        {
            return false;
        }
    }
    const float scale = std::max(distance(hand.landmarks[5], hand.landmarks[17]),
                                 distance(hand.landmarks[0], hand.landmarks[9]));
    if (!std::isfinite(scale) || scale < kMinimumPalmSpanPixels)
    {
        return false;
    }
    const float angle = angle_degrees(hand.landmarks[5], hand.landmarks[6],
                                      hand.landmarks[8]);
    const float ratio = distance(hand.landmarks[5], hand.landmarks[8]) / scale;
    if (!std::isfinite(angle) || !std::isfinite(ratio))
    {
        return false;
    }
    output = { ratio, angle };
    return true;
}

IndexPose classify_index(const IndexGeometry& geometry,
                         const HandPoseOptions& options)
{
    if (geometry.mcp_tip_palm_ratio >=
            options.index_extended_mcp_tip_palm_ratio &&
        geometry.pip_angle_degrees >= options.index_extended_pip_angle_degrees)
    {
        return IndexPose::Extended;
    }
    if (geometry.mcp_tip_palm_ratio <=
            options.index_pressed_mcp_tip_palm_ratio &&
        geometry.pip_angle_degrees <= options.index_pressed_pip_angle_degrees)
    {
        return IndexPose::Pressed;
    }
    return IndexPose::Intermediate;
}

bool finger_extended(const HandResult& hand, std::size_t tip,
                     std::size_t pip, std::size_t mcp,
                     float beyond_pip_ratio, float mcp_distance_ratio)
{
    const float scale = hand_scale(hand);
    return distance(hand.landmarks[tip], hand.landmarks[0]) >=
               distance(hand.landmarks[pip], hand.landmarks[0]) +
                   scale * beyond_pip_ratio &&
           distance(hand.landmarks[tip], hand.landmarks[mcp]) >=
               scale * mcp_distance_ratio;
}

bool finger_folded(const HandResult& hand, std::size_t tip,
                   std::size_t pip, std::size_t mcp,
                   const HandPoseOptions& options)
{
    const float scale = hand_scale(hand);
    return distance(hand.landmarks[tip], hand.landmarks[0]) <=
               distance(hand.landmarks[pip], hand.landmarks[0]) +
                   scale * options.folded_tip_beyond_pip_max_ratio &&
           distance(hand.landmarks[tip], hand.landmarks[mcp]) <=
               scale * options.folded_tip_mcp_distance_max_ratio;
}

bool is_v_pose(const HandResult& hand, float confidence,
               const HandPoseOptions& options)
{
    return confidence >= options.minimum_confidence &&
           finger_extended(hand, 8, 6, 5,
                           options.v_extended_tip_beyond_pip_ratio,
                           options.v_extended_tip_mcp_distance_ratio) &&
           finger_extended(hand, 12, 10, 9,
                           options.v_extended_tip_beyond_pip_ratio,
                           options.v_extended_tip_mcp_distance_ratio) &&
           finger_folded(hand, 16, 14, 13, options) &&
           finger_folded(hand, 20, 18, 17, options) &&
           distance(hand.landmarks[8], hand.landmarks[12]) >=
               hand_scale(hand) * options.v_tip_separation_ratio;
}

bool is_ok_pose(const HandResult& hand, float confidence,
                const HandPoseOptions& options)
{
    if (confidence < options.minimum_confidence)
    {
        return false;
    }
    const auto average = [](std::initializer_list<HandLandmark> points) {
        HandLandmark result {};
        for (const auto& point : points)
        {
            result.x += point.x;
            result.y += point.y;
        }
        const float count = static_cast<float>(points.size());
        result.x /= count;
        result.y /= count;
        return result;
    };
    const float scale = hand_scale(hand);
    const float contact = distance(hand.landmarks[4], hand.landmarks[8]);
    const float mcp_span = std::max(1.0F,
        distance(hand.landmarks[2], hand.landmarks[5]));
    const HandLandmark palm_center = average({ hand.landmarks[0],
                                               hand.landmarks[5],
                                               hand.landmarks[9],
                                               hand.landmarks[13],
                                               hand.landmarks[17] });
    const HandLandmark contact_center = average({ hand.landmarks[4],
                                                  hand.landmarks[8] });
    const bool contact_matches =
        contact <= scale * options.thumb_index_contact_ratio &&
        contact <= mcp_span * options.thumb_index_contact_to_mcp_ratio &&
        distance(contact_center, palm_center) >=
            scale * options.contact_palm_distance_ratio;
    const auto extended = [&](std::size_t tip, std::size_t pip,
                              std::size_t mcp) {
        return finger_extended(hand, tip, pip, mcp,
                               options.extended_tip_beyond_pip_ratio,
                               options.extended_tip_mcp_distance_ratio);
    };
    return contact_matches && extended(12, 10, 9) && extended(16, 14, 13) &&
           extended(20, 18, 17);
}

float observation_confidence(const HandResult& hand)
{
    return std::min(hand.palm.confidence, hand.landmark_confidence);
}

thig::Observation observation(const GestureFrameContext& context,
                              int canonical_id, float confidence,
                              std::string relation, std::string producer)
{
    thig::Observation result;
    result.serial       = context.serial;
    result.source       = { "hand", canonical_id };
    result.relation     = std::move(relation);
    result.confidence   = confidence;
    result.producer     = std::move(producer);
    result.evidenceType = thig::EvidenceType::State;
    return result;
}

std::string model_shape(Gesture gesture)
{
    switch (gesture)
    {
    case Gesture::Open:
        return "Shape Open";
    case Gesture::Closed:
        return "Shape Fist";
    case Gesture::Pointer:
        return "Shape Pointer";
    case Gesture::Unknown:
    default:
        return "Shape Unclassified";
    }
}

void validate_options(const HandPrimitiveOptions& options)
{
    const auto finite = [](float value) { return std::isfinite(value); };
    if (options.max_hands == 0 || options.max_hands > kMaximumSupportedHands ||
        options.identity.maximum_identities == 0 ||
        options.identity.maximum_identities < options.max_hands)
    {
        throw std::invalid_argument(
            "hand primitive capacities must be positive and identity capacity "
            "must cover max_hands");
    }
    if (!finite(options.pose.minimum_confidence) || options.pose.minimum_confidence < 0.0F ||
        options.pose.minimum_confidence > 1.0F ||
        !finite(options.identity.minimum_confidence) ||
        options.identity.minimum_confidence < 0.0F ||
        options.identity.minimum_confidence > 1.0F)
    {
        throw std::invalid_argument("hand primitive confidence must be within [0, 1]");
    }
    if (options.identity.reacquire_frames <= 0 ||
        !std::isfinite(options.identity.maximum_distance_scale_ratio) ||
        options.identity.maximum_distance_scale_ratio <= 0.0F ||
        !std::isfinite(options.identity.maximum_linear_scale_ratio) ||
        options.identity.maximum_linear_scale_ratio < 1.0F ||
        !std::isfinite(options.identity.ambiguity_cost_margin) ||
        options.identity.ambiguity_cost_margin < 0.0F ||
        !std::isfinite(options.identity.scale_cost_weight) ||
        options.identity.scale_cost_weight < 0.0F ||
        !std::isfinite(options.identity.age_cost_weight) ||
        options.identity.age_cost_weight < 0.0F ||
        !std::isfinite(options.identity.raw_id_continuity_bonus) ||
        options.identity.raw_id_continuity_bonus < 0.0F ||
        !std::isfinite(options.identity.handedness_conflict_cost) ||
        options.identity.handedness_conflict_cost < 0.0F ||
        !std::isfinite(options.identity.velocity_observation_weight) ||
        options.identity.velocity_observation_weight < 0.0F ||
        options.identity.velocity_observation_weight > 1.0F ||
        options.identity.maximum_prediction_frames <= 0 ||
        !std::isfinite(options.identity.maximum_shape_distance) ||
        options.identity.maximum_shape_distance <= 0.0F ||
        options.identity.maximum_shape_distance > kMaximumHandShapeDistance ||
        !std::isfinite(options.identity.shape_cost_weight) ||
        options.identity.shape_cost_weight < 0.0F ||
        !std::isfinite(options.identity.shape_update_weight) ||
        options.identity.shape_update_weight <= 0.0F ||
        options.identity.shape_update_weight > 1.0F ||
        !std::isfinite(options.identity.maximum_appearance_distance) ||
        options.identity.maximum_appearance_distance <= 0.0F ||
        options.identity.maximum_appearance_distance > 2.0F ||
        !std::isfinite(options.identity.maximum_appearance_part_distance) ||
        options.identity.maximum_appearance_part_distance <= 0.0F ||
        options.identity.maximum_appearance_part_distance > 2.0F ||
        !std::isfinite(options.identity.appearance_cost_weight) ||
        options.identity.appearance_cost_weight < 0.0F ||
        !std::isfinite(options.identity.appearance_update_weight) ||
        options.identity.appearance_update_weight <= 0.0F ||
        options.identity.appearance_update_weight > 1.0F ||
        options.identity.minimum_comparable_appearance_parts == 0U ||
        options.identity.minimum_comparable_appearance_parts >
            vision_models::kHandAppearancePartCount)
    {
        throw std::invalid_argument("invalid canonical hand identity configuration");
    }
    if (options.pose.index_pressed_mcp_tip_palm_ratio < 0.0F ||
        !finite(options.pose.thumb_index_contact_ratio) ||
        options.pose.thumb_index_contact_ratio <= 0.0F ||
        !finite(options.pose.thumb_index_contact_to_mcp_ratio) ||
        options.pose.thumb_index_contact_to_mcp_ratio <= 0.0F ||
        !finite(options.pose.contact_palm_distance_ratio) ||
        options.pose.contact_palm_distance_ratio <= 0.0F ||
        !finite(options.pose.extended_tip_beyond_pip_ratio) ||
        options.pose.extended_tip_beyond_pip_ratio <= 0.0F ||
        !finite(options.pose.extended_tip_mcp_distance_ratio) ||
        options.pose.extended_tip_mcp_distance_ratio <= 0.0F ||
        !finite(options.pose.v_extended_tip_beyond_pip_ratio) ||
        options.pose.v_extended_tip_beyond_pip_ratio <= 0.0F ||
        !finite(options.pose.v_extended_tip_mcp_distance_ratio) ||
        options.pose.v_extended_tip_mcp_distance_ratio <= 0.0F ||
        !finite(options.pose.folded_tip_beyond_pip_max_ratio) ||
        options.pose.folded_tip_beyond_pip_max_ratio < 0.0F ||
        !finite(options.pose.folded_tip_mcp_distance_max_ratio) ||
        options.pose.folded_tip_mcp_distance_max_ratio <= 0.0F ||
        !finite(options.pose.v_tip_separation_ratio) ||
        options.pose.v_tip_separation_ratio <= 0.0F ||
        !finite(options.pose.index_extended_mcp_tip_palm_ratio) ||
        !finite(options.pose.index_pressed_mcp_tip_palm_ratio) ||
        options.pose.index_extended_mcp_tip_palm_ratio <=
            options.pose.index_pressed_mcp_tip_palm_ratio ||
        !finite(options.pose.index_pressed_pip_angle_degrees) ||
        !finite(options.pose.index_extended_pip_angle_degrees) ||
        options.pose.index_pressed_pip_angle_degrees < 0.0F ||
        options.pose.index_extended_pip_angle_degrees <=
            options.pose.index_pressed_pip_angle_degrees ||
        options.pose.index_extended_pip_angle_degrees > 180.0F)
    {
        throw std::invalid_argument("invalid index articulation thresholds");
    }
    const std::size_t maximum_pairs =
        options.max_hands > 1
            ? options.max_hands * (options.max_hands - 1U) / 2U
            : 0U;
    if (options.motion.max_history_samples == 0 ||
        options.motion.history_window_ms <= 0 ||
        options.motion.movement_window_samples < 2 ||
        options.motion.movement_window_samples >
            options.motion.max_history_samples ||
        options.motion.stationary_samples < 2 ||
        options.motion.stationary_samples > options.motion.max_history_samples ||
        options.motion.track_state_ttl_frames == 0 ||
        options.motion.empty_reset_frames == 0 ||
        !finite(options.motion.nominal_samples_per_second) ||
        options.motion.nominal_samples_per_second <= 0.0F ||
        !finite(options.motion.stop_speed_min_px_per_sample) ||
        options.motion.stop_speed_min_px_per_sample < 0.0F ||
        !finite(options.motion.stop_speed_hand_ratio) ||
        options.motion.stop_speed_hand_ratio < 0.0F ||
        !finite(options.motion.move_distance_min_px) ||
        options.motion.move_distance_min_px < 0.0F ||
        !finite(options.motion.move_distance_hand_ratio) ||
        options.motion.move_distance_hand_ratio < 0.0F ||
        !finite(options.motion.direction_distance_min_px) ||
        options.motion.direction_distance_min_px < 0.0F ||
        !finite(options.motion.direction_distance_hand_ratio) ||
        options.motion.direction_distance_hand_ratio < 0.0F ||
        !finite(options.motion.direction_dominance_ratio) ||
        options.motion.direction_dominance_ratio < 1.0F ||
        !finite(options.motion.stationary_low_speed_px_per_sample_at_30fps) ||
        options.motion.stationary_low_speed_px_per_sample_at_30fps < 0.0F ||
        !finite(options.motion.stationary_total_step_px) ||
        options.motion.stationary_total_step_px < 0.0F ||
        !finite(options.motion.stationary_displacement_px) ||
        options.motion.stationary_displacement_px < 0.0F ||
        !finite(options.motion.safe_area_inset_ratio) ||
        options.motion.safe_area_inset_ratio < 0.0F ||
        options.motion.safe_area_inset_ratio >= 0.5F)
    {
        throw std::invalid_argument("invalid bounded hand motion configuration");
    }
    if (options.spatial.window_ms <= 0 || options.spatial.minimum_samples < 2 ||
        options.spatial.max_samples_per_hand < options.spatial.minimum_samples ||
        options.spatial.max_pair_histories < maximum_pairs ||
        !finite(options.spatial.scale_change_ratio) ||
        options.spatial.scale_change_ratio <= 0.0F ||
        !finite(options.spatial.rotation_change_degrees) ||
        options.spatial.rotation_change_degrees <= 0.0F ||
        !finite(options.spatial.palm_axis_horizontal_max_degrees) ||
        options.spatial.palm_axis_horizontal_max_degrees <= 0.0F ||
        !finite(options.spatial.palm_axis_vertical_min_degrees) ||
        options.spatial.palm_axis_vertical_min_degrees >= kRightAngleDegrees ||
        options.spatial.palm_axis_horizontal_max_degrees >=
            options.spatial.palm_axis_vertical_min_degrees ||
        !finite(options.spatial.two_hand_distance_change_ratio) ||
        options.spatial.two_hand_distance_change_ratio <= 0.0F)
    {
        throw std::invalid_argument("invalid bounded hand spatial configuration");
    }
}

std::uint64_t pair_key(int first, int second)
{
    const auto lower = static_cast<std::uint32_t>(std::min(first, second));
    const auto upper = static_cast<std::uint32_t>(std::max(first, second));
    return (static_cast<std::uint64_t>(lower) << 32U) | upper;
}

float shortest_angle_delta(float first, float last)
{
    float delta = std::fmod(last - first, kFullTurnDegrees);
    if (delta > kHalfTurnDegrees)
    {
        delta -= kFullTurnDegrees;
    }
    else if (delta < -kHalfTurnDegrees)
    {
        delta += kFullTurnDegrees;
    }
    return delta;
}

std::string palm_axis_relation(float orientation_degrees,
                               const HandSpatialOptions& options)
{
    float axis_degrees = std::fmod(orientation_degrees, kHalfTurnDegrees);
    if (axis_degrees >= kRightAngleDegrees)
    {
        axis_degrees -= kHalfTurnDegrees;
    }
    else if (axis_degrees < -kRightAngleDegrees)
    {
        axis_degrees += kHalfTurnDegrees;
    }

    const float absolute_axis_degrees = std::fabs(axis_degrees);
    if (absolute_axis_degrees <= options.palm_axis_horizontal_max_degrees)
    {
        return "Palm Axis Horizontal";
    }
    if (absolute_axis_degrees >= options.palm_axis_vertical_min_degrees)
    {
        return "Palm Axis Vertical";
    }
    return "Palm Axis Diagonal";
}

std::string trend_relation(const char* prefix, float delta, float threshold,
                           const char* negative, const char* stable,
                           const char* positive)
{
    const char* value = stable;
    if (delta >= threshold)
    {
        value = positive;
    }
    else if (delta <= -threshold)
    {
        value = negative;
    }
    return std::string(prefix) + value;
}

template <typename Sample>
void prune_timed_history(std::deque<Sample>& history,
                         std::chrono::steady_clock::time_point now,
                         std::chrono::milliseconds window,
                         std::size_t max_samples)
{
    while (!history.empty() && now - history.front().observed_at > window)
    {
        history.pop_front();
    }
    while (history.size() > max_samples)
    {
        history.pop_front();
    }
}

detail::HandIdentityConfig identity_config(const HandIdentityOptions& options)
{
    detail::HandIdentityConfig result;
    result.reacquire_frames             = options.reacquire_frames;
    result.minimum_confidence            = options.minimum_confidence;
    result.maximum_distance_scale_ratio = options.maximum_distance_scale_ratio;
    result.maximum_linear_scale_ratio   = options.maximum_linear_scale_ratio;
    result.ambiguity_cost_margin        = options.ambiguity_cost_margin;
    result.scale_cost_weight            = options.scale_cost_weight;
    result.age_cost_weight              = options.age_cost_weight;
    result.raw_id_continuity_bonus      = options.raw_id_continuity_bonus;
    result.handedness_conflict_cost     = options.handedness_conflict_cost;
    result.velocity_observation_weight  = options.velocity_observation_weight;
    result.maximum_prediction_frames    = options.maximum_prediction_frames;
    result.maximum_identities            = options.maximum_identities;
    result.maximum_shape_distance        = options.maximum_shape_distance;
    result.shape_cost_weight             = options.shape_cost_weight;
    result.shape_update_weight           = options.shape_update_weight;
    result.maximum_appearance_distance   = options.maximum_appearance_distance;
    result.maximum_appearance_part_distance =
        options.maximum_appearance_part_distance;
    result.appearance_cost_weight        = options.appearance_cost_weight;
    result.appearance_update_weight      = options.appearance_update_weight;
    result.minimum_comparable_appearance_parts =
        options.minimum_comparable_appearance_parts;
    return result;
}

} // namespace

struct HandPrimitiveExtractor::Impl
{
    explicit Impl(const HandPrimitiveOptions& configured_options)
        : options(configured_options), identities(identity_config(options.identity))
    {
    }

    HandPrimitiveOptions options;
    detail::HandTrackIdentityRegistry identities;
    std::unordered_map<int, std::deque<TimedPoint>> motion_history;
    std::unordered_map<int, std::deque<TimedGeometry>> geometry_history;
    std::unordered_map<std::uint64_t, std::deque<TimedScalar>> pair_history;
    std::unordered_map<int, std::uint64_t> last_seen_frame;
    std::uint64_t frame_index = 0;
    std::uint64_t consecutive_empty_frames = 0;
    bool has_previous_context = false;
    std::uint64_t previous_serial = 0;
    std::chrono::steady_clock::time_point previous_time {};

    void append_hand_dynamics(const HandResult& hand, int canonical_id,
                              const PalmGeometry& geometry, float confidence,
                              const GestureFrameContext& context,
                              PrimitiveFrame& result)
    {
        result.observations.push_back(observation(
            context, canonical_id, confidence,
            palm_axis_relation(geometry.orientation_degrees, options.spatial),
            "palm_geometry"));

        auto& points = motion_history[canonical_id];
        points.push_back(
            { geometry.center_x, geometry.center_y, context.observed_at });
        prune_timed_history(points, context.observed_at,
            std::chrono::milliseconds(options.motion.history_window_ms),
            options.motion.max_history_samples);

        std::string direction = "Direction Neutral";
        float speed = 0.0F;
        if (points.size() >= options.motion.movement_window_samples)
        {
            const std::size_t first_index =
                points.size() - options.motion.movement_window_samples;
            const TimedPoint& first = points[first_index];
            const TimedPoint& last  = points.back();
            const float dx = last.x - first.x;
            const float dy = last.y - first.y;
            const float elapsed =
                std::chrono::duration<float>(last.observed_at - first.observed_at)
                    .count();
            if (elapsed > 0.0F)
            {
                const float nominal_samples =
                    elapsed * options.motion.nominal_samples_per_second;
                speed = std::hypot(dx / nominal_samples, dy / nominal_samples);
            }
            const float displacement = std::hypot(dx, dy);
            const float stop_speed = std::max(
                options.motion.stop_speed_min_px_per_sample,
                geometry.scale * options.motion.stop_speed_hand_ratio);
            const float move_distance = std::max(
                options.motion.move_distance_min_px,
                geometry.scale * options.motion.move_distance_hand_ratio);
            const float direction_distance = std::max(
                options.motion.direction_distance_min_px,
                geometry.scale * options.motion.direction_distance_hand_ratio);
            const bool stopped = speed <= stop_speed &&
                                 displacement < move_distance;
            if (!stopped)
            {
                if (dx >= direction_distance)
                {
                    direction = "Direction Right";
                }
                else if (dx <= -direction_distance)
                {
                    direction = "Direction Left";
                }
                else if (std::fabs(dy) >= direction_distance &&
                         std::fabs(dy) >=
                             std::fabs(dx) *
                                 options.motion.direction_dominance_ratio)
                {
                    direction = dy > 0.0F ? "Direction Down" : "Direction Up";
                }
                else
                {
                    direction = "Direction Unclassified";
                }
            }
        }
        result.observations.push_back(observation(
            context, canonical_id, confidence, std::move(direction),
            "landmark_kinematics"));

        bool stationary = points.size() >= options.motion.stationary_samples &&
                          speed <= options.motion
                                       .stationary_low_speed_px_per_sample_at_30fps;
        if (stationary)
        {
            const std::size_t first_index =
                points.size() - options.motion.stationary_samples;
            const float inset_x = static_cast<float>(context.image_width) *
                                  options.motion.safe_area_inset_ratio;
            const float inset_y = static_cast<float>(context.image_height) *
                                  options.motion.safe_area_inset_ratio;
            float total_step = 0.0F;
            for (std::size_t index = first_index; index < points.size(); ++index)
            {
                const TimedPoint& point = points[index];
                if (point.x < inset_x ||
                    point.x > static_cast<float>(context.image_width) - inset_x ||
                    point.y < inset_y ||
                    point.y > static_cast<float>(context.image_height) - inset_y)
                {
                    stationary = false;
                    break;
                }
                if (index > first_index)
                {
                    total_step += std::hypot(
                        point.x - points[index - 1U].x,
                        point.y - points[index - 1U].y);
                }
            }
            if (stationary)
            {
                const float displacement = std::hypot(
                    points.back().x - points[first_index].x,
                    points.back().y - points[first_index].y);
                stationary =
                    total_step <= options.motion.stationary_total_step_px &&
                    displacement <=
                        options.motion.stationary_displacement_px;
            }
        }
        result.observations.push_back(observation(
            context, canonical_id, confidence,
            stationary ? "Motion Stationary" : "Motion Unstable",
            "landmark_kinematics"));

        auto& geometries = geometry_history[canonical_id];
        geometries.push_back({ geometry, context.observed_at });
        prune_timed_history(
            geometries, context.observed_at,
            std::chrono::milliseconds(options.spatial.window_ms),
            options.spatial.max_samples_per_hand);
        std::string scale_relation    = "Scale Unclassified";
        std::string rotation_relation = "Rotation Unclassified";
        if (geometries.size() >= options.spatial.minimum_samples)
        {
            const PalmGeometry& first = geometries.front().geometry;
            const PalmGeometry& last  = geometries.back().geometry;
            const float scale_delta =
                first.normalized_scale > 0.0F
                    ? (last.normalized_scale - first.normalized_scale) /
                          first.normalized_scale
                    : 0.0F;
            scale_relation = trend_relation(
                "Scale ", scale_delta, options.spatial.scale_change_ratio,
                "Decreasing", "Stable", "Increasing");
            const float rotation_delta = shortest_angle_delta(
                first.orientation_degrees, last.orientation_degrees);
            rotation_relation = trend_relation(
                "Rotation ", rotation_delta,
                options.spatial.rotation_change_degrees,
                "CounterClockwise", "Stable", "Clockwise");
        }
        result.observations.push_back(observation(
            context, canonical_id, confidence, std::move(scale_relation),
            "palm_geometry"));
        result.observations.push_back(observation(
            context, canonical_id, confidence, std::move(rotation_relation),
            "palm_geometry"));
        (void)hand;
    }

    void append_pair_dynamics(
        const std::vector<int>& canonical_ids,
        const std::vector<PalmGeometry>& geometries,
        const std::vector<bool>& reliable,
        const std::vector<HandResult>& hands,
        const GestureFrameContext& context,
        PrimitiveFrame& result)
    {
        const float diagonal = std::hypot(
            static_cast<float>(context.image_width),
            static_cast<float>(context.image_height));
        std::unordered_set<std::uint64_t> active_pairs;
        for (std::size_t first_index = 0; first_index < hands.size(); ++first_index)
        {
            if (!reliable[first_index] || canonical_ids[first_index] <= 0)
            {
                continue;
            }
            for (std::size_t second_index = first_index + 1U;
                 second_index < hands.size(); ++second_index)
            {
                if (!reliable[second_index] || canonical_ids[second_index] <= 0 ||
                    canonical_ids[first_index] == canonical_ids[second_index])
                {
                    continue;
                }
                const int source_id = std::min(canonical_ids[first_index],
                                               canonical_ids[second_index]);
                const int target_id = std::max(canonical_ids[first_index],
                                               canonical_ids[second_index]);
                const std::uint64_t key = pair_key(source_id, target_id);
                active_pairs.insert(key);
                const float normalized_distance = std::hypot(
                    geometries[second_index].center_x -
                        geometries[first_index].center_x,
                    geometries[second_index].center_y -
                        geometries[first_index].center_y) /
                    diagonal;
                auto& history = pair_history[key];
                history.push_back({ normalized_distance, context.observed_at });
                prune_timed_history(
                    history, context.observed_at,
                    std::chrono::milliseconds(options.spatial.window_ms),
                    options.spatial.max_samples_per_hand);
                std::string relation = "Hands Distance Unclassified";
                if (history.size() >= options.spatial.minimum_samples)
                {
                    relation = trend_relation(
                        "Hands Distance ",
                        history.back().value - history.front().value,
                        options.spatial.two_hand_distance_change_ratio,
                        "Contracting", "Stable", "Expanding");
                }
                const float confidence = std::min(
                    observation_confidence(hands[first_index]),
                    observation_confidence(hands[second_index]));
                thig::Observation pair_observation = observation(
                    context, source_id, confidence, std::move(relation),
                    "pair_geometry");
                pair_observation.target = thig::EntityRef { "hand", target_id };
                result.observations.push_back(std::move(pair_observation));
            }
        }
        for (auto iterator = pair_history.begin(); iterator != pair_history.end();)
        {
            if (active_pairs.find(iterator->first) == active_pairs.end())
            {
                iterator = pair_history.erase(iterator);
            }
            else
            {
                ++iterator;
            }
        }
    }

    void cleanup(const std::unordered_set<int>& active_ids)
    {
        if (consecutive_empty_frames >= options.motion.empty_reset_frames)
        {
            motion_history.clear();
            geometry_history.clear();
            pair_history.clear();
        }
        for (auto iterator = last_seen_frame.begin();
             iterator != last_seen_frame.end();)
        {
            if (active_ids.find(iterator->first) != active_ids.end() ||
                frame_index - iterator->second <=
                    options.motion.track_state_ttl_frames)
            {
                ++iterator;
                continue;
            }
            motion_history.erase(iterator->first);
            geometry_history.erase(iterator->first);
            iterator = last_seen_frame.erase(iterator);
        }
    }
};

HandPrimitiveExtractor::HandPrimitiveExtractor(const HandPrimitiveOptions& options)
{
    validate_options(options);
    impl_ = std::make_unique<Impl>(options);
}

HandPrimitiveExtractor::~HandPrimitiveExtractor() = default;
HandPrimitiveExtractor::HandPrimitiveExtractor(HandPrimitiveExtractor&&) noexcept = default;
HandPrimitiveExtractor& HandPrimitiveExtractor::operator=(HandPrimitiveExtractor&&) noexcept = default;

PrimitiveFrame HandPrimitiveExtractor::process(
    const vision_models::HandFrame& frame,
    const GestureFrameContext& context)
{
    if (context.image_width <= 0 || context.image_height <= 0)
    {
        throw std::invalid_argument("gesture frame dimensions must be positive");
    }
    if (frame.hands.size() > impl_->options.max_hands)
    {
        throw std::length_error("gesture frame exceeds max_hands");
    }
    const std::size_t potential_pairs =
        frame.hands.size() > 1U
            ? frame.hands.size() * (frame.hands.size() - 1U) / 2U
            : 0U;
    if (potential_pairs > impl_->options.spatial.max_pair_histories)
    {
        throw std::length_error("gesture frame exceeds pair history capacity");
    }
    if (impl_->has_previous_context &&
        (context.serial <= impl_->previous_serial ||
         context.observed_at < impl_->previous_time))
    {
        throw std::invalid_argument(
            "gesture frame serial and timestamp must be monotonic");
    }

    std::vector<detail::HandIdentityObservation> identity_observations;
    identity_observations.reserve(frame.hands.size());
    std::vector<PalmGeometry> geometries(frame.hands.size());
    std::vector<bool> geometry_valid(frame.hands.size(), false);
    for (std::size_t index = 0; index < frame.hands.size(); ++index)
    {
        const HandResult& hand = frame.hands[index];
        geometry_valid[index] = estimate_palm_geometry(
            hand, context.image_width, context.image_height, geometries[index]);
        const float confidence = observation_confidence(hand);
        const auto shape = detail::MakeHandShapeDescriptor(hand.landmarks);
        identity_observations.push_back(
            { hand.track_id,
              geometries[index].center_x,
              geometries[index].center_y,
              geometries[index].scale,
              geometry_valid[index] ? confidence : 0.0F,
              shape,
              hand.appearance,
              hand.handedness });
    }

    const std::vector<detail::HandIdentityResolution> identity_resolutions =
        impl_->identities.Resolve(identity_observations);
    std::vector<int> canonical_ids;
    canonical_ids.reserve(identity_resolutions.size());
    for (const auto& resolution : identity_resolutions)
    {
        canonical_ids.push_back(resolution.canonical_id);
    }
    PrimitiveFrame result;
    result.hands.reserve(frame.hands.size());
    result.observations.reserve(frame.hands.size() * 8U + potential_pairs);
    ++impl_->frame_index;
    impl_->consecutive_empty_frames = frame.hands.empty()
        ? std::min(impl_->consecutive_empty_frames + 1U,
                   impl_->options.motion.empty_reset_frames)
        : 0U;
    std::vector<bool> reliable(frame.hands.size(), false);
    std::unordered_set<int> active_ids;
    for (std::size_t index = 0; index < frame.hands.size(); ++index)
    {
        const HandResult& hand = frame.hands[index];
        const int canonical_id = canonical_ids[index];
        result.hands.push_back(
            { index, hand.track_id, canonical_id,
              identity_resolutions[index].association });
        const float confidence = observation_confidence(hand);
        if (canonical_id <= 0 || !geometry_valid[index] ||
            confidence < impl_->options.pose.minimum_confidence)
        {
            continue;
        }
        reliable[index] = true;
        active_ids.insert(canonical_id);
        impl_->last_seen_frame[canonical_id] = impl_->frame_index;

        const bool v_pose = is_v_pose(hand, confidence, impl_->options.pose);
        result.observations.push_back(observation(
            context, canonical_id, confidence,
            v_pose ? "Shape V" : model_shape(hand.gesture),
            v_pose ? "shape_geometry" : "shape_model"));

        IndexGeometry index_geometry;
        const IndexPose index_pose = estimate_index_geometry(hand, index_geometry)
                                         ? classify_index(index_geometry,
                                                          impl_->options.pose)
                                         : IndexPose::Unknown;
        std::string index_relation = "Index Unclassified";
        if (index_pose == IndexPose::Extended)
        {
            index_relation = "Index Extended";
        }
        else if (index_pose == IndexPose::Pressed)
        {
            index_relation = "Index Pressed";
        }
        else if (index_pose == IndexPose::Intermediate)
        {
            index_relation = "Index Intermediate";
        }
        result.observations.push_back(observation(
            context, canonical_id, confidence, std::move(index_relation),
            "index_geometry"));

        result.observations.push_back(observation(
            context, canonical_id, confidence,
            is_ok_pose(hand, confidence, impl_->options.pose) ? "Pose OK"
                                                              : "Pose Not OK",
            "pose_geometry"));

        impl_->append_hand_dynamics(hand, canonical_id, geometries[index],
                                    confidence, context, result);
    }

    impl_->append_pair_dynamics(canonical_ids, geometries, reliable,
                                frame.hands, context, result);
    impl_->cleanup(active_ids);

    impl_->has_previous_context = true;
    impl_->previous_serial      = context.serial;
    impl_->previous_time        = context.observed_at;
    return result;
}

void HandPrimitiveExtractor::reset()
{
    impl_->identities.Reset();
    impl_->motion_history.clear();
    impl_->geometry_history.clear();
    impl_->pair_history.clear();
    impl_->last_seen_frame.clear();
    impl_->frame_index = 0;
    impl_->consecutive_empty_frames = 0;
    impl_->has_previous_context = false;
    impl_->previous_serial      = 0;
    impl_->previous_time        = {};
}

HandPrimitiveExtractor HandPrimitiveExtractor::clone() const
{
    HandPrimitiveExtractor result(impl_->options);
    result.impl_ = std::make_unique<Impl>(*impl_);
    return result;
}

} // namespace kfcore::hand_interaction
