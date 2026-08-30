#include "hand_interaction_demo_ui.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace kfcore::hand_interaction::demo
{
namespace
{

const cv::Scalar kBoxColor { 40, 220, 80 };
const cv::Scalar kLandmarkColor { 20, 180, 255 };
const cv::Scalar kFaceBoxColor { 255, 180, 40 };
const cv::Scalar kFaceLandmarkColor { 255, 80, 180 };
const cv::Scalar kPrimaryTextColor { 245, 245, 245 };
const cv::Scalar kActionTextColor { 60, 230, 255 };
const cv::Scalar kTextBackground { 24, 24, 24 };
const cv::Scalar kTextOutlineColor { 0, 0, 0 };
constexpr int    kBoxThickness = 2;
constexpr int    kPointRadius = 2;
constexpr int    kTextThickness = 2;
constexpr int    kTextOutlineThickness = 4;
constexpr int    kTextLineHeight = 20;
constexpr int    kTextMargin = 8;
constexpr double kTextScale = 0.52;

int clamp_coordinate(float value, int extent)
{
    if (!std::isfinite(value) || extent <= 0)
    {
        return 0;
    }
    return std::clamp(static_cast<int>(std::lround(value)), 0, extent - 1);
}

int mirror_coordinate(float value, int extent)
{
    return extent - 1 - clamp_coordinate(value, extent);
}

std::string gesture_name(hand_models::Gesture gesture)
{
    switch (gesture)
    {
    case hand_models::Gesture::Open:
        return "Open";
    case hand_models::Gesture::Closed:
        return "Closed";
    case hand_models::Gesture::Pointer:
        return "Pointer";
    default:
        return "Unknown";
    }
}

const CanonicalHand* canonical_hand_for(const HandInteractionFrame& interaction,
                                        std::size_t input_index)
{
    for (const CanonicalHand& hand : interaction.primitives.hands)
    {
        if (hand.input_index == input_index)
        {
            return &hand;
        }
    }
    return nullptr;
}

const char* association_name(HandIdentityAssociation association) noexcept
{
    switch (association)
    {
    case HandIdentityAssociation::NewIdentity:
        return "New";
    case HandIdentityAssociation::RawTrackContinuity:
        return "Raw";
    case HandIdentityAssociation::ShapeReacquired:
        return "Shape";
    case HandIdentityAssociation::AppearanceReacquired:
        return "Appearance";
    case HandIdentityAssociation::Ambiguous:
        return "Ambiguous";
    case HandIdentityAssociation::UnreliableObservation:
    default:
        return "Unreliable";
    }
}

const char* handedness_name(hand_models::Handedness handedness) noexcept
{
    switch (handedness)
    {
    case hand_models::Handedness::Left:
        return "Left";
    case hand_models::Handedness::Right:
        return "Right";
    case hand_models::Handedness::Unknown:
    default:
        return "Unknown";
    }
}

void draw_readable_text(cv::Mat& image, const std::string& text,
                        const cv::Point& origin, const cv::Scalar& color)
{
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, kTextScale,
                kTextOutlineColor, kTextOutlineThickness, cv::LINE_AA);
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, kTextScale,
                color, kTextThickness, cv::LINE_AA);
}

std::string capture_line(const DemoMetrics& metrics)
{
    std::ostringstream text;
    text << "captured " << metrics.capture.captured_frames << " | consumed "
         << metrics.capture.consumed_frames << " | coalesced "
         << metrics.capture.coalesced_frames << " | rejected "
         << metrics.capture.rejected_frames;
    return text.str();
}

void draw_status_background(cv::Mat& image, int line_count)
{
    const int height = std::min(image.rows, kTextMargin * 2 + line_count * kTextLineHeight);
    if (height > 0)
    {
        cv::rectangle(image, cv::Rect(0, 0, image.cols, height), kTextBackground, cv::FILLED);
    }
}

} // namespace

DemoAction action_from_key(int key) noexcept
{
    if (key < 0)
    {
        return DemoAction::None;
    }
    switch (key & 0xff)
    {
    case 27:
    case 'q':
    case 'Q':
        return DemoAction::Quit;
    case 'r':
    case 'R':
        return DemoAction::Reset;
    default:
        return DemoAction::None;
    }
}

bool HandIdentityDiagnostic::operator==(
    const HandIdentityDiagnostic& other) const noexcept
{
    return input_index == other.input_index &&
           raw_track_id == other.raw_track_id &&
           canonical_id == other.canonical_id &&
           association == other.association &&
           handedness == other.handedness &&
           appearance_parts == other.appearance_parts;
}

RecentActionHistory::RecentActionHistory(std::chrono::milliseconds retention,
                                         std::size_t maximum_actions)
    : retention_(retention)
    , maximum_actions_(maximum_actions)
{
    if (retention_.count() <= 0 || maximum_actions_ == 0U)
    {
        throw std::invalid_argument("recent action history requires positive bounds");
    }
    entries_.reserve(maximum_actions_);
}

std::vector<thig::ActionEvent> RecentActionHistory::update(
    const std::vector<thig::ActionEvent>& actions,
    std::chrono::steady_clock::time_point now)
{
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [now](const Entry& entry)
                                  { return entry.expires_at <= now; }),
                   entries_.end());
    for (const thig::ActionEvent& action : actions)
    {
        if (entries_.size() == maximum_actions_)
        {
            entries_.erase(entries_.begin());
        }
        entries_.push_back({ action, now + retention_ });
    }

    std::vector<thig::ActionEvent> result;
    result.reserve(entries_.size());
    for (const Entry& entry : entries_)
    {
        result.push_back(entry.action);
    }
    return result;
}

void RecentActionHistory::reset() noexcept
{
    entries_.clear();
}

std::array<std::string, 3> format_timing_lines(const DemoMetrics& metrics)
{
    std::array<std::ostringstream, 3> text;
    for (auto& line : text)
    {
        line << std::fixed << std::setprecision(2);
    }
    text[0] << "FPS " << metrics.fps << " | ms conv " << metrics.convert_ms << " pre "
            << metrics.model.preprocess_ms << " palm "
            << metrics.model.palm_inference_ms << " land "
            << metrics.model.landmark_inference_ms;
    text[1] << "ms cls " << metrics.model.classifier_inference_ms << " reid "
            << metrics.model.appearance_ms << " track "
            << metrics.model.tracking_ms << " model " << metrics.model.total_ms
            << " thig " << metrics.thig_ms << " pipe " << metrics.frame_ms;
    text[2] << "ms face-pre " << metrics.face.detection_preprocess_ms
            << " face-det " << metrics.face.detection_inference_ms
            << " mesh-pre " << metrics.face.landmark_preprocess_ms
            << " mesh " << metrics.face.landmark_inference_ms
            << " face " << metrics.face.total_ms;
    return { text[0].str(), text[1].str(), text[2].str() };
}

HandOverlayText format_hand_overlay_text(const CanonicalHand& identity_result,
                                         const hand_models::HandResult& hand,
                                         const PrimitiveFrame& primitives)
{
    const int canonical_id = identity_result.canonical_id;
    std::string shape     = "Unknown";
    std::string motion    = "Unknown";
    std::string direction = "Unknown";
    std::string rotation  = "Unknown";
    std::string axis      = "Unknown";
    bool        ok        = false;
    for (const thig::Observation& observation : primitives.observations)
    {
        if (observation.source.kind != "hand" ||
            observation.source.id != canonical_id)
        {
            continue;
        }
        if (observation.relation.compare(0U, 6U, "Shape ") == 0)
        {
            shape = observation.relation.substr(6U);
        }
        else if (observation.relation.compare(0U, 7U, "Motion ") == 0)
        {
            motion = observation.relation.substr(7U);
        }
        else if (observation.relation.compare(0U, 10U, "Direction ") == 0)
        {
            direction = observation.relation.substr(10U);
        }
        else if (observation.relation.compare(0U, 9U, "Rotation ") == 0)
        {
            rotation = observation.relation.substr(9U);
        }
        else if (observation.relation.compare(0U, 10U, "Palm Axis ") == 0)
        {
            axis = observation.relation.substr(10U);
        }
        else if (observation.relation == "Pose OK")
        {
            ok = true;
        }
    }

    HandOverlayText result;
    std::ostringstream identity;
    identity << "Hand ID:" << canonical_id << " | Track ID:" << hand.track_id
             << " | Match:" << association_name(identity_result.association)
             << " | Raw:" << gesture_name(hand.gesture);
    result.identity = identity.str();

    std::ostringstream derived;
    derived << "Derived:" << shape << " | Motion:" << motion
            << " | Axis:" << axis;
    result.derived = derived.str();

    std::ostringstream dynamics;
    dynamics << "Direction:" << direction << " | Rotation:" << rotation;
    if (ok)
    {
        dynamics << " | Pose:OK";
    }
    result.dynamics = dynamics.str();
    return result;
}

std::vector<HandIdentityDiagnostic> make_identity_diagnostics(
    const hand_models::HandFrame& hands,
    const HandInteractionFrame& interaction)
{
    std::vector<HandIdentityDiagnostic> result;
    result.reserve(hands.hands.size());
    for (std::size_t index = 0U; index < hands.hands.size(); ++index)
    {
        const hand_models::HandResult& hand = hands.hands[index];
        const CanonicalHand* identity = canonical_hand_for(interaction, index);
        result.push_back(
            { index,
              hand.track_id,
              identity != nullptr ? identity->canonical_id : 0,
              identity != nullptr
                  ? identity->association
                  : HandIdentityAssociation::UnreliableObservation,
              hand.handedness,
              hand.appearance.has_value()
                  ? hand.appearance->valid_parts
                  : static_cast<std::uint8_t>(0U) });
    }
    return result;
}

std::string format_identity_diagnostic_line(
    std::uint64_t frame_serial, const hand_models::HandFrame& hands,
    const HandInteractionFrame& interaction)
{
    const std::vector<HandIdentityDiagnostic> diagnostics =
        make_identity_diagnostics(hands, interaction);
    std::ostringstream line;
    line << "[ID] frame=" << frame_serial << " hands=" << diagnostics.size();
    for (const HandIdentityDiagnostic& diagnostic : diagnostics)
    {
        const hand_models::HandResult& hand =
            hands.hands[diagnostic.input_index];
        const hand_models::RectF& box = hand.palm.box;
        line << " | input=" << diagnostic.input_index
             << " raw=" << diagnostic.raw_track_id
             << " hand=" << diagnostic.canonical_id
             << " match=" << association_name(diagnostic.association)
             << " side=" << handedness_name(diagnostic.handedness)
             << " parts=0x" << std::hex << std::setw(2) << std::setfill('0')
             << static_cast<unsigned int>(diagnostic.appearance_parts)
             << std::dec << std::setfill(' ')
             << std::fixed << std::setprecision(1)
             << " center=(" << box.x + box.width * 0.5F << ','
             << box.y + box.height * 0.5F << ')'
             << " scale=" << std::max(box.width, box.height)
             << std::setprecision(2)
             << " confidence=(" << hand.palm.confidence << ','
             << hand.landmark_confidence << ')';
    }
    return line.str();
}

std::string format_thig_state_line(const DemoThigStatus& status)
{
    const auto state_or_unknown = [](const std::string& state) -> const std::string&
    {
        static const std::string unknown = "unknown";
        return state.empty() ? unknown : state;
    };
    std::ostringstream line;
    line << "THIG hand=" << state_or_unknown(status.hand_state)
         << " | wave=" << state_or_unknown(status.wave_state)
         << " | click=" << state_or_unknown(status.click_state);
    return line.str();
}

std::string format_action_line(const thig::ActionEvent& action)
{
    std::ostringstream line;
    line << "ACTION: " << action.action << " | ";
    if (action.source.kind == "hand")
    {
        line << "Hand ID:" << action.source.id;
    }
    else
    {
        line << action.source.kind << " ID:" << action.source.id;
    }
    return line.str();
}

cv::Mat compose_overlay(const cv::Mat& source, const hand_models::HandFrame& hands,
                        const HandInteractionFrame& interaction,
                        const DemoThigStatus& status,
                        const face_models::FaceMeshFrame* face,
                        const DemoMetrics& metrics)
{
    if (source.empty() || source.type() != CV_8UC3)
    {
        throw std::invalid_argument("hand interaction demo source must be non-empty CV_8UC3");
    }

    cv::Mat output;
    cv::flip(source, output, 1);
    const int action_lines = std::min(
        static_cast<int>(status.recent_actions.size()),
        static_cast<int>(kMaximumDisplayedActions));
    const int face_timing_lines = face != nullptr ? 1 : 0;
    const auto timing_lines = format_timing_lines(metrics);
    draw_status_background(output, 5 + face_timing_lines + action_lines);
    draw_readable_text(output, "R reset tracking/THIG | Q/Esc quit",
                       cv::Point(kTextMargin, kTextMargin + kTextLineHeight),
                       kPrimaryTextColor);
    draw_readable_text(
        output, timing_lines[0],
        cv::Point(kTextMargin, kTextMargin + 2 * kTextLineHeight),
        kPrimaryTextColor);
    draw_readable_text(
        output, timing_lines[1],
        cv::Point(kTextMargin, kTextMargin + 3 * kTextLineHeight),
        kPrimaryTextColor);
    draw_readable_text(
        output, capture_line(metrics),
        cv::Point(kTextMargin,
                  kTextMargin + (4 + face_timing_lines) * kTextLineHeight),
        kPrimaryTextColor);
    if (face != nullptr)
    {
        draw_readable_text(
            output, timing_lines[2],
            cv::Point(kTextMargin, kTextMargin + 4 * kTextLineHeight),
            kPrimaryTextColor);
    }

    draw_readable_text(
        output, format_thig_state_line(status),
        cv::Point(kTextMargin,
                  kTextMargin + (5 + face_timing_lines) * kTextLineHeight),
        kPrimaryTextColor);

    for (int index = 0; index < action_lines; ++index)
    {
        const auto& action = status.recent_actions[static_cast<std::size_t>(index)];
        draw_readable_text(
            output, format_action_line(action),
            cv::Point(kTextMargin,
                      kTextMargin + (6 + face_timing_lines + index) *
                          kTextLineHeight),
            kActionTextColor);
    }

    for (std::size_t index = 0U; index < hands.hands.size(); ++index)
    {
        const auto& hand = hands.hands[index];
        const int source_left = clamp_coordinate(hand.palm.box.x, output.cols);
        const int top = clamp_coordinate(hand.palm.box.y, output.rows);
        const int source_right = clamp_coordinate(
            hand.palm.box.x + hand.palm.box.width, output.cols);
        const int left = output.cols - 1 - source_right;
        const int right = output.cols - 1 - source_left;
        const int bottom = clamp_coordinate(hand.palm.box.y + hand.palm.box.height,
                                            output.rows);
        if (right > left && bottom > top)
        {
            cv::rectangle(output, cv::Point(left, top), cv::Point(right, bottom),
                          kBoxColor, kBoxThickness, cv::LINE_AA);
        }
        for (const auto& landmark : hand.landmarks)
        {
            cv::circle(output,
                       cv::Point(mirror_coordinate(landmark.x, output.cols),
                                 clamp_coordinate(landmark.y, output.rows)),
                       kPointRadius, kLandmarkColor, cv::FILLED, cv::LINE_AA);
        }

        const CanonicalHand missing_identity {
            index, hand.track_id, 0,
            HandIdentityAssociation::UnreliableObservation
        };
        const CanonicalHand* identity = canonical_hand_for(interaction, index);
        const HandOverlayText label = format_hand_overlay_text(
            identity != nullptr ? *identity : missing_identity, hand,
            interaction.primitives);
        const int identity_y = std::max(15, top - kTextLineHeight - 6);
        draw_readable_text(output, label.identity, cv::Point(left, identity_y),
                           kPrimaryTextColor);
        draw_readable_text(output, label.derived,
                           cv::Point(left, identity_y + kTextLineHeight),
                           kPrimaryTextColor);
        draw_readable_text(output, label.dynamics,
                           cv::Point(left, identity_y + 2 * kTextLineHeight),
                           kPrimaryTextColor);
    }
    if (face != nullptr && face->detection.has_value())
    {
        const auto& box = face->detection->box;
        const int source_left = clamp_coordinate(box.x, output.cols);
        const int top = clamp_coordinate(box.y, output.rows);
        const int source_right = clamp_coordinate(box.x + box.width, output.cols);
        const int left = output.cols - 1 - source_right;
        const int right = output.cols - 1 - source_left;
        const int bottom = clamp_coordinate(box.y + box.height, output.rows);
        if (right > left && bottom > top)
        {
            cv::rectangle(output, cv::Point(left, top), cv::Point(right, bottom),
                          kFaceBoxColor, kBoxThickness, cv::LINE_AA);
        }
        if (face->landmarks.has_value())
        {
            for (const auto& landmark : face->landmarks->landmarks)
            {
                cv::circle(output,
                           cv::Point(mirror_coordinate(landmark.x, output.cols),
                                     clamp_coordinate(landmark.y, output.rows)),
                           1, kFaceLandmarkColor, cv::FILLED, cv::LINE_AA);
            }
        }
    }
    return output;
}

} // namespace kfcore::hand_interaction::demo
