#include "hand_interaction_demo_ui.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

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
constexpr int    kBoxThickness = 2;
constexpr int    kPointRadius = 2;
constexpr int    kTextThickness = 1;
constexpr int    kTextLineHeight = 20;
constexpr int    kTextMargin = 8;
constexpr int    kMaximumActionLines = 4;
constexpr double kTextScale = 0.48;

int clamp_coordinate(float value, int extent)
{
    if (!std::isfinite(value) || extent <= 0)
    {
        return 0;
    }
    return std::clamp(static_cast<int>(std::lround(value)), 0, extent - 1);
}

std::string gesture_name(vision_models::Gesture gesture)
{
    switch (gesture)
    {
    case vision_models::Gesture::Open:
        return "Open";
    case vision_models::Gesture::Closed:
        return "Closed";
    case vision_models::Gesture::Pointer:
        return "Pointer";
    default:
        return "Unknown";
    }
}

int canonical_id_for(const HandInteractionFrame& interaction, std::size_t input_index)
{
    for (const CanonicalHand& hand : interaction.primitives.hands)
    {
        if (hand.input_index == input_index)
        {
            return hand.canonical_id;
        }
    }
    return 0;
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

std::array<std::string, 3> format_timing_lines(const DemoMetrics& metrics)
{
    std::array<std::ostringstream, 3> text;
    for (auto& line : text)
    {
        line << std::fixed << std::setprecision(2);
    }
    text[0] << "ms conv " << metrics.convert_ms << " pre "
            << metrics.model.preprocess_ms << " palm "
            << metrics.model.palm_inference_ms << " land "
            << metrics.model.landmark_inference_ms;
    text[1] << "ms cls " << metrics.model.classifier_inference_ms << " track "
            << metrics.model.tracking_ms << " model " << metrics.model.total_ms
            << " thig " << metrics.thig_ms << " pipe " << metrics.frame_ms;
    text[2] << "ms face-pre " << metrics.face.detection_preprocess_ms
            << " face-det " << metrics.face.detection_inference_ms
            << " mesh-pre " << metrics.face.landmark_preprocess_ms
            << " mesh " << metrics.face.landmark_inference_ms
            << " face " << metrics.face.total_ms;
    return { text[0].str(), text[1].str(), text[2].str() };
}

cv::Mat compose_overlay(const cv::Mat& source, const vision_models::HandFrame& hands,
                        const HandInteractionFrame& interaction,
                        const vision_models::FaceMeshFrame* face,
                        const DemoMetrics& metrics)
{
    if (source.empty() || source.type() != CV_8UC3)
    {
        throw std::invalid_argument("hand interaction demo source must be non-empty CV_8UC3");
    }

    cv::Mat output = source.clone();
    const int action_lines =
        std::min(static_cast<int>(interaction.actions.size()), kMaximumActionLines);
    const int face_timing_lines = face != nullptr ? 1 : 0;
    const auto timing_lines = format_timing_lines(metrics);
    draw_status_background(output, 4 + face_timing_lines + action_lines);
    cv::putText(output, "R reset tracking/THIG | Q/Esc quit",
                cv::Point(kTextMargin, kTextMargin + kTextLineHeight),
                cv::FONT_HERSHEY_SIMPLEX, kTextScale, kPrimaryTextColor,
                kTextThickness, cv::LINE_AA);
    cv::putText(output, timing_lines[0],
                cv::Point(kTextMargin, kTextMargin + 2 * kTextLineHeight),
                cv::FONT_HERSHEY_SIMPLEX, kTextScale, kPrimaryTextColor,
                kTextThickness, cv::LINE_AA);
    cv::putText(output, timing_lines[1],
                cv::Point(kTextMargin, kTextMargin + 3 * kTextLineHeight),
                cv::FONT_HERSHEY_SIMPLEX, kTextScale, kPrimaryTextColor,
                kTextThickness, cv::LINE_AA);
    cv::putText(output, capture_line(metrics),
                cv::Point(kTextMargin,
                          kTextMargin + (4 + face_timing_lines) * kTextLineHeight),
                cv::FONT_HERSHEY_SIMPLEX, kTextScale, kPrimaryTextColor,
                kTextThickness, cv::LINE_AA);
    if (face != nullptr)
    {
        cv::putText(output, timing_lines[2],
                    cv::Point(kTextMargin, kTextMargin + 4 * kTextLineHeight),
                    cv::FONT_HERSHEY_SIMPLEX, kTextScale, kPrimaryTextColor,
                    kTextThickness, cv::LINE_AA);
    }

    for (int index = 0; index < action_lines; ++index)
    {
        const auto& action = interaction.actions[static_cast<std::size_t>(index)];
        cv::putText(output, "ACTION: " + action.action,
                    cv::Point(kTextMargin,
                              kTextMargin + (5 + face_timing_lines + index) *
                                  kTextLineHeight),
                    cv::FONT_HERSHEY_SIMPLEX, kTextScale, kActionTextColor,
                    kTextThickness, cv::LINE_AA);
    }

    for (std::size_t index = 0U; index < hands.hands.size(); ++index)
    {
        const auto& hand = hands.hands[index];
        const int left = clamp_coordinate(hand.palm.box.x, output.cols);
        const int top = clamp_coordinate(hand.palm.box.y, output.rows);
        const int right = clamp_coordinate(hand.palm.box.x + hand.palm.box.width,
                                           output.cols);
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
                       cv::Point(clamp_coordinate(landmark.x, output.cols),
                                 clamp_coordinate(landmark.y, output.rows)),
                       kPointRadius, kLandmarkColor, cv::FILLED, cv::LINE_AA);
        }

        std::ostringstream label;
        label << "hand " << canonical_id_for(interaction, index) << " track "
              << hand.track_id << " " << gesture_name(hand.gesture);
        cv::putText(output, label.str(), cv::Point(left, std::max(15, top - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, kTextScale, kBoxColor,
                    kTextThickness, cv::LINE_AA);
    }
    if (face != nullptr && face->detection.has_value())
    {
        const auto& box = face->detection->box;
        const int left = clamp_coordinate(box.x, output.cols);
        const int top = clamp_coordinate(box.y, output.rows);
        const int right = clamp_coordinate(box.x + box.width, output.cols);
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
                           cv::Point(clamp_coordinate(landmark.x, output.cols),
                                     clamp_coordinate(landmark.y, output.rows)),
                           1, kFaceLandmarkColor, cv::FILLED, cv::LINE_AA);
            }
        }
    }
    return output;
}

} // namespace kfcore::hand_interaction::demo
