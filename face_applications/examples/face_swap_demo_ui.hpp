#pragma once

#include <opencv2/core.hpp>

#include <string_view>

namespace kfcore::face_applications::demo
{

inline constexpr int kPanelWidth   = 400;
inline constexpr int kPanelHeight  = 400;
inline constexpr int kCanvasGap    = 16;
inline constexpr int kHeaderHeight = 42;
inline constexpr int kFooterHeight = 58;
inline constexpr int kCanvasWidth  = 3 * kPanelWidth + 4 * kCanvasGap;
inline constexpr int kCanvasHeight =
    kHeaderHeight + kPanelHeight + kFooterHeight + 2 * kCanvasGap;

enum class DemoAction
{
    None,
    Rerun,
    Save,
    Quit,
};

[[nodiscard]] DemoAction action_from_key(int key) noexcept;

// Source and target must be non-empty CV_8UC3 images. Result may be empty while
// inference is pending; a non-empty result follows the same image contract.
[[nodiscard]] cv::Mat compose_canvas(const cv::Mat& source, const cv::Mat& target,
                                     const cv::Mat& result, std::string_view status);

} // namespace kfcore::face_applications::demo
