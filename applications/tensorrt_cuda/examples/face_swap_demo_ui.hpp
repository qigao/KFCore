#pragma once

#include "face_swap_demo_metrics.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <string_view>
#include <vector>

namespace kfcore::face_applications::demo
{

inline constexpr int kPanelWidth   = 320;
inline constexpr int kPanelHeight  = 280;
inline constexpr int kCanvasGap    = 16;
inline constexpr int kHeaderHeight = 42;
inline constexpr int kFooterHeight = 300;
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
                                     const cv::Mat& result, std::string_view status,
                                     const std::vector<TimingRow>& timing_rows = {},
                                     std::size_t timing_samples = 0U);

} // namespace kfcore::face_applications::demo
