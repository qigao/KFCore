#pragma once

#include "hand_interaction_demo_capture.hpp"

#include <opencv2/core.hpp>

namespace kfcore::hand_interaction::demo
{

// Returns independent packed BGR storage; no mailbox view is retained.
[[nodiscard]] cv::Mat to_bgr(const CapturedFrame& frame);

} // namespace kfcore::hand_interaction::demo
