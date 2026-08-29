#pragma once

#include "yolo_domain_capture.hpp"

#include <opencv2/core.hpp>

namespace kfcore::yolo::demo
{

// Returns independent packed BGR storage; no mailbox view is retained.
[[nodiscard]] cv::Mat to_bgr(const CapturedFrame& frame);

} // namespace kfcore::yolo::demo
