#pragma once

#include "kfcore/yolo/error.hpp"
#include "kfcore/yolo/types.hpp"

namespace cv {
class Mat;
}

namespace kfcore::yolo {

struct DrawOptions {
    bool draw_unconfirmed = false;
    int line_thickness = 2;
    double font_scale = 0.5;
};

// Returns a non-owning Host view. It expires if image is released, reallocated, or mutated in a
// way that changes its backing storage. The caller must keep image alive and storage-stable while
// the ImageView is consumed.
ImageView image_view(const cv::Mat& image, PixelFormat pixel_format = PixelFormat::Bgr8);

// Renders a validated TrackFrame without reading or changing tracker/session state.
void draw_tracks(cv::Mat& image, const TrackFrame& tracks, const DrawOptions& options = {});

}  // namespace kfcore::yolo
