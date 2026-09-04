#pragma once

#include "yolo_domain_capture.hpp"
#include "kfcore/image_processor/types.hpp"
#include "kfcore/yolo/types.hpp"

#include <cstddef>
#include <filesystem>

namespace kfcore::yolo::demo
{

// Returns independent packed BGR storage; no mailbox view is retained.
[[nodiscard]] kfcore::image::BgrImage to_bgr(
    const CapturedFrame& frame, std::size_t max_image_bytes);
[[nodiscard]] kfcore::image::BgrImage load_bgr(
    const std::filesystem::path& path, std::size_t max_file_bytes,
    std::size_t max_image_bytes);
void save_bgr(const std::filesystem::path& path,
              const kfcore::image::BgrImage& image,
              std::size_t max_image_bytes);
[[nodiscard]] ImageView capture_image_view(const CapturedFrame& frame);
[[nodiscard]] ImageView bgr_image_view(const kfcore::image::BgrImage& image);
void mirror_bgr_horizontal(kfcore::image::BgrImage& image);

} // namespace kfcore::yolo::demo
