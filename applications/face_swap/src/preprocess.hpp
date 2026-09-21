#pragma once

#include "kfcore/image_processor/types.hpp"

#include <cstddef>
#include <vector>

namespace kfcore::face_applications::detail
{

[[nodiscard]] std::vector<float> preprocess_face68(
    const kfcore::image::BgrImage& image, std::size_t max_tensor_bytes);
[[nodiscard]] std::vector<float> preprocess_arcface(
    const kfcore::image::BgrImage& image, std::size_t max_tensor_bytes);
[[nodiscard]] std::vector<float> preprocess_inswapper(
    const kfcore::image::BgrImage& image, std::size_t max_tensor_bytes);
[[nodiscard]] std::vector<float> preprocess_gfpgan(
    const kfcore::image::BgrImage& image, std::size_t max_tensor_bytes);
[[nodiscard]] std::vector<float> preprocess_age_gender(
    const kfcore::image::BgrImage& image, std::size_t max_image_bytes,
    std::size_t max_tensor_bytes);

} // namespace kfcore::face_applications::detail

