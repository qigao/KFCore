#pragma once

#include "kfcore/vision_models/types.hpp"

#include <optional>

namespace kfcore::vision_models::detail
{

void validate_hand_appearance_source(const image::ImageView& image);

[[nodiscard]] std::optional<HandAppearanceDescriptor>
make_hand_appearance_descriptor(
    const image::ImageView& image,
    const std::array<HandLandmark, kHandLandmarkCount>& landmarks,
    const HandAppearanceOptions& options);

} // namespace kfcore::vision_models::detail
