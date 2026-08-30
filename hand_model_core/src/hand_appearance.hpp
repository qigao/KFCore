#pragma once

#include "kfcore/hand_models/types.hpp"

#include <optional>

namespace kfcore::hand_models::detail
{

void validate_hand_appearance_source(const image::ImageView& image);

[[nodiscard]] std::optional<HandAppearanceDescriptor>
make_hand_appearance_descriptor(
    const image::ImageView& image,
    const std::array<HandLandmark, kHandLandmarkCount>& landmarks,
    const HandAppearanceOptions& options);

} // namespace kfcore::hand_models::detail
