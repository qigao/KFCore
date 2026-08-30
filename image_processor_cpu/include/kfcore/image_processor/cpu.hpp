#pragma once

#include "kfcore/image_processor/error.hpp"
#include "kfcore/image_processor/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kfcore::image
{

class CpuImageProcessor final
{
public:
    [[nodiscard]] static BgrImage copy_bgr(const ImageView& source,
                                           std::size_t max_image_bytes);

    [[nodiscard]] static BgrImage resize_bgr(const BgrImage& source,
                                             std::int32_t destination_width,
                                             std::int32_t destination_height,
                                             std::size_t max_image_bytes);

    // The affine matrix maps integer destination coordinates to source coordinates.
    [[nodiscard]] static BgrImage warp_affine_bgr(
        const BgrImage& source, std::int32_t destination_width,
        std::int32_t destination_height, const AffineTransform& destination_to_source,
        float border_value, std::size_t max_image_bytes);

    [[nodiscard]] static std::vector<float>
    to_nchw(const BgrImage& source, const PreprocessOptions& options,
            std::size_t max_tensor_bytes);

    [[nodiscard]] static std::vector<float>
    letterbox_nchw(const ImageView& source, std::int32_t destination_width,
                   std::int32_t destination_height, const PreprocessOptions& options,
                   std::size_t max_source_bytes, std::size_t max_tensor_bytes,
                   LetterboxTransform* transform);
};

} // namespace kfcore::image
