#pragma once

#include "kfcore/image_processor/types.hpp"
#include "kfcore/sift/error.hpp"
#include "kfcore/sift/types.hpp"

namespace kfcore::sift
{

class SiftExtractor
{
public:
    virtual ~SiftExtractor();

    /**
     * Extract SIFT descriptors from a borrowed image view.
     *
     * The view and its data must remain valid only until this call returns. The returned
     * FeatureSet owns all feature and descriptor storage. Implementations report invalid input,
     * configured capacity violations, and backend failures with SiftError.
     */
    [[nodiscard]] virtual FeatureSet extract(const image::ImageView& image) = 0;
};

} // namespace kfcore::sift
