#pragma once

#include "kfcore/sift/popsift_options.hpp"
#include "kfcore/sift/sift_extractor.hpp"

#include <memory>

namespace kfcore::sift
{

class PopSiftExtractor final : public SiftExtractor
{
public:
    /**
     * Create a synchronous PopSift adapter.
     *
     * The constructor validates all options and initializes the selected CUDA device through
     * PopSift. It throws SiftError when options are invalid or the backend cannot initialize.
     */
    explicit PopSiftExtractor(PopSiftOptions options = {});
    ~PopSiftExtractor() override;

    PopSiftExtractor(const PopSiftExtractor&)            = delete;
    PopSiftExtractor& operator=(const PopSiftExtractor&) = delete;
    PopSiftExtractor(PopSiftExtractor&&)                 = delete;
    PopSiftExtractor& operator=(PopSiftExtractor&&)      = delete;

    /**
     * Extract one Host Gray8/BGR8/RGB8 image.
     *
     * Calls on the same instance are serialized and block until PopSift returns. CUDA-device
     * ImageView inputs are rejected; this adapter never performs an implicit device-to-host copy.
     */
    [[nodiscard]] FeatureSet extract(const image::ImageView& image) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::sift
