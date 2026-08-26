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
     * The constructor validates all options and acquires a process-local backend for the selected
     * CUDA device. Compatible instances share that backend. It throws SiftError when options are
     * invalid, an active backend has conflicting algorithm options, or initialization fails.
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
     * Concurrent calls may enqueue independent jobs on the shared backend and each call blocks
     * until its own result is ready. The caller must keep this object alive for the duration of
     * every call. CUDA-device ImageView inputs are rejected; this adapter never performs an
     * implicit device-to-host copy.
     */
    [[nodiscard]] FeatureSet extract(const image::ImageView& image) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::sift
