#pragma once

#include "kfcore/hand_models/error.hpp"
#include "kfcore/hand_models/types.hpp"

#include <memory>

namespace kfcore::hand_models
{

class HandInferenceBackend
{
public:
    virtual ~HandInferenceBackend() = default;

    virtual HandFrame infer(const image::ImageView& image) = 0;
};

class HandPipeline final
{
public:
    ~HandPipeline();

    HandPipeline(const HandPipeline&)            = delete;
    HandPipeline& operator=(const HandPipeline&) = delete;

    [[nodiscard]] static std::unique_ptr<HandPipeline>
    create(std::unique_ptr<HandInferenceBackend> backend,
           const HandPipelineOptions& options = {});

    HandFrame process(const image::ImageView& image);
    HandFrame process(const image::FrameView& frame);
    void      reset();

private:
    struct Impl;
    explicit HandPipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::hand_models
