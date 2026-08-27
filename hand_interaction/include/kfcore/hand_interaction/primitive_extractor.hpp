#pragma once

#include "kfcore/hand_interaction/types.hpp"
#include "kfcore/vision_models/types.hpp"

#include <memory>

namespace kfcore::hand_interaction
{

class HandPrimitiveExtractor final
{
public:
    explicit HandPrimitiveExtractor(const HandPrimitiveOptions& options = {});
    ~HandPrimitiveExtractor();

    HandPrimitiveExtractor(const HandPrimitiveExtractor&)            = delete;
    HandPrimitiveExtractor& operator=(const HandPrimitiveExtractor&) = delete;
    HandPrimitiveExtractor(HandPrimitiveExtractor&&) noexcept;
    HandPrimitiveExtractor& operator=(HandPrimitiveExtractor&&) noexcept;

    [[nodiscard]] PrimitiveFrame process(
        const vision_models::HandFrame& frame,
        const GestureFrameContext& context);
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::hand_interaction
