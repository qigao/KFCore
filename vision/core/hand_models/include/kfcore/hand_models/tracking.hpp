#pragma once

#include "kfcore/hand_models/error.hpp"
#include "kfcore/hand_models/types.hpp"

#include <memory>

namespace kfcore::hand_models
{

class HandTracker final
{
public:
    ~HandTracker();

    HandTracker(const HandTracker&) = delete;
    HandTracker& operator=(const HandTracker&) = delete;

    [[nodiscard]] static std::unique_ptr<HandTracker>
    create(const HandTrackingOptions& options = {});

    [[nodiscard]] HandFrame update(HandFrame frame);
    [[nodiscard]] HandFrame update(const image::ImageView& source, HandFrame frame);
    void reset();

private:
    struct Impl;
    explicit HandTracker(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::hand_models
