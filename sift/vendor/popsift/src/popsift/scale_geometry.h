/*
 * Copyright 2026 KFCore contributors
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "sift_conf.h"

#include <algorithm>
#include <cmath>

namespace popsift
{

struct ScaledImageGeometry
{
    int width;
    int height;
    int octaves;
};

inline ScaledImageGeometry scaleImageGeometry(const Config& config, int width, int height)
{
    const float scale_factor = std::pow(2.0f, config.getUpscaleFactor());
    const int octaves = config.octaves < 0
        ? std::max(static_cast<int>(std::floor(
                       std::log(static_cast<float>(std::min(width, height))) /
                       std::log(2.0f)) -
                   3.0f + scale_factor),
                   1)
        : config.octaves;

    return {
        static_cast<int>(std::ceil(width * scale_factor)),
        static_cast<int>(std::ceil(height * scale_factor)),
        octaves,
    };
}

} // namespace popsift
