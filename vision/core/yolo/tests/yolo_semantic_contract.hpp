#pragma once

#include "kfcore/yolo/types.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace kfcore::yolo::test
{

inline void check_semantic_contract(const DetectionFrame& frame,
                                    std::int32_t expected_width,
                                    std::int32_t expected_height,
                                    std::size_t class_count = 0U)
{
    check(frame.image_width == expected_width);
    check(frame.image_height == expected_height);

    for (const Detection& detection : frame.detections)
    {
        check_true(std::isfinite(detection.score));
        check_true(detection.score > 0.0F && detection.score <= 1.0F);
        check(detection.class_id >= 0);
        if (class_count != 0U)
        {
            check(static_cast<std::size_t>(detection.class_id) < class_count);
        }

        check_true(std::isfinite(detection.box.left));
        check_true(std::isfinite(detection.box.top));
        check_true(std::isfinite(detection.box.right));
        check_true(std::isfinite(detection.box.bottom));
        check_true(detection.box.left >= 0.0F);
        check_true(detection.box.top >= 0.0F);
        check_true(detection.box.right <= static_cast<float>(expected_width));
        check_true(detection.box.bottom <= static_cast<float>(expected_height));
        check_true(detection.box.left <= detection.box.right);
        check_true(detection.box.top <= detection.box.bottom);
    }
}

} // namespace kfcore::yolo::test
