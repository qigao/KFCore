#pragma once

#include "kfcore/image_processor/types.hpp"

#include <utility>

namespace kfcore::hand_models::detail
{

template <typename StageOperation>
image::ImageView stage_host_or_borrow_cuda(
    const image::ImageView& source, StageOperation&& stage_operation)
{
    if (source.memory_kind == image::MemoryKind::CudaDevice)
    {
        return source;
    }
    return std::forward<StageOperation>(stage_operation)(source);
}

} // namespace kfcore::hand_models::detail
