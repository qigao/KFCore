#pragma once

#include "kfcore/image_processor/error.hpp"
#include "kfcore/image_processor/types.hpp"

#include <cstddef>
#include <vector>

namespace kfcore::image
{

class ImageProcessor final
{
public:
    static LetterboxTransform letterbox_transform(std::int32_t source_width,
                                                  std::int32_t source_height,
                                                  std::int32_t destination_width,
                                                  std::int32_t destination_height);

    static BatchPlan plan(const std::vector<ImageView>& images, const TensorView& destination,
                          std::size_t max_source_bytes, std::size_t max_tensor_bytes);

    static void stage_host_inputs(const std::vector<ImageView>& images, const BatchPlan& plan,
                                  MutableBufferView pinned_host_workspace);

    // All borrowed CUDA memory and pinned_host_workspace must remain valid and unmodified until
    // work previously submitted to stream completes. This function retains no pointers.
    static void enqueue(const std::vector<ImageView>& images, const TensorView& destination,
                        const BatchPlan& plan, MutableBufferView pinned_host_workspace,
                        MutableBufferView device_workspace, const PreprocessOptions& options,
                        void* stream);
};

} // namespace kfcore::image
