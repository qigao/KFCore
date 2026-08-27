#pragma once

#include "kfcore/image_processor/error.hpp"
#include "kfcore/image_processor/types.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace kfcore::image
{

class ImageProcessor final
{
public:
    // Returns the packed Gray8 destination size after validating a borrowed Host image.
    static std::size_t packed_grayscale_bytes(const ImageView& image,
                                              std::size_t max_source_bytes);

    // Converts Host Gray8/BGR8/RGB8 to packed Gray8. Source and destination must not overlap;
    // the source is borrowed only for this call.
    static void stage_host_grayscale(const ImageView& image, MutableBufferView destination,
                                     std::size_t max_source_bytes);

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

class CudaImageProcessor final
{
public:
    ~CudaImageProcessor();

    CudaImageProcessor(const CudaImageProcessor&)            = delete;
    CudaImageProcessor& operator=(const CudaImageProcessor&) = delete;

    [[nodiscard]] static std::unique_ptr<CudaImageProcessor>
    create(const CudaImageProcessorOptions& options = {});

    // Calls on one processor are synchronous and non-reentrant.
    // Synchronously copies one image into owned packed CUDA storage. The returned image view is
    // borrowed until the next stage call or processor destruction; process_affine does not
    // invalidate it.
    [[nodiscard]] ImageView stage(const ImageView& source);

    // Synchronously produces an owned CUDA NCHW tensor. The returned view is borrowed until the
    // next process_affine call or processor destruction. The source is borrowed only for this call.
    [[nodiscard]] TensorView
    process_affine(const ImageView& source, std::int32_t destination_width,
                   std::int32_t destination_height, const AffineTransform& transform,
                   const PreprocessOptions& options      = {},
                   TensorElementType        element_type = TensorElementType::Float32);

private:
    struct Impl;
    explicit CudaImageProcessor(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::image
