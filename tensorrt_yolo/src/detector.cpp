#include "kfcore/yolo/tensorrt.hpp"

#include "detector_helpers.hpp"
#include "letterbox.hpp"
#include "tensorrt_raii.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::yolo
{
namespace
{

    constexpr std::int32_t kInputChannels = 3;

    [[noreturn]] void throw_invalid(std::string message)
    {
        throw YoloError(YoloErrorCode::InvalidArgument, std::move(message));
    }

    [[noreturn]] void throw_tensorrt(std::string message)
    {
        throw YoloError(YoloErrorCode::TensorRtFailure, std::move(message));
    }

    [[noreturn]] void throw_resource(std::string message)
    {
        throw YoloError(YoloErrorCode::ResourceLimitExceeded, std::move(message));
    }

    class DetectorCallGuard final
    {
    public:
        explicit DetectorCallGuard(std::atomic_flag& in_use)
            : in_use_(in_use)
        {
            if (in_use_.test_and_set(std::memory_order_acquire))
            {
                throw_invalid("detection stage: concurrent calls on one detector are unsupported");
            }
        }

        ~DetectorCallGuard()
        {
            in_use_.clear(std::memory_order_release);
        }

        DetectorCallGuard(const DetectorCallGuard&) = delete;
        DetectorCallGuard& operator=(const DetectorCallGuard&) = delete;

    private:
        std::atomic_flag& in_use_;
    };

    std::size_t floating_element_size(TensorDataType type)
    {
        switch (type)
        {
        case TensorDataType::Float16:
            return sizeof(std::uint16_t);
        case TensorDataType::Float32:
            return sizeof(float);
        case TensorDataType::Int32:
            throw_tensorrt("detection setup stage: floating tensor type is invalid");
        }
        throw_tensorrt("detection setup stage: floating tensor type is unknown");
    }

    std::int32_t checked_dimension(std::int64_t dimension, const char* name)
    {
        if (dimension <= 0 || dimension > (std::numeric_limits<std::int32_t>::max)())
        {
            throw_tensorrt(std::string("detection setup stage: ") + name +
                           " cannot be represented as a TensorRT dimension");
        }
        return static_cast<std::int32_t>(dimension);
    }

    void validate_device_view(const ImageView& image, int device_id)
    {
        if (image.memory_kind != MemoryKind::CudaDevice)
        {
            return;
        }
        cudaPointerAttributes attributes {};
        detail::check_cuda(cudaPointerGetAttributes(&attributes, image.data),
                           "cudaPointerGetAttributes", "CUDA-device input validation");
        if (attributes.type != cudaMemoryTypeDevice)
        {
            throw_invalid("CUDA-device input validation stage: pointer is not device memory");
        }
        if (attributes.device != device_id)
        {
            throw_invalid(
                "CUDA-device input validation stage: pointer belongs to a different device");
        }
    }

    void set_tensor_address(nvinfer1::IExecutionContext& context, const std::string& name,
                            void* address)
    {
        if (!context.setTensorAddress(name.c_str(), address))
        {
            throw_tensorrt("tensor address stage: setTensorAddress failed for " + name);
        }
    }

} // namespace

TensorRtDetector::TensorRtDetector(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

TensorRtDetector::~TensorRtDetector() = default;

DetectionFrame TensorRtDetector::detect(const ImageView& image)
{
    std::vector<ImageView> images;
    images.reserve(1);
    images.push_back(image);
    std::vector<DetectionFrame> results = detect_batch(images);
    return std::move(results.front());
}

std::vector<DetectionFrame>
TensorRtDetector::detect_batch(const std::vector<ImageView>& images)
{
    if (!impl_ || !impl_->state || !impl_->context)
    {
        throw_invalid("detection stage: detector state is unavailable");
    }
    DetectorCallGuard call_guard(impl_->in_use);

    const ValidatedContract& contract = impl_->state->contract;
    const EngineOptions& engine_options = impl_->state->options;
    const std::int32_t input_width = impl_->input_size.width;
    const std::int32_t input_height = impl_->input_size.height;
    const std::size_t minimum_batch = static_cast<std::size_t>(
        checked_dimension(contract.min_batch, "minimum batch"));
    const std::size_t maximum_batch = static_cast<std::size_t>(
        checked_dimension(contract.max_batch, "maximum batch"));
    const std::size_t max_detections = static_cast<std::size_t>(
        checked_dimension(contract.max_detections, "maximum detections"));

    try
    {
        // ImageView remains borrowed through the final stream synchronization. Host bytes are
        // copied into detector-owned pinned storage before asynchronous work is submitted.
        const detail::BatchInputPlan input_plan = detail::prepare_batch(
            images, minimum_batch, maximum_batch, input_width, input_height,
            contract.input_type, engine_options.max_input_bytes);
        const detail::DetectionBufferLayout output_layout =
            detail::compute_detection_buffer_layout(images.size(), max_detections,
                                                    contract.output_type,
                                                    engine_options.max_output_bytes);
        if (input_plan.input_bytes > contract.images.max_bytes ||
            output_layout.num_dets_bytes > contract.num_dets.max_bytes ||
            output_layout.boxes_bytes > contract.boxes.max_bytes ||
            output_layout.scores_bytes > contract.scores.max_bytes ||
            output_layout.labels_bytes > contract.labels.max_bytes)
        {
            throw_tensorrt(
                "detection setup stage: dynamic tensor bytes exceed the validated contract");
        }

        detail::check_cuda(cudaSetDevice(engine_options.device_id), "cudaSetDevice",
                           "detection");
        for (const ImageView& image : images)
        {
            validate_device_view(image, engine_options.device_id);
        }

        impl_->source_device.reserve(input_plan.host_staging_bytes,
                                     engine_options.max_input_bytes);
        impl_->input_device.reserve(input_plan.input_bytes, engine_options.max_input_bytes);
        impl_->input_host.reserve(input_plan.host_staging_bytes,
                                  engine_options.max_input_bytes);
        impl_->num_dets_device.reserve(output_layout.num_dets_bytes,
                                       engine_options.max_output_bytes);
        impl_->boxes_device.reserve(output_layout.boxes_bytes, engine_options.max_output_bytes);
        impl_->scores_device.reserve(output_layout.scores_bytes, engine_options.max_output_bytes);
        impl_->labels_device.reserve(output_layout.labels_bytes, engine_options.max_output_bytes);
        impl_->num_dets_host.reserve(output_layout.num_dets_bytes,
                                     engine_options.max_output_bytes);
        impl_->boxes_host.reserve(output_layout.boxes_bytes, engine_options.max_output_bytes);
        impl_->scores_host.reserve(output_layout.scores_bytes, engine_options.max_output_bytes);
        impl_->labels_host.reserve(output_layout.labels_bytes, engine_options.max_output_bytes);

        auto* pinned_bytes = static_cast<std::byte*>(impl_->input_host.data());
        for (std::size_t image_index = 0; image_index < images.size(); ++image_index)
        {
            const ImageView& image = images[image_index];
            const detail::ImageInputPlan& plan = input_plan.images[image_index];
            if (image.memory_kind != MemoryKind::Host)
            {
                continue;
            }
            const std::size_t packed_row_bytes =
                plan.packed_bytes / static_cast<std::size_t>(image.height);
            const auto* source = static_cast<const std::byte*>(image.data);
            std::byte* destination = pinned_bytes + plan.staging_offset;
            for (std::int32_t row = 0; row < image.height; ++row)
            {
                std::memcpy(destination + static_cast<std::size_t>(row) * packed_row_bytes,
                            source + static_cast<std::size_t>(row) * image.row_stride,
                            packed_row_bytes);
            }
        }

        cudaStream_t stream = impl_->stream.get();
        bool stream_work_pending = false;
        try
        {
            if (input_plan.host_staging_bytes != 0)
            {
                detail::check_cuda(
                    cudaMemcpyAsync(impl_->source_device.data(), impl_->input_host.data(),
                                    input_plan.host_staging_bytes, cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync", "host input upload");
                stream_work_pending = true;
            }

            const std::size_t input_bytes_per_image = input_plan.input_bytes / images.size();
            auto* input_device_bytes = static_cast<std::byte*>(impl_->input_device.data());
            auto* staged_device_bytes = static_cast<std::byte*>(impl_->source_device.data());
            for (std::size_t image_index = 0; image_index < images.size(); ++image_index)
            {
                const ImageView& image = images[image_index];
                const detail::ImageInputPlan& plan = input_plan.images[image_index];
                const std::uint8_t* source = nullptr;
                std::size_t source_stride = image.row_stride;
                if (image.memory_kind == MemoryKind::Host)
                {
                    source = reinterpret_cast<const std::uint8_t*>(
                        staged_device_bytes + plan.staging_offset);
                    source_stride = plan.packed_bytes / static_cast<std::size_t>(image.height);
                }
                else
                {
                    source = static_cast<const std::uint8_t*>(image.data);
                }
                void* destination = input_device_bytes + image_index * input_bytes_per_image;
                detail::launch_letterbox(source, source_stride, image.pixel_format, destination,
                                         input_width, input_height,
                                         contract.input_type, plan.transform,
                                         impl_->options.mean, impl_->options.stddev,
                                         impl_->options.border_value, stream);
                stream_work_pending = true;
            }
            nvinfer1::Dims input_shape {};
            input_shape.nbDims = 4;
            input_shape.d[0] = static_cast<std::int32_t>(images.size());
            input_shape.d[1] = kInputChannels;
            input_shape.d[2] = input_height;
            input_shape.d[3] = input_width;
            const TensorNames& names = engine_options.tensor_names;
            if (!impl_->context->setInputShape(names.images.c_str(), input_shape))
            {
                throw_tensorrt("input shape stage: setInputShape rejected the batch shape");
            }

            set_tensor_address(*impl_->context, names.images, impl_->input_device.data());
            set_tensor_address(*impl_->context, names.num_dets, impl_->num_dets_device.data());
            set_tensor_address(*impl_->context, names.boxes, impl_->boxes_device.data());
            set_tensor_address(*impl_->context, names.scores, impl_->scores_device.data());
            set_tensor_address(*impl_->context, names.labels, impl_->labels_device.data());
            if (!impl_->context->enqueueV3(stream))
            {
                throw_tensorrt("enqueue stage: enqueueV3 failed");
            }
            stream_work_pending = true;

            detail::check_cuda(
                cudaMemcpyAsync(impl_->num_dets_host.data(), impl_->num_dets_device.data(),
                                output_layout.num_dets_bytes, cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync", "num_dets output download");
            detail::check_cuda(
                cudaMemcpyAsync(impl_->boxes_host.data(), impl_->boxes_device.data(),
                                output_layout.boxes_bytes, cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync", "boxes output download");
            detail::check_cuda(
                cudaMemcpyAsync(impl_->scores_host.data(), impl_->scores_device.data(),
                                output_layout.scores_bytes, cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync", "scores output download");
            detail::check_cuda(
                cudaMemcpyAsync(impl_->labels_host.data(), impl_->labels_device.data(),
                                output_layout.labels_bytes, cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync", "labels output download");
            detail::check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                               "detection completion");
            stream_work_pending = false;

            const std::size_t float_bytes = floating_element_size(contract.output_type);
            const detail::EfficientNmsOutputView output_view {
                static_cast<const std::int32_t*>(impl_->num_dets_host.data()),
                output_layout.num_dets_bytes / sizeof(std::int32_t),
                impl_->boxes_host.data(),
                output_layout.boxes_bytes / float_bytes,
                impl_->scores_host.data(),
                output_layout.scores_bytes / float_bytes,
                static_cast<const std::int32_t*>(impl_->labels_host.data()),
                output_layout.labels_bytes / sizeof(std::int32_t),
                max_detections,
                contract.output_type,
            };
            std::vector<detail::LetterboxTransform> transforms;
            transforms.reserve(input_plan.images.size());
            for (const detail::ImageInputPlan& plan : input_plan.images)
            {
                transforms.push_back(plan.transform);
            }
            return detail::decode_efficient_nms(images, transforms, output_view);
        }
        catch (...)
        {
            if (stream_work_pending)
            {
                (void)cudaStreamSynchronize(stream);
            }
            throw;
        }
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("detection stage: allocation failed");
    }
    catch (const std::length_error&)
    {
        throw_resource("detection stage: container capacity exceeded");
    }
}

} // namespace kfcore::yolo
