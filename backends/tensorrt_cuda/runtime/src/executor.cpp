#include "executor.hpp"

#include "cuda_device.hpp"
#include "shared_lifetime.hpp"
#include "tensor_validation.hpp"

#include "kfcore/tensorrt/error.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::tensorrt
{
namespace
{

    [[noreturn]] void throw_invalid(std::string message)
    {
        throw TensorRtError(TensorRtErrorCode::InvalidTensorView, std::move(message));
    }

    [[noreturn]] void throw_contract(std::string message)
    {
        throw TensorRtError(TensorRtErrorCode::EngineContractMismatch, std::move(message));
    }

    [[noreturn]] void throw_tensorrt(std::string message)
    {
        throw TensorRtError(TensorRtErrorCode::TensorRtFailure, std::move(message));
    }

    [[noreturn]] void throw_resource(std::string message)
    {
        throw TensorRtError(TensorRtErrorCode::ResourceLimitExceeded, std::move(message));
    }

    std::vector<TensorDescriptor> descriptors_with_mode(
        const std::vector<TensorDescriptor>& tensors, TensorIoMode mode)
    {
        std::vector<TensorDescriptor> result;
        result.reserve(tensors.size());
        for (const TensorDescriptor& tensor : tensors)
        {
            if (tensor.mode == mode)
            {
                result.push_back(tensor);
            }
        }
        return result;
    }

    const TensorView& find_input(const std::vector<TensorView>& inputs,
                                 const std::string& name)
    {
        for (const TensorView& input : inputs)
        {
            if (input.name == name)
            {
                return input;
            }
        }
        throw_invalid("input binding stage: validated input tensor is missing: " + name);
    }

    const MutableTensorView& find_output(const std::vector<MutableTensorView>& outputs,
                                         const std::string& name)
    {
        for (const MutableTensorView& output : outputs)
        {
            if (output.name == name)
            {
                return output;
            }
        }
        throw_invalid("output binding stage: validated output tensor is missing: " + name);
    }

    const DynamicOutputRequest& find_dynamic_output(
        const std::vector<DynamicOutputRequest>& outputs, const std::string& name)
    {
        for (const DynamicOutputRequest& output : outputs)
        {
            if (output.name == name)
            {
                return output;
            }
        }
        throw_invalid("dynamic output binding stage: validated output request is missing: " +
                      name);
    }

    nvinfer1::Dims dims_from_shape(const TensorShape& shape, const std::string& tensor_name)
    {
        if (shape.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS))
        {
            throw_invalid("input shape stage: tensor rank exceeds TensorRT capacity: " +
                          tensor_name);
        }
        nvinfer1::Dims dims {};
        dims.nbDims = static_cast<std::int32_t>(shape.size());
        for (std::size_t index = 0; index < shape.size(); ++index)
        {
            dims.d[index] = shape[index];
        }
        return dims;
    }

    TensorShape resolved_output_shape(nvinfer1::IExecutionContext& context,
                                      const std::string& tensor_name)
    {
        const nvinfer1::Dims dims = context.getTensorShape(tensor_name.c_str());
        if (dims.nbDims < 0 || dims.nbDims > nvinfer1::Dims::MAX_DIMS)
        {
            throw_contract("output shape stage: output rank is unresolved for " + tensor_name);
        }
        TensorShape shape;
        shape.reserve(static_cast<std::size_t>(dims.nbDims));
        for (std::int32_t index = 0; index < dims.nbDims; ++index)
        {
            if (dims.d[index] <= 0)
            {
                throw_contract("output shape stage: data-dependent or unresolved output is "
                               "unsupported: " +
                               tensor_name);
            }
            shape.push_back(dims.d[index]);
        }
        return shape;
    }

    TensorShape resolved_data_dependent_output_shape(const nvinfer1::Dims& dims,
                                                      const std::string& tensor_name)
    {
        if (dims.nbDims < 0 || dims.nbDims > nvinfer1::Dims::MAX_DIMS)
        {
            throw_contract("dynamic output shape stage: output rank is unresolved for " +
                           tensor_name);
        }
        TensorShape shape;
        shape.reserve(static_cast<std::size_t>(dims.nbDims));
        for (std::int32_t index = 0; index < dims.nbDims; ++index)
        {
            if (dims.d[index] < 0)
            {
                throw_contract("dynamic output shape stage: output dimension remains unresolved: " +
                               tensor_name);
            }
            shape.push_back(dims.d[index]);
        }
        return shape;
    }

    void validate_memory_kind(MemoryKind memory_kind, const std::string& tensor_name)
    {
        if (memory_kind != MemoryKind::Host && memory_kind != MemoryKind::CudaDevice)
        {
            throw_invalid("tensor view validation stage: invalid memory kind for " + tensor_name);
        }
    }

    struct AddressRange
    {
        std::uintptr_t begin;
        std::uintptr_t end;
        MemoryKind     memory_kind;
        const std::string* name;
    };

    AddressRange address_range(const void* data, std::size_t bytes, MemoryKind memory_kind,
                               const std::string& name)
    {
        const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
        if (bytes > (std::numeric_limits<std::uintptr_t>::max)() - begin)
        {
            throw_invalid("tensor view validation stage: address range overflows for " + name);
        }
        return { begin, begin + bytes, memory_kind, &name };
    }

    bool overlaps(const AddressRange& left, const AddressRange& right) noexcept
    {
        return left.memory_kind == right.memory_kind && left.begin < right.end &&
               right.begin < left.end;
    }

    void reject_view_aliases(const std::vector<TensorView>& inputs,
                             const std::vector<MutableTensorView>& outputs)
    {
        std::vector<AddressRange> ranges;
        ranges.reserve(inputs.size() + outputs.size());
        for (const TensorView& input : inputs)
        {
            validate_memory_kind(input.memory_kind, input.name);
            ranges.push_back(address_range(
                input.data,
                detail::checked_shape_byte_size(input.shape, input.data_type,
                                                "tensor view alias validation"),
                input.memory_kind, input.name));
        }
        for (const MutableTensorView& output : outputs)
        {
            validate_memory_kind(output.memory_kind, output.name);
            ranges.push_back(address_range(
                output.data,
                detail::checked_shape_byte_size(output.shape, output.data_type,
                                                "tensor view alias validation"),
                output.memory_kind, output.name));
        }

        for (std::size_t index = 0; index < ranges.size(); ++index)
        {
            for (std::size_t previous = 0; previous < index; ++previous)
            {
                if (overlaps(ranges[previous], ranges[index]))
                {
                    throw_invalid("tensor view alias validation stage: tensor storage overlaps: " +
                                  *ranges[previous].name + " and " + *ranges[index].name);
                }
            }
        }
    }

    void set_input_shapes_and_validate_outputs(
        nvinfer1::IExecutionContext& context,
        const std::vector<TensorDescriptor>& input_descriptors,
        const std::vector<TensorDescriptor>& output_descriptors,
        const std::vector<TensorView>& inputs,
        const std::vector<MutableTensorView>& outputs, std::size_t max_output_bytes)
    {
        for (const TensorDescriptor& descriptor : input_descriptors)
        {
            const TensorView& input = find_input(inputs, descriptor.name);
            if (!context.setInputShape(descriptor.name.c_str(),
                                       dims_from_shape(input.shape, descriptor.name)))
            {
                throw_tensorrt("input shape stage: setInputShape rejected " + descriptor.name);
            }
        }
        if (context.inferShapes(0, nullptr) != 0)
        {
            throw_tensorrt("output shape stage: TensorRT could not infer all tensor shapes");
        }
        std::size_t aggregate_output_bytes = 0;
        for (const TensorDescriptor& descriptor : output_descriptors)
        {
            const MutableTensorView& output = find_output(outputs, descriptor.name);
            const TensorShape shape = resolved_output_shape(context, descriptor.name);
            if (shape != output.shape)
            {
                throw_invalid("output shape stage: caller shape does not match resolved shape for " +
                              descriptor.name);
            }
            const std::size_t bytes = detail::checked_shape_byte_size(
                shape, descriptor.data_type, "output shape validation");
            if (output.byte_size < bytes)
            {
                throw_invalid("output shape stage: caller capacity is smaller than resolved bytes for " +
                              descriptor.name);
            }
            if (bytes > max_output_bytes - aggregate_output_bytes)
            {
                throw_resource(
                    "output shape stage: aggregate resolved bytes exceed configured limit");
            }
            aggregate_output_bytes += bytes;
        }
    }

    void set_input_shapes(nvinfer1::IExecutionContext& context,
                          const std::vector<TensorDescriptor>& input_descriptors,
                          const std::vector<TensorView>& inputs)
    {
        for (const TensorDescriptor& descriptor : input_descriptors)
        {
            const TensorView& input = find_input(inputs, descriptor.name);
            if (!context.setInputShape(descriptor.name.c_str(),
                                       dims_from_shape(input.shape, descriptor.name)))
            {
                throw_tensorrt("input shape stage: setInputShape rejected " + descriptor.name);
            }
        }
        if (context.inferShapes(0, nullptr) != 0)
        {
            throw_tensorrt("output shape stage: TensorRT could not infer all tensor shapes");
        }
    }

    void validate_cuda_views(const std::vector<TensorView>& inputs,
                             const std::vector<MutableTensorView>& outputs, int device_id)
    {
        for (const TensorView& input : inputs)
        {
            if (input.memory_kind == MemoryKind::CudaDevice)
            {
                detail::validate_cuda_device_pointer(input.data, device_id, input.name);
            }
        }
        for (const MutableTensorView& output : outputs)
        {
            if (output.memory_kind == MemoryKind::CudaDevice)
            {
                detail::validate_cuda_device_pointer(output.data, device_id, output.name);
            }
        }
    }

    void bind_input(nvinfer1::IExecutionContext& context, cudaStream_t stream,
                    detail::ExecutorStagingBuffers& staging,
                    const TensorDescriptor& descriptor, const TensorView& input,
                    const EngineOptions& options, bool& stream_work_pending)
    {
        const std::size_t bytes =
            detail::checked_shape_byte_size(input.shape, input.data_type, "input binding");
        const void* address = input.data;
        if (input.memory_kind == MemoryKind::Host)
        {
            staging.host.reserve(bytes, options.max_input_bytes);
            staging.device.reserve(bytes, options.max_input_bytes);
            std::memcpy(staging.host.data(), input.data, bytes);
            stream_work_pending = true;
            detail::check_cuda(cudaMemcpyAsync(staging.device.data(), staging.host.data(), bytes,
                                               cudaMemcpyHostToDevice, stream),
                               "cudaMemcpyAsync", "input upload");
            address = staging.device.data();
        }
        if (!context.setInputTensorAddress(descriptor.name.c_str(), address))
        {
            throw_tensorrt("tensor address stage: input address rejected for " +
                           descriptor.name);
        }
    }

    void bind_output(nvinfer1::IExecutionContext& context,
                     detail::ExecutorStagingBuffers& staging,
                     const TensorDescriptor& descriptor, const MutableTensorView& output,
                     const EngineOptions& options)
    {
        const std::size_t bytes =
            detail::checked_shape_byte_size(output.shape, output.data_type, "output binding");
        void* address = output.data;
        if (output.memory_kind == MemoryKind::Host)
        {
            staging.host.reserve(bytes, options.max_output_bytes);
            staging.device.reserve(bytes, options.max_output_bytes);
            address = staging.device.data();
        }
        if (!context.setOutputTensorAddress(descriptor.name.c_str(), address))
        {
            throw_tensorrt("tensor address stage: output address rejected for " +
                           descriptor.name);
        }
    }

    void bind_tensors(nvinfer1::IExecutionContext& context, cudaStream_t stream,
                      const std::vector<TensorDescriptor>& tensors,
                      std::vector<detail::ExecutorStagingBuffers>& staging,
                      const std::vector<TensorView>& inputs,
                      const std::vector<MutableTensorView>& outputs,
                      const EngineOptions& options, bool& stream_work_pending)
    {
        for (std::size_t index = 0; index < tensors.size(); ++index)
        {
            const TensorDescriptor& descriptor = tensors[index];
            if (descriptor.mode == TensorIoMode::Input)
            {
                bind_input(context, stream, staging[index], descriptor,
                           find_input(inputs, descriptor.name), options, stream_work_pending);
            }
            else
            {
                bind_output(context, staging[index], descriptor,
                            find_output(outputs, descriptor.name), options);
            }
        }
    }

    void download_host_outputs(cudaStream_t stream,
                               const std::vector<TensorDescriptor>& tensors,
                               std::vector<detail::ExecutorStagingBuffers>& staging,
                               const std::vector<MutableTensorView>& outputs)
    {
        for (std::size_t index = 0; index < tensors.size(); ++index)
        {
            const TensorDescriptor& descriptor = tensors[index];
            if (descriptor.mode != TensorIoMode::Output)
            {
                continue;
            }
            const MutableTensorView& output = find_output(outputs, descriptor.name);
            if (output.memory_kind != MemoryKind::Host)
            {
                continue;
            }
            const std::size_t bytes = detail::checked_shape_byte_size(
                output.shape, output.data_type, "output download");
            detail::check_cuda(cudaMemcpyAsync(staging[index].host.data(),
                                               staging[index].device.data(), bytes,
                                               cudaMemcpyDeviceToHost, stream),
                               "cudaMemcpyAsync", "output download");
        }
    }

    void copy_host_outputs(const std::vector<TensorDescriptor>& tensors,
                           const std::vector<detail::ExecutorStagingBuffers>& staging,
                           const std::vector<MutableTensorView>& outputs)
    {
        for (std::size_t index = 0; index < tensors.size(); ++index)
        {
            const TensorDescriptor& descriptor = tensors[index];
            if (descriptor.mode != TensorIoMode::Output)
            {
                continue;
            }
            const MutableTensorView& output = find_output(outputs, descriptor.name);
            if (output.memory_kind == MemoryKind::Host)
            {
                const std::size_t bytes = detail::checked_shape_byte_size(
                    output.shape, output.data_type, "output copy");
                std::memcpy(output.data, staging[index].host.data(), bytes);
            }
        }
    }

    class BoundedOutputAllocator final : public nvinfer1::IOutputAllocator
    {
    public:
        BoundedOutputAllocator(void* storage, std::size_t capacity) noexcept
            : storage_(storage)
            , capacity_(capacity)
        {
        }

        void* reallocateOutputAsync(const char*, void*, std::uint64_t size,
                                    std::uint64_t, cudaStream_t) noexcept override
        {
            requested_bytes_ = size;
            if (size > capacity_)
            {
                capacity_exceeded_ = true;
                return nullptr;
            }
            return storage_;
        }

        void notifyShape(const char*, const nvinfer1::Dims& dims) noexcept override
        {
            shape_          = dims;
            shape_notified_ = true;
        }

        bool capacity_exceeded() const noexcept
        {
            return capacity_exceeded_;
        }

        std::uint64_t requested_bytes() const noexcept
        {
            return requested_bytes_;
        }

        bool shape_notified() const noexcept
        {
            return shape_notified_;
        }

        const nvinfer1::Dims& shape() const noexcept
        {
            return shape_;
        }

    private:
        void*         storage_          = nullptr;
        std::size_t   capacity_         = 0;
        std::uint64_t requested_bytes_  = 0;
        bool          capacity_exceeded_ = false;
        nvinfer1::Dims shape_ {};
        bool           shape_notified_ = false;
    };

    void bind_dynamic_tensors(
        nvinfer1::IExecutionContext& context, cudaStream_t stream,
        const std::vector<TensorDescriptor>& tensors,
        std::vector<detail::ExecutorStagingBuffers>& staging,
        const std::vector<TensorView>& inputs,
        const std::vector<DynamicOutputRequest>& outputs,
        const EngineOptions& options, bool& stream_work_pending,
        std::vector<std::unique_ptr<BoundedOutputAllocator>>& allocators)
    {
        allocators.clear();
        allocators.reserve(outputs.size());
        std::size_t aggregate_upper_bound = 0;
        for (std::size_t index = 0; index < tensors.size(); ++index)
        {
            const TensorDescriptor& descriptor = tensors[index];
            if (descriptor.mode == TensorIoMode::Input)
            {
                bind_input(context, stream, staging[index], descriptor,
                           find_input(inputs, descriptor.name), options, stream_work_pending);
                continue;
            }

            const DynamicOutputRequest& request = find_dynamic_output(outputs, descriptor.name);
            const std::int64_t engine_upper_bound =
                context.getMaxOutputSize(descriptor.name.c_str());
            if (engine_upper_bound <= 0)
            {
                throw_tensorrt("dynamic output capacity stage: TensorRT did not provide an upper "
                               "bound for " + descriptor.name);
            }
            if (static_cast<std::uintmax_t>(engine_upper_bound) >
                static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)()))
            {
                throw_resource("dynamic output capacity stage: engine upper bound cannot be "
                               "represented for " + descriptor.name);
            }
            const std::size_t bytes = static_cast<std::size_t>(engine_upper_bound);
            if (bytes > request.max_byte_size)
            {
                throw_resource("dynamic output capacity stage: engine upper bound exceeds caller "
                               "limit for " + descriptor.name);
            }
            if (bytes > options.max_output_bytes - aggregate_upper_bound)
            {
                throw_resource("dynamic output capacity stage: aggregate engine upper bounds "
                               "exceed configured limit");
            }
            aggregate_upper_bound += bytes;
            staging[index].device.reserve(bytes, request.max_byte_size);
            staging[index].host.reserve(bytes, request.max_byte_size);
            auto allocator = std::make_unique<BoundedOutputAllocator>(staging[index].device.data(),
                                                                      bytes);
            if (!context.setOutputTensorAddress(descriptor.name.c_str(),
                                                staging[index].device.data()) ||
                !context.setOutputAllocator(descriptor.name.c_str(), allocator.get()))
            {
                throw_tensorrt("dynamic output binding stage: TensorRT rejected binding for " +
                               descriptor.name);
            }
            allocators.push_back(std::move(allocator));
        }
    }

    bool clear_dynamic_tensor_bindings(nvinfer1::IExecutionContext& context,
                                       const std::vector<TensorDescriptor>& tensors) noexcept
    {
        bool cleared = true;
        for (const TensorDescriptor& tensor : tensors)
        {
            if (tensor.mode == TensorIoMode::Output &&
                !context.setOutputAllocator(tensor.name.c_str(), nullptr))
            {
                cleared = false;
            }
            if (!context.setTensorAddress(tensor.name.c_str(), nullptr))
            {
                cleared = false;
            }
        }
        return cleared;
    }

    bool clear_tensor_addresses(nvinfer1::IExecutionContext& context,
                                const std::vector<TensorDescriptor>& tensors) noexcept
    {
        bool cleared = true;
        for (const TensorDescriptor& tensor : tensors)
        {
            if (!context.setTensorAddress(tensor.name.c_str(), nullptr))
            {
                cleared = false;
            }
        }
        return cleared;
    }

} // namespace

Executor::Impl::Impl(std::shared_ptr<const Engine::Impl> engine_state,
                     detail::TensorRtOwner<nvinfer1::IExecutionContext> execution_context)
    : engine_owner(detail::make_shared_lifetime_anchor(std::move(engine_state)))
    , engine(engine_owner->get())
    , context(std::move(execution_context))
    , stream(std::make_unique<detail::CudaStream>())
    , staging(engine->tensors.size())
{
}

Executor::Impl::~Impl() noexcept
{
    if (!engine)
    {
        return;
    }

    const int device_id = engine->options.device_id;
    const detail::DeviceCleanupActions actions {
        this,
        [](void* opaque) noexcept
        {
            auto& self = *static_cast<Impl*>(opaque);
            if (self.stream)
            {
                (void)cudaStreamSynchronize(self.stream->get());
            }
            if (self.context)
            {
                (void)clear_tensor_addresses(*self.context, self.engine->tensors);
            }
            self.context.reset();
            self.staging.clear();
            self.stream.reset();
            self.engine = nullptr;
            self.engine_owner.reset();
        },
        [](void* opaque) noexcept
        {
            auto& self = *static_cast<Impl*>(opaque);
            // Device selection failed, so calling any CUDA/TensorRT destructor would be unsafe.
            // Relinquish every owner; this intentionally leaks only on this unrecoverable path.
            (void)self.context.release();
            for (detail::ExecutorStagingBuffers& buffers : self.staging)
            {
                buffers.device.abandon();
                buffers.host.abandon();
            }
            self.staging.clear();
            if (self.stream)
            {
                self.stream->abandon();
                self.stream.reset();
            }
            self.engine = nullptr;
            (void)detail::abandon_shared_lifetime_anchor(self.engine_owner);
        },
    };
    detail::cleanup_on_cuda_device_or_abandon(device_id, actions);
}

Executor::Executor(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

Executor::~Executor() = default;

std::unique_ptr<Executor> Engine::create_executor() const
{
    if (!impl_ || !*impl_ || !(*impl_)->engine)
    {
        throw TensorRtError(TensorRtErrorCode::InvalidArgument,
                            "executor creation stage: engine state is unavailable");
    }

    detail::CudaDeviceScope device_scope((*impl_)->options.device_id);
    detail::TensorRtOwner<nvinfer1::IExecutionContext> context(
        (*impl_)->engine->createExecutionContext());
    if (!context)
    {
        throw_tensorrt("executor creation stage: createExecutionContext returned null");
    }

    try
    {
        auto executor_impl = std::make_unique<Executor::Impl>(*impl_, std::move(context));
        std::unique_ptr<Executor> result(new Executor(std::move(executor_impl)));
        device_scope.restore();
        return result;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("executor creation stage: allocation failed");
    }
    catch (const std::length_error&)
    {
        throw_resource("executor creation stage: container capacity exceeded");
    }
}

void Executor::run(const std::vector<TensorView>& inputs,
                   const std::vector<MutableTensorView>& outputs)
{
    if (!impl_)
    {
        throw TensorRtError(TensorRtErrorCode::InvalidArgument,
                            "execution stage: executor state is unavailable");
    }
    detail::ExecutorStateGuard state_guard(
        impl_->in_use,
        [&]
        {
            detail::validate_executor_run_state(impl_->engine != nullptr,
                                                static_cast<bool>(impl_->context),
                                                impl_->invalidation_reason);
        });

    try
    {
        const std::vector<TensorDescriptor> input_descriptors =
            descriptors_with_mode(impl_->engine->tensors, TensorIoMode::Input);
        const std::vector<TensorDescriptor> output_descriptors =
            descriptors_with_mode(impl_->engine->tensors, TensorIoMode::Output);
        detail::validate_input_views(input_descriptors, inputs,
                                     impl_->engine->options.max_input_bytes);
        detail::validate_output_views(output_descriptors, outputs,
                                      impl_->engine->options.max_output_bytes);
        reject_view_aliases(inputs, outputs);

        detail::CudaDeviceScope device_scope(impl_->engine->options.device_id);
        validate_cuda_views(inputs, outputs, impl_->engine->options.device_id);
        set_input_shapes_and_validate_outputs(*impl_->context, input_descriptors,
                                              output_descriptors, inputs, outputs,
                                              impl_->engine->options.max_output_bytes);

        cudaStream_t stream              = impl_->stream->get();
        bool         stream_work_pending = false;
        try
        {
            bind_tensors(*impl_->context, stream, impl_->engine->tensors, impl_->staging, inputs,
                         outputs, impl_->engine->options, stream_work_pending);

            stream_work_pending = true;
            if (!impl_->context->enqueueV3(stream))
            {
                throw_tensorrt("enqueue stage: enqueueV3 failed");
            }

            download_host_outputs(stream, impl_->engine->tensors, impl_->staging, outputs);

            detail::check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                               "execution completion");
            stream_work_pending = false;
            if (!clear_tensor_addresses(*impl_->context, impl_->engine->tensors))
            {
                detail::record_tensor_address_cleanup_failure(impl_->invalidation_reason);
                impl_->context.reset();
                throw_tensorrt("tensor address cleanup stage: TensorRT rejected address reset");
            }

            copy_host_outputs(impl_->engine->tensors, impl_->staging, outputs);
        }
        catch (...)
        {
            if (stream_work_pending)
            {
                (void)cudaStreamSynchronize(stream);
            }
            if (impl_->context &&
                !clear_tensor_addresses(*impl_->context, impl_->engine->tensors))
            {
                // Discarding the context guarantees that no caller-owned pointer remains retained
                // when TensorRT refuses to restore the default null bindings.
                detail::record_tensor_address_cleanup_failure(impl_->invalidation_reason);
                impl_->context.reset();
            }
            throw;
        }
        device_scope.restore();
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("execution stage: allocation failed");
    }
    catch (const std::length_error&)
    {
        throw_resource("execution stage: container capacity exceeded");
    }
}

std::vector<HostTensor> Executor::run_dynamic(
    const std::vector<TensorView>& inputs,
    const std::vector<DynamicOutputRequest>& outputs)
{
    if (!impl_)
    {
        throw TensorRtError(TensorRtErrorCode::InvalidArgument,
                            "dynamic execution stage: executor state is unavailable");
    }
    detail::ExecutorStateGuard state_guard(
        impl_->in_use,
        [&]
        {
            detail::validate_executor_run_state(impl_->engine != nullptr,
                                                static_cast<bool>(impl_->context),
                                                impl_->invalidation_reason);
        });

    try
    {
        const std::vector<TensorDescriptor> input_descriptors =
            descriptors_with_mode(impl_->engine->tensors, TensorIoMode::Input);
        const std::vector<TensorDescriptor> output_descriptors =
            descriptors_with_mode(impl_->engine->tensors, TensorIoMode::Output);
        detail::validate_input_views(input_descriptors, inputs,
                                     impl_->engine->options.max_input_bytes);
        detail::validate_dynamic_output_requests(output_descriptors, outputs,
                                                 impl_->engine->options.max_output_bytes);
        reject_view_aliases(inputs, {});

        detail::CudaDeviceScope device_scope(impl_->engine->options.device_id);
        validate_cuda_views(inputs, {}, impl_->engine->options.device_id);
        set_input_shapes(*impl_->context, input_descriptors, inputs);

        cudaStream_t stream              = impl_->stream->get();
        bool         stream_work_pending = false;
        std::vector<std::unique_ptr<BoundedOutputAllocator>> allocators;
        try
        {
            bind_dynamic_tensors(*impl_->context, stream, impl_->engine->tensors,
                                 impl_->staging, inputs, outputs, impl_->engine->options,
                                 stream_work_pending, allocators);

            stream_work_pending = true;
            if (!impl_->context->enqueueV3(stream))
            {
                for (const auto& allocator : allocators)
                {
                    if (allocator->capacity_exceeded())
                    {
                        throw_resource(
                            "dynamic output execution stage: TensorRT requested " +
                            std::to_string(allocator->requested_bytes()) +
                            " bytes beyond the prevalidated output bound");
                    }
                }
                throw_tensorrt("dynamic output enqueue stage: enqueueV3 failed");
            }

            detail::check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                               "dynamic output execution completion");
            stream_work_pending = false;

            std::vector<HostTensor> results;
            results.reserve(output_descriptors.size());
            std::size_t aggregate_actual_bytes = 0;
            std::size_t allocator_index = 0;
            for (std::size_t index = 0; index < impl_->engine->tensors.size(); ++index)
            {
                const TensorDescriptor& descriptor = impl_->engine->tensors[index];
                if (descriptor.mode != TensorIoMode::Output)
                {
                    continue;
                }
                const DynamicOutputRequest& request =
                    find_dynamic_output(outputs, descriptor.name);
                const BoundedOutputAllocator& allocator = *allocators[allocator_index++];
                if (!allocator.shape_notified())
                {
                    throw_tensorrt("dynamic output shape stage: TensorRT did not notify shape for " +
                                   descriptor.name);
                }
                TensorShape shape = resolved_data_dependent_output_shape(
                    allocator.shape(), descriptor.name);
                const std::size_t actual_bytes =
                    detail::checked_data_dependent_shape_byte_size(
                        shape, descriptor.data_type, "dynamic output shape validation");
                if (actual_bytes > request.max_byte_size)
                {
                    throw_resource("dynamic output shape stage: actual bytes exceed caller limit "
                                   "for " + descriptor.name);
                }
                if (actual_bytes >
                    impl_->engine->options.max_output_bytes - aggregate_actual_bytes)
                {
                    throw_resource("dynamic output shape stage: aggregate actual bytes exceed "
                                   "configured limit");
                }
                aggregate_actual_bytes += actual_bytes;

                HostTensor result;
                result.name      = descriptor.name;
                result.data_type = descriptor.data_type;
                result.shape     = std::move(shape);
                result.bytes.resize(actual_bytes);
                results.push_back(std::move(result));

                if (actual_bytes != 0)
                {
                    stream_work_pending = true;
                    detail::check_cuda(
                        cudaMemcpyAsync(impl_->staging[index].host.data(),
                                        impl_->staging[index].device.data(), actual_bytes,
                                        cudaMemcpyDeviceToHost, stream),
                        "cudaMemcpyAsync", "dynamic output download");
                }
            }

            if (stream_work_pending)
            {
                detail::check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                                   "dynamic output download completion");
                stream_work_pending = false;
            }
            if (!clear_dynamic_tensor_bindings(*impl_->context, impl_->engine->tensors))
            {
                detail::record_tensor_address_cleanup_failure(impl_->invalidation_reason);
                impl_->context.reset();
                throw_tensorrt("dynamic tensor binding cleanup stage: TensorRT rejected reset");
            }

            std::size_t result_index = 0;
            for (std::size_t index = 0; index < impl_->engine->tensors.size(); ++index)
            {
                if (impl_->engine->tensors[index].mode != TensorIoMode::Output)
                {
                    continue;
                }
                HostTensor& result = results[result_index++];
                if (!result.bytes.empty())
                {
                    std::memcpy(result.bytes.data(), impl_->staging[index].host.data(),
                                result.bytes.size());
                }
            }

            device_scope.restore();
            return results;
        }
        catch (...)
        {
            if (stream_work_pending)
            {
                (void)cudaStreamSynchronize(stream);
            }
            if (impl_->context &&
                !clear_dynamic_tensor_bindings(*impl_->context, impl_->engine->tensors))
            {
                detail::record_tensor_address_cleanup_failure(impl_->invalidation_reason);
                impl_->context.reset();
            }
            throw;
        }
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("dynamic execution stage: allocation failed");
    }
    catch (const std::length_error&)
    {
        throw_resource("dynamic execution stage: container capacity exceeded");
    }
}

} // namespace kfcore::tensorrt
