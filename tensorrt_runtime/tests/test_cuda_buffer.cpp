#include "cuda_buffer.hpp"
#include "kfcore/tensorrt/error.hpp"
#include "tinytest.hpp"

#include <cstddef>

using namespace kfcore::tensorrt;
using namespace kfcore::tensorrt::detail;

namespace
{

struct FakeMemoryState
{
    std::byte   device_allocations[2] {};
    std::byte   pinned_allocations[2] {};
    std::size_t device_alloc_count = 0;
    std::size_t pinned_alloc_count = 0;
    std::size_t device_free_count  = 0;
    std::size_t pinned_free_count  = 0;
    void*       last_device_free   = nullptr;
    void*       last_pinned_free   = nullptr;
    bool        fail_device_alloc  = false;
    bool        fail_pinned_alloc  = false;
    bool        fail_device_free   = false;
    bool        fail_pinned_free   = false;
};

FakeMemoryState state;

cudaError_t fake_malloc_device(void** pointer, std::size_t)
{
    if (state.fail_device_alloc)
    {
        return cudaErrorMemoryAllocation;
    }
    *pointer = &state.device_allocations[state.device_alloc_count++];
    return cudaSuccess;
}

cudaError_t fake_free_device(void* pointer)
{
    state.last_device_free = pointer;
    ++state.device_free_count;
    if (state.fail_device_free)
    {
        state.fail_device_free = false;
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

cudaError_t fake_malloc_pinned(void** pointer, std::size_t)
{
    if (state.fail_pinned_alloc)
    {
        return cudaErrorMemoryAllocation;
    }
    *pointer = &state.pinned_allocations[state.pinned_alloc_count++];
    return cudaSuccess;
}

cudaError_t fake_free_pinned(void* pointer)
{
    state.last_pinned_free = pointer;
    ++state.pinned_free_count;
    if (state.fail_pinned_free)
    {
        state.fail_pinned_free = false;
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

const CudaMemoryApi fake_api {
    fake_malloc_device,
    fake_free_device,
    fake_malloc_pinned,
    fake_free_pinned,
};

} // namespace

spec("TensorRT runtime CUDA buffers")
{
    before_each()
    {
        state = {};
    }

    it("keeps the device allocation when replacement allocation fails")
    {
        CudaBuffer buffer(fake_api);
        buffer.reserve(8, 16);
        void* original          = buffer.data();
        state.fail_device_alloc = true;

        check_throws_as(buffer.reserve(16, 16), TensorRtError);
        check(buffer.data() == original);
        check(buffer.capacity() == std::size_t { 8 });
        check(state.device_free_count == std::size_t { 0 });
    }

    it("does not retain an old device pointer after its release reports an error")
    {
        {
            CudaBuffer buffer(fake_api);
            buffer.reserve(8, 16);
            void* original         = buffer.data();
            state.fail_device_free = true;

            bool threw = false;
            try
            {
                buffer.reserve(16, 16);
            }
            catch (const TensorRtError& error)
            {
                threw = true;
                check(std::string(error.what()).find("committed cleanup") != std::string::npos);
            }
            check_true(threw);
            check(buffer.data() != original);
            check(buffer.data() == static_cast<void*>(&state.device_allocations[1]));
            check(buffer.capacity() == std::size_t { 16 });
            check(state.last_device_free == original);
            check(state.device_free_count == std::size_t { 1 });
        }
        check(state.device_free_count == std::size_t { 2 });
        check(state.last_device_free == static_cast<void*>(&state.device_allocations[1]));
    }

    it("keeps the pinned allocation when replacement allocation fails")
    {
        PinnedHostBuffer buffer(fake_api);
        buffer.reserve(8, 16);
        void* original          = buffer.data();
        state.fail_pinned_alloc = true;

        check_throws_as(buffer.reserve(16, 16), TensorRtError);
        check(buffer.data() == original);
        check(buffer.capacity() == std::size_t { 8 });
        check(state.pinned_free_count == std::size_t { 0 });
    }

    it("does not retain an old pinned pointer after its release reports an error")
    {
        {
            PinnedHostBuffer buffer(fake_api);
            buffer.reserve(8, 16);
            void* original         = buffer.data();
            state.fail_pinned_free = true;

            bool threw = false;
            try
            {
                buffer.reserve(16, 16);
            }
            catch (const TensorRtError& error)
            {
                threw = true;
                check(std::string(error.what()).find("committed cleanup") != std::string::npos);
            }
            check_true(threw);
            check(buffer.data() != original);
            check(buffer.data() == static_cast<void*>(&state.pinned_allocations[1]));
            check(buffer.capacity() == std::size_t { 16 });
            check(state.last_pinned_free == original);
            check(state.pinned_free_count == std::size_t { 1 });
        }
        check(state.pinned_free_count == std::size_t { 2 });
        check(state.last_pinned_free == static_cast<void*>(&state.pinned_allocations[1]));
    }
}
