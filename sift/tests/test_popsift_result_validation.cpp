#include "popsift_result_validation.hpp"

#include "tinytest.hpp"

#include <array>
#include <cstddef>
#include <string>

using namespace kfcore::sift;

namespace
{

template <typename Callable>
void expect_backend_error(Callable&& callable)
{
    try
    {
        callable();
        check(false);
    }
    catch (const SiftError& error)
    {
        check(error.code() == SiftErrorCode::BackendFailure);
        check(std::string(error.what()).find("descriptor") != std::string::npos);
    }
}

} // namespace

spec("PopSift descriptor storage validation")
{
    it("maps aligned pointers inside one continuous allocation")
    {
        constexpr std::size_t kElementBytes = 32;
        alignas(32) std::array<std::byte, kElementBytes * 3U> storage = {};

        check(detail::descriptor_storage_index(storage.data(), 3U, kElementBytes,
                                               storage.data() + kElementBytes) == 1U);
    }

    it("rejects null, misaligned, and out-of-range descriptor pointers")
    {
        constexpr std::size_t kElementBytes = 32;
        alignas(32) std::array<std::byte, kElementBytes * 3U> storage = {};

        expect_backend_error([&] {
            (void)detail::descriptor_storage_index(storage.data(), 3U, kElementBytes, nullptr);
        });
        expect_backend_error([&] {
            (void)detail::descriptor_storage_index(storage.data(), 3U, kElementBytes,
                                                   storage.data() + 1U);
        });
        expect_backend_error([&] {
            (void)detail::descriptor_storage_index(storage.data(), 3U, kElementBytes,
                                                   storage.data() + storage.size());
        });
    }
}
