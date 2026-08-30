#include "engine_file.hpp"
#include "kfcore/tensorrt/error.hpp"
#include "tinytest.hpp"

#include <cstddef>
#include <sstream>
#include <string>

using namespace kfcore::tensorrt;

namespace
{

void check_file_error(std::size_t expected_bytes, const char* expected_message)
{
    std::istringstream stream("ab");
    bool               threw = false;
    try
    {
        (void)detail::read_engine_stream_exact(stream, expected_bytes);
    }
    catch (const TensorRtError& error)
    {
        threw = true;
        check(error.code() == TensorRtErrorCode::FileIo);
        check(std::string(error.what()).find(expected_message) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

spec("TensorRT runtime serialized engine file reads")
{
    it("accepts exactly the initially measured byte count")
    {
        std::istringstream stream("ab");

        const auto bytes = detail::read_engine_stream_exact(stream, 2);

        check(bytes.size() == std::size_t { 2 });
        check(static_cast<char>(bytes[0]) == 'a');
        check(static_cast<char>(bytes[1]) == 'b');
    }

    it("rejects a stream shorter than the initially measured byte count")
    {
        check_file_error(3, "complete initial size");
    }

    it("rejects bytes appended after the initially measured byte count")
    {
        check_file_error(1, "grew or contains extra bytes");
    }
}
