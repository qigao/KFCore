#include "engine_file.hpp"
#include "tinytest.hpp"

#include "kfcore/yolo/error.hpp"

#include <cstddef>
#include <sstream>
#include <string>

using namespace kfcore::yolo;

spec("TensorRT serialized engine file reads")
{
    it("accepts exactly the initially measured byte count")
    {
        std::istringstream stream("ab");

        const auto bytes = detail::read_engine_stream_exact(stream, 2);

        check(bytes.size() == std::size_t { 2 });
        check(static_cast<char>(bytes[0]) == 'a');
        check(static_cast<char>(bytes[1]) == 'b');
    }

    it("rejects bytes appended after the initially measured byte count")
    {
        std::istringstream stream("ab");

        bool threw = false;
        try
        {
            (void)detail::read_engine_stream_exact(stream, 1);
        }
        catch (const YoloError& error)
        {
            threw = true;
            check(error.code() == YoloErrorCode::FileIo);
            check(std::string(error.what()).find("grew or contains extra bytes") !=
                  std::string::npos);
        }
        check(threw);
    }
}
