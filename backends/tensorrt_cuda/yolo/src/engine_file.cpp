#include "engine_file.hpp"

#include "kfcore/yolo/error.hpp"

#include <ios>
#include <limits>
#include <new>
#include <stdexcept>

namespace kfcore::yolo::detail
{

std::vector<std::byte> read_engine_stream_exact(std::istream& stream, std::size_t expected_bytes)
{
    if (expected_bytes > static_cast<std::size_t>((std::numeric_limits<std::streamsize>::max)()))
    {
        throw YoloError(YoloErrorCode::ResourceLimitExceeded,
                        "engine file read stage: serialized engine exceeds stream capacity");
    }

    try
    {
        std::vector<std::byte> bytes(expected_bytes);
        stream.seekg(0, std::ios::beg);
        if (!stream)
        {
            throw YoloError(YoloErrorCode::FileIo,
                            "engine file read stage: could not seek to the beginning");
        }
        if (!stream.read(reinterpret_cast<char*>(bytes.data()),
                         static_cast<std::streamsize>(bytes.size())))
        {
            throw YoloError(YoloErrorCode::FileIo,
                            "engine file read stage: could not read the complete initial size");
        }

        char extra_byte = 0;
        stream.read(&extra_byte, 1);
        if (stream.gcount() != 0)
        {
            throw YoloError(
                YoloErrorCode::FileIo,
                "engine file read stage: serialized engine grew or contains extra bytes after "
                "the initial size measurement; publish engines by temporary file and atomic "
                "rename");
        }
        if (!stream.eof())
        {
            throw YoloError(YoloErrorCode::FileIo,
                            "engine file read stage: could not verify end of file");
        }
        return bytes;
    }
    catch (const std::bad_alloc&)
    {
        throw YoloError(YoloErrorCode::ResourceLimitExceeded,
                        "engine file read stage: serialized engine allocation failed");
    }
    catch (const std::length_error&)
    {
        throw YoloError(YoloErrorCode::ResourceLimitExceeded,
                        "engine file read stage: serialized engine size exceeds container "
                        "capacity");
    }
}

} // namespace kfcore::yolo::detail
