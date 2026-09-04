#include "engine_file.hpp"

#include "kfcore/tensorrt/error.hpp"

#include <cstdint>
#include <fstream>
#include <ios>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

namespace kfcore::tensorrt::detail
{
namespace
{

    [[noreturn]] void throw_file(std::string message)
    {
        throw TensorRtError(TensorRtErrorCode::FileIo, std::move(message));
    }

    [[noreturn]] void throw_deserialize(std::string message)
    {
        throw TensorRtError(TensorRtErrorCode::EngineDeserialize, std::move(message));
    }

    [[noreturn]] void throw_resource(std::string message)
    {
        throw TensorRtError(TensorRtErrorCode::ResourceLimitExceeded, std::move(message));
    }

} // namespace

std::vector<std::byte> read_engine_stream_exact(std::istream& stream,
                                                std::size_t expected_bytes)
{
    if (expected_bytes > static_cast<std::size_t>((std::numeric_limits<std::streamsize>::max)()))
    {
        throw_resource("engine file read stage: serialized engine exceeds stream capacity");
    }

    try
    {
        std::vector<std::byte> bytes(expected_bytes);
        stream.seekg(0, std::ios::beg);
        if (!stream)
        {
            throw_file("engine file read stage: could not seek to the beginning");
        }
        if (!stream.read(reinterpret_cast<char*>(bytes.data()),
                         static_cast<std::streamsize>(bytes.size())))
        {
            throw_file("engine file read stage: could not read the complete initial size");
        }

        char extra_byte = 0;
        stream.read(&extra_byte, 1);
        if (stream.gcount() != 0)
        {
            throw_file(
                "engine file read stage: serialized engine grew or contains extra bytes after "
                "the initial size measurement; publish engines by temporary file and atomic "
                "rename");
        }
        if (!stream.eof())
        {
            throw_file("engine file read stage: could not verify end of file");
        }
        return bytes;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("engine file read stage: serialized engine allocation failed");
    }
    catch (const std::length_error&)
    {
        throw_resource("engine file read stage: serialized engine size exceeds container capacity");
    }
}

std::vector<std::byte> read_engine_file_bounded(const std::filesystem::path& path,
                                                std::size_t hard_limit)
{
    if (path.empty())
    {
        throw_file("engine file open stage: path must not be empty");
    }
    if (hard_limit == 0)
    {
        throw_resource("engine file read stage: serialized engine hard limit must be positive");
    }

    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream.is_open())
    {
        throw_file("engine file open stage: cannot open " + path.string());
    }
    const std::ifstream::pos_type end_position = stream.tellg();
    if (end_position <= std::ifstream::pos_type { 0 })
    {
        throw_deserialize("engine file read stage: serialized engine is empty");
    }

    const auto engine_bytes = static_cast<std::uintmax_t>(end_position);
    if (engine_bytes > static_cast<std::uintmax_t>(hard_limit) ||
        engine_bytes > static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)()) ||
        engine_bytes > static_cast<std::uintmax_t>((std::numeric_limits<std::streamsize>::max)()))
    {
        throw_resource("engine file read stage: serialized engine exceeds configured hard limit");
    }
    return read_engine_stream_exact(stream, static_cast<std::size_t>(engine_bytes));
}

} // namespace kfcore::tensorrt::detail
