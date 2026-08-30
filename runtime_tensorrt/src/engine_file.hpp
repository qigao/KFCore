#pragma once

#include <cstddef>
#include <filesystem>
#include <istream>
#include <vector>

namespace kfcore::tensorrt::detail
{

std::vector<std::byte> read_engine_stream_exact(std::istream& stream,
                                                std::size_t expected_bytes);

std::vector<std::byte> read_engine_file_bounded(const std::filesystem::path& path,
                                                std::size_t hard_limit);

} // namespace kfcore::tensorrt::detail
