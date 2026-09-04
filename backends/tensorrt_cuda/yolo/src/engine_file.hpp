#pragma once

#include <cstddef>
#include <istream>
#include <vector>

namespace kfcore::yolo::detail
{

std::vector<std::byte> read_engine_stream_exact(std::istream& stream, std::size_t expected_bytes);

} // namespace kfcore::yolo::detail
