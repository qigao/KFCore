#pragma once

#include <string>
#include <vector>

namespace kfcore::face_applications::cli
{

struct Arguments
{
    std::string source;
    std::string target;
    std::string output;
};

[[nodiscard]] Arguments parse_arguments(const std::vector<std::string>& values);

} // namespace kfcore::face_applications::cli
