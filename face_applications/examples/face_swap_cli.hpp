#pragma once

#include <optional>
#include <string>
#include <vector>

namespace kfcore::face_applications::cli
{

struct Arguments
{
    std::string source;
    std::string target;
    std::string output;
    std::string detector;
    std::string face68;
    std::string arcface;
    std::string inswapper;
    std::string matrix;
    std::optional<std::string> gfpgan;
    std::optional<std::string> age_gender;
};

[[nodiscard]] Arguments parse_arguments(const std::vector<std::string>& values);

} // namespace kfcore::face_applications::cli
