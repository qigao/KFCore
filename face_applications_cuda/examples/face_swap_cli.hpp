#pragma once

#include <filesystem>
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

[[nodiscard]] Arguments parse_arguments(
    const std::vector<std::string>& values,
    const std::filesystem::path& model_root = {},
    const std::string& tensorrt_profile = {});
[[nodiscard]] Arguments parse_arguments_from_environment(
    const std::vector<std::string>& values);

} // namespace kfcore::face_applications::cli
