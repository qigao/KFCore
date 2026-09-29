#include "kfcore/runtime/model_package.hpp"

#include <filesystem>
#include <string>

namespace kfcore::runtime
{

std::string compute_model_artifact_sha256(
    const std::filesystem::path& artifact_path)
{
    const auto size = std::filesystem::file_size(artifact_path);
    return "test-" + std::to_string(size);
}

} // namespace kfcore::runtime
