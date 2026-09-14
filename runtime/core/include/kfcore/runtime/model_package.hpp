#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace kfcore::runtime
{

struct ModelArtifact
{
    std::string id;
    std::string format;
    std::filesystem::path path;
    std::string flavor;
    std::string sha256;

    std::string backend;
    std::string device;

    std::string source_artifact;
    std::string source_sha256;
    std::uint32_t runtime_major = 0U;
    std::string compute_capability;
    std::string precision;
    std::string profile;
};

class ModelPackage final
{
public:
    [[nodiscard]] static ModelPackage load(const std::filesystem::path& package_directory);

    [[nodiscard]] const std::filesystem::path& root() const noexcept;
    [[nodiscard]] const std::string& id() const noexcept;
    [[nodiscard]] const std::string& version() const noexcept;
    [[nodiscard]] const std::string& model_type() const noexcept;
    [[nodiscard]] const std::string& variant() const noexcept;
    [[nodiscard]] const std::vector<ModelArtifact>& artifacts() const noexcept;
    [[nodiscard]] const ModelArtifact& artifact(std::string_view artifact_id) const;
    [[nodiscard]] std::filesystem::path artifact_path(const ModelArtifact& artifact) const;

private:
    std::filesystem::path root_;
    std::string id_;
    std::string version_;
    std::string model_type_;
    std::string variant_;
    std::vector<ModelArtifact> artifacts_;
};

} // namespace kfcore::runtime
