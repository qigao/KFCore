#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
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

    // TensorRT deployment compatibility. V1 supports exact runtime plans only;
    // version-compatible plans with embedded/externally loaded lean runtimes are
    // intentionally outside the V1 trust model.
    std::string runtime_version;
    std::string platform;
    std::string hardware_compatibility;
    std::string device_name;
    std::string compute_capability;
    std::string precision;
    std::string profile;
};

class ModelPackage final
{
public:
    /** Load a flat <name>.json manifest or a legacy package directory. */
    [[nodiscard]] static ModelPackage load(const std::filesystem::path& package_path);

    [[nodiscard]] const std::filesystem::path& root() const noexcept;
    [[nodiscard]] const std::string& id() const noexcept;
    [[nodiscard]] const std::string& version() const noexcept;
    [[nodiscard]] const std::string& model_type() const noexcept;
    [[nodiscard]] const std::string& variant() const noexcept;
    [[nodiscard]] const std::string& predicate_order_sha256() const noexcept;
    [[nodiscard]] const std::vector<ModelArtifact>& artifacts() const noexcept;
    [[nodiscard]] const ModelArtifact& artifact(std::string_view artifact_id) const;
    [[nodiscard]] std::filesystem::path artifact_path(const ModelArtifact& artifact) const;

private:
    std::filesystem::path root_;
    std::string id_;
    std::string version_;
    std::string model_type_;
    std::string variant_;
    std::string predicate_order_sha256_;
    std::vector<ModelArtifact> artifacts_;
};

void verify_model_artifact(const ModelPackage& package, const ModelArtifact& artifact);
[[nodiscard]] std::string compute_model_artifact_sha256(
    const std::filesystem::path& artifact_path);

} // namespace kfcore::runtime
