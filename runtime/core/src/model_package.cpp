#include "kfcore/runtime/model_package.hpp"

#include "kfcore/runtime/error.hpp"

#include <data_bind.h>
#include <salts/crypto.h>

#include <array>
#include <fstream>
#include <limits>
#include <memory>
#include <string_view>

namespace kfcore::runtime
{
namespace
{

constexpr std::size_t kMaxManifestBytes = 1024U * 1024U;
constexpr char kManifestSchema[] =
    "group Artifact { "
    "string id; string format; string path; optional string flavor; string sha256; "
    "string backend; string device; optional string source_artifact; optional string source_sha256; "
    "optional uint32 runtime_major; optional string compute_capability; optional string precision; "
    "optional string profile; } "
    "message Package { string schema; string id; string version; string model_type; "
    "optional string variant; group<Artifact> artifacts; }";

struct DataBindDeleter
{
    void operator()(DataBind* value) const noexcept { data_bind_free(value); }
};

struct RecordDeleter
{
    void operator()(DataBindRecord* value) const noexcept { data_bind_record_free(value); }
};

[[noreturn]] void invalid_package(std::string message)
{
    throw RuntimeError(RuntimeErrorCode::InvalidModelPackage, std::move(message));
}

std::string read_manifest(const std::filesystem::path& path)
{
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    if (error || bytes == 0U || bytes > kMaxManifestBytes)
    {
        throw RuntimeError(RuntimeErrorCode::FileIo,
                           "model.json is missing, empty, or exceeds the 1 MiB limit: " +
                               path.u8string());
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw RuntimeError(RuntimeErrorCode::FileIo,
                           "cannot open model.json: " + path.u8string());
    }
    std::string text(static_cast<std::size_t>(bytes), '\0');
    stream.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!stream)
    {
        throw RuntimeError(RuntimeErrorCode::FileIo,
                           "cannot read complete model.json: " + path.u8string());
    }
    return text;
}

std::string required_string(const DataBindRecord* record, const char* name)
{
    DataBindStringView value = DATA_BIND_STRING_VIEW_INIT;
    DataBindError error = DATA_BIND_ERROR_INIT;
    if (data_bind_record_get_string(record, name, &value, &error) != DATA_BIND_OK ||
        value.data == nullptr || value.length == 0U)
    {
        invalid_package(std::string("model.json requires non-empty string '") + name + "'");
    }
    return std::string(value.data, value.length);
}

std::string optional_string(const DataBindRecord* record, const char* name)
{
    DataBindRecordField field = DATA_BIND_RECORD_FIELD_INIT;
    DataBindError error = DATA_BIND_ERROR_INIT;
    if (data_bind_record_find_field(record, name, &field, &error) != DATA_BIND_OK)
    {
        return {};
    }
    DataBindStringView value = DATA_BIND_STRING_VIEW_INIT;
    if (data_bind_record_field_get_string(&field, &value, &error) != DATA_BIND_OK)
    {
        invalid_package(std::string("model.json field '") + name + "' must be a string");
    }
    return value.data == nullptr ? std::string{} : std::string(value.data, value.length);
}

std::string required_string(const DataBindRecordView& view, const char* name,
                            DataBindError& error)
{
    DataBindRecordField field = DATA_BIND_RECORD_FIELD_INIT;
    DataBindStringView value = DATA_BIND_STRING_VIEW_INIT;
    if (data_bind_record_view_find_field(&view, name, &field, &error) != DATA_BIND_OK ||
        data_bind_record_field_get_string(&field, &value, &error) != DATA_BIND_OK ||
        value.data == nullptr || value.length == 0U)
    {
        invalid_package(std::string("artifact requires non-empty string '") + name + "'");
    }
    return std::string(value.data, value.length);
}

std::string optional_string(const DataBindRecordView& view, const char* name,
                            DataBindError& error)
{
    DataBindRecordField field = DATA_BIND_RECORD_FIELD_INIT;
    if (data_bind_record_view_find_field(&view, name, &field, &error) != DATA_BIND_OK)
    {
        return {};
    }
    DataBindStringView value = DATA_BIND_STRING_VIEW_INIT;
    if (data_bind_record_field_get_string(&field, &value, &error) != DATA_BIND_OK)
    {
        invalid_package(std::string("artifact field '") + name + "' must be a string");
    }
    return value.data == nullptr ? std::string{} : std::string(value.data, value.length);
}

std::uint32_t optional_u32(const DataBindRecordView& view, const char* name,
                           DataBindError& error)
{
    DataBindRecordField field = DATA_BIND_RECORD_FIELD_INIT;
    if (data_bind_record_view_find_field(&view, name, &field, &error) != DATA_BIND_OK)
    {
        return 0U;
    }
    std::uint32_t value = 0U;
    if (data_bind_record_field_get_u32(&field, &value, &error) != DATA_BIND_OK)
    {
        invalid_package(std::string("artifact field '") + name + "' must be uint32");
    }
    return value;
}

bool valid_sha256(std::string_view value)
{
    if (value.size() != SALTS_CRYPTO_SHA256_DIGEST_SIZE * 2U)
    {
        return false;
    }
    for (const char c : value)
    {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
        {
            return false;
        }
    }
    return true;
}

std::filesystem::path contained_path(const std::filesystem::path& root,
                                     const std::filesystem::path& relative)
{
    if (relative.empty() || relative.is_absolute())
    {
        invalid_package("artifact path must be package-relative");
    }
    const auto lexical = relative.lexically_normal();
    if (lexical.empty() || *lexical.begin() == "..")
    {
        invalid_package("artifact path escapes package root: " + relative.u8string());
    }

    std::error_code error;
    const auto candidate = std::filesystem::weakly_canonical(root / lexical, error);
    if (error)
    {
        invalid_package("artifact path cannot be canonicalized: " + relative.u8string());
    }

    auto root_it = root.begin();
    auto candidate_it = candidate.begin();
    for (; root_it != root.end(); ++root_it, ++candidate_it)
    {
        if (candidate_it == candidate.end() || *root_it != *candidate_it)
        {
            invalid_package("artifact path escapes package root: " + relative.u8string());
        }
    }
    return candidate;
}

std::string sha256_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw RuntimeError(RuntimeErrorCode::FileIo,
                           "cannot open model artifact: " + path.u8string());
    }

    salts_crypto_sha256_ctx_t context{};
    if (salts_crypto_sha256_init(&context) != SALTS_CRYPTO_OK)
    {
        throw RuntimeError(RuntimeErrorCode::BackendFailure, "SHA-256 initialization failed");
    }

    std::array<char, 64U * 1024U> buffer{};
    while (stream)
    {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        if (count > 0 && salts_crypto_sha256_update(
                           &context, buffer.data(), static_cast<std::size_t>(count)) != SALTS_CRYPTO_OK)
        {
            throw RuntimeError(RuntimeErrorCode::BackendFailure, "SHA-256 update failed");
        }
    }
    if (!stream.eof())
    {
        throw RuntimeError(RuntimeErrorCode::FileIo,
                           "failed while reading model artifact: " + path.u8string());
    }

    std::array<std::uint8_t, SALTS_CRYPTO_SHA256_DIGEST_SIZE> digest{};
    if (salts_crypto_sha256_final(&context, digest.data()) != SALTS_CRYPTO_OK)
    {
        throw RuntimeError(RuntimeErrorCode::BackendFailure, "SHA-256 finalization failed");
    }

    static constexpr char kHex[] = "0123456789abcdef";
    std::string result(digest.size() * 2U, '0');
    for (std::size_t i = 0U; i < digest.size(); ++i)
    {
        result[i * 2U] = kHex[digest[i] >> 4U];
        result[i * 2U + 1U] = kHex[digest[i] & 0x0fU];
    }
    return result;
}

void validate_artifact(const ModelArtifact& artifact)
{
    if (artifact.id.empty() || artifact.format.empty() || artifact.path.empty() ||
        artifact.sha256.empty() || artifact.backend.empty() || artifact.device.empty())
    {
        invalid_package("each artifact requires id, format, path, sha256, backend, and device");
    }
    if (!valid_sha256(artifact.sha256))
    {
        invalid_package("artifact '" + artifact.id + "' has invalid lowercase SHA-256");
    }
    if (!artifact.source_sha256.empty() && !valid_sha256(artifact.source_sha256))
    {
        invalid_package("artifact '" + artifact.id + "' has invalid source SHA-256");
    }
}

} // namespace

ModelPackage ModelPackage::load(const std::filesystem::path& package_directory)
{
    std::error_code filesystem_error;
    const auto root = std::filesystem::weakly_canonical(package_directory, filesystem_error);
    if (filesystem_error || !std::filesystem::is_directory(root, filesystem_error) || filesystem_error)
    {
        throw RuntimeError(RuntimeErrorCode::InvalidModelPackage,
                           "invalid model package directory: " + package_directory.u8string());
    }

    const std::string json = read_manifest(root / "model.json");
    DataBind* raw_codec = nullptr;
    DataBindError error = DATA_BIND_ERROR_INIT;
    if (data_bind_create_from_text(kManifestSchema, sizeof(kManifestSchema) - 1U,
                                   &raw_codec, &error) != DATA_BIND_OK)
    {
        invalid_package(std::string("internal Model Package V1 schema is invalid: ") + error.message);
    }
    std::unique_ptr<DataBind, DataBindDeleter> codec(raw_codec);

    DataBindRecord* raw_record = nullptr;
    if (data_bind_record_from_json(codec.get(), "Package", json.data(), json.size(),
                                   &raw_record, &error) != DATA_BIND_OK)
    {
        invalid_package(std::string("invalid model.json: ") + error.message);
    }
    std::unique_ptr<DataBindRecord, RecordDeleter> record(raw_record);

    if (required_string(record.get(), "schema") != "kfcore.model/1")
    {
        invalid_package("model.json schema must be 'kfcore.model/1'");
    }

    ModelPackage result;
    result.root_ = root;
    result.id_ = required_string(record.get(), "id");
    result.version_ = required_string(record.get(), "version");
    result.model_type_ = required_string(record.get(), "model_type");
    result.variant_ = optional_string(record.get(), "variant");

    DataBindListView artifacts = DATA_BIND_LIST_VIEW_INIT;
    if (data_bind_record_get_list(record.get(), "artifacts", &artifacts, &error) != DATA_BIND_OK ||
        data_bind_list_view_count(&artifacts) == 0U)
    {
        invalid_package("model.json requires a non-empty artifacts group");
    }

    const std::size_t count = data_bind_list_view_count(&artifacts);
    result.artifacts_.reserve(count);
    for (std::size_t index = 0U; index < count; ++index)
    {
        DataBindRecordField item = DATA_BIND_RECORD_FIELD_INIT;
        DataBindRecordView view = DATA_BIND_RECORD_VIEW_INIT;
        if (data_bind_list_view_at(&artifacts, index, &item, &error) != DATA_BIND_OK ||
            data_bind_record_field_get_object(&item, &view, &error) != DATA_BIND_OK)
        {
            invalid_package("artifact group contains a non-object item");
        }

        ModelArtifact artifact;
        artifact.id = required_string(view, "id", error);
        artifact.format = required_string(view, "format", error);
        artifact.path = required_string(view, "path", error);
        artifact.flavor = optional_string(view, "flavor", error);
        artifact.sha256 = required_string(view, "sha256", error);
        artifact.backend = required_string(view, "backend", error);
        artifact.device = required_string(view, "device", error);
        artifact.source_artifact = optional_string(view, "source_artifact", error);
        artifact.source_sha256 = optional_string(view, "source_sha256", error);
        artifact.runtime_major = optional_u32(view, "runtime_major", error);
        artifact.compute_capability = optional_string(view, "compute_capability", error);
        artifact.precision = optional_string(view, "precision", error);
        artifact.profile = optional_string(view, "profile", error);
        validate_artifact(artifact);
        (void)contained_path(result.root_, artifact.path);

        for (const ModelArtifact& existing : result.artifacts_)
        {
            if (existing.id == artifact.id)
            {
                invalid_package("duplicate artifact id: " + artifact.id);
            }
        }
        result.artifacts_.push_back(std::move(artifact));
    }

    for (const ModelArtifact& artifact : result.artifacts_)
    {
        if (artifact.source_artifact.empty())
        {
            continue;
        }
        const ModelArtifact& source = result.artifact(artifact.source_artifact);
        if (artifact.source_sha256.empty() || artifact.source_sha256 != source.sha256)
        {
            invalid_package("artifact '" + artifact.id +
                            "' source_sha256 does not match its declared source artifact");
        }
    }
    return result;
}

const std::filesystem::path& ModelPackage::root() const noexcept { return root_; }
const std::string& ModelPackage::id() const noexcept { return id_; }
const std::string& ModelPackage::version() const noexcept { return version_; }
const std::string& ModelPackage::model_type() const noexcept { return model_type_; }
const std::string& ModelPackage::variant() const noexcept { return variant_; }
const std::vector<ModelArtifact>& ModelPackage::artifacts() const noexcept { return artifacts_; }

const ModelArtifact& ModelPackage::artifact(std::string_view artifact_id) const
{
    for (const ModelArtifact& value : artifacts_)
    {
        if (value.id == artifact_id)
        {
            return value;
        }
    }
    throw RuntimeError(RuntimeErrorCode::NotFound,
                       "model artifact not found: " + std::string(artifact_id));
}

std::filesystem::path ModelPackage::artifact_path(const ModelArtifact& artifact) const
{
    return contained_path(root_, artifact.path);
}

void verify_model_artifact(const ModelPackage& package, const ModelArtifact& artifact)
{
    const auto path = package.artifact_path(artifact);
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
    {
        throw RuntimeError(RuntimeErrorCode::FileIo,
                           "model artifact is not a regular file: " + path.u8string());
    }
    if (sha256_file(path) != artifact.sha256)
    {
        throw RuntimeError(RuntimeErrorCode::ArtifactIntegrity,
                           "SHA-256 mismatch for artifact '" + artifact.id + "'");
    }

    if (!artifact.source_artifact.empty())
    {
        const ModelArtifact& source = package.artifact(artifact.source_artifact);
        const auto source_path = package.artifact_path(source);
        if (source.sha256 != artifact.source_sha256 ||
            !std::filesystem::is_regular_file(source_path, error) || error ||
            sha256_file(source_path) != source.sha256)
        {
            throw RuntimeError(RuntimeErrorCode::ArtifactIntegrity,
                               "source provenance verification failed for artifact '" +
                                   artifact.id + "'");
        }
    }
}

} // namespace kfcore::runtime
