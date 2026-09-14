#include "kfcore/runtime/model_package.hpp"

#include "kfcore/runtime/error.hpp"

#include <data_bind.h>
#include <salts/crypto.h>

#include <array>
#include <cctype>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string_view>

namespace kfcore::runtime
{
namespace
{

constexpr std::size_t kMaxManifestBytes = 1024U * 1024U;

constexpr char kManifestSchema[] =
    "message Artifact { "
    "string id; string format; string path; optional string flavor; string sha256; "
    "string backend; string device; optional string source_artifact; optional string source_sha256; "
    "optional uint32 runtime_major; optional string compute_capability; optional string precision; "
    "optional string profile; } "
    "message Package { string schema; string id; string version; string model_type; "
    "optional string variant; list<Artifact> artifacts; }";

struct DataBindDeleter
{
    void operator()(DataBind* value) const noexcept { data_bind_free(value); }
};
struct RecordDeleter
{
    void operator()(DataBindRecord* value) const noexcept { data_bind_record_free(value); }
};

[[noreturn]] void throw_manifest(const std::string& message)
{
    throw RuntimeError(RuntimeErrorCode::InvalidModelPackage, message);
}

std::string read_text_file(const std::filesystem::path& path, std::size_t limit)
{
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    if (error || bytes == 0U || bytes > limit)
    {
        throw RuntimeError(RuntimeErrorCode::FileIo,
                           "model package manifest is missing, empty, or too large: " + path.u8string());
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw RuntimeError(RuntimeErrorCode::FileIo,
                           "cannot open model package manifest: " + path.u8string());
    }
    std::string text(static_cast<std::size_t>(bytes), '\0');
    stream.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!stream)
    {
        throw RuntimeError(RuntimeErrorCode::FileIo,
                           "cannot read complete model package manifest: " + path.u8string());
    }
    return text;
}

std::string required_string(const DataBindRecord* record, const char* name)
{
    DataBindStringView view = DATA_BIND_STRING_VIEW_INIT;
    DataBindError error = DATA_BIND_ERROR_INIT;
    if (data_bind_record_get_string(record, name, &view, &error) != DATA_BIND_OK ||
        view.data == nullptr || view.length == 0U)
    {
        throw_manifest(std::string("model.json requires non-empty string field '") + name + "'");
    }
    return std::string(view.data, view.length);
}

std::string optional_string(const DataBindRecord* record, const char* name)
{
    DataBindRecordField field = DATA_BIND_RECORD_FIELD_INIT;
    DataBindError error = DATA_BIND_ERROR_INIT;
    if (data_bind_record_find_field(record, name, &field, &error) != DATA_BIND_OK)
    {
        return {};
    }
    DataBindStringView view = DATA_BIND_STRING_VIEW_INIT;
    if (data_bind_record_field_get_string(&field, &view, &error) != DATA_BIND_OK)
    {
        throw_manifest(std::string("model.json field '") + name + "' must be a string");
    }
    return view.data == nullptr ? std::string{} : std::string(view.data, view.length);
}

std::uint32_t optional_u32(const DataBindRecord* record, const char* name)
{
    DataBindRecordField field = DATA_BIND_RECORD_FIELD_INIT;
    DataBindError error = DATA_BIND_ERROR_INIT;
    if (data_bind_record_find_field(record, name, &field, &error) != DATA_BIND_OK)
    {
        return 0U;
    }
    std::uint32_t value = 0U;
    if (data_bind_record_field_get_u32(&field, &value, &error) != DATA_BIND_OK)
    {
        throw_manifest(std::string("model.json field '") + name + "' must be uint32");
    }
    return value;
}

bool is_lower_hex_sha256(std::string_view value)
{
    if (value.size() != SALTS_CRYPTO_SHA256_DIGEST_SIZE * 2U)
    {
        return false;
    }
    for (char c : value)
    {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
        {
            return false;
        }
    }
    return true;
}

bool path_within(const std::filesystem::path& root, const std::filesystem::path& candidate)
{
    auto root_it = root.begin();
    auto candidate_it = candidate.begin();
    for (; root_it != root.end(); ++root_it, ++candidate_it)
    {
        if (candidate_it == candidate.end() || *root_it != *candidate_it)
        {
            return false;
        }
    }
    return true;
}

std::filesystem::path normalize_artifact_path(const std::filesystem::path& root,
                                               const std::filesystem::path& relative)
{
    if (relative.empty() || relative.is_absolute())
    {
        throw_manifest("artifact path must be non-empty and relative to package root");
    }
    const auto lexical = relative.lexically_normal();
    if (lexical.empty() || *lexical.begin() == "..")
    {
        throw_manifest("artifact path escapes package root: " + relative.u8string());
    }
    std::error_code error;
    const auto candidate = std::filesystem::weakly_canonical(root / lexical, error);
    if (error || !path_within(root, candidate))
    {
        throw_manifest("artifact path escapes package root: " + relative.u8string());
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
        throw RuntimeError(RuntimeErrorCode::BackendFailure, "failed to initialize SHA-256");
    }
    std::array<char, 64U * 1024U> buffer{};
    while (stream)
    {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = stream.gcount();
        if (count > 0 && salts_crypto_sha256_update(
                           &context, buffer.data(), static_cast<std::size_t>(count)) != SALTS_CRYPTO_OK)
        {
            throw RuntimeError(RuntimeErrorCode::BackendFailure, "failed to update SHA-256");
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
        throw RuntimeError(RuntimeErrorCode::BackendFailure, "failed to finalize SHA-256");
    }
    static constexpr char hex[] = "0123456789abcdef";
    std::string result(digest.size() * 2U, '0');
    for (std::size_t i = 0U; i < digest.size(); ++i)
    {
        result[i * 2U] = hex[digest[i] >> 4U];
        result[i * 2U + 1U] = hex[digest[i] & 0x0fU];
    }
    return result;
}

void validate_artifact_identity(const ModelArtifact& artifact)
{
    if (artifact.id.empty() || artifact.format.empty() || artifact.path.empty() ||
        artifact.sha256.empty() || artifact.backend.empty() || artifact.device.empty())
    {
        throw_manifest("each artifact requires id, format, path, sha256, backend, and device");
    }
    if (!is_lower_hex_sha256(artifact.sha256))
    {
        throw_manifest("artifact '" + artifact.id + "' has invalid lowercase SHA-256");
    }
    if (!artifact.source_sha256.empty() && !is_lower_hex_sha256(artifact.source_sha256))
    {
        throw_manifest("artifact '" + artifact.id + "' has invalid source SHA-256");
    }
}

} // namespace

ModelPackage ModelPackage::load(const std::filesystem::path& package_directory)
{
    std::error_code error;
    const auto root = std::filesystem::weakly_canonical(package_directory, error);
    if (error || !std::filesystem::is_directory(root, error) || error)
    {
        throw RuntimeError(RuntimeErrorCode::InvalidModelPackage,
                           "model package directory is invalid: " + package_directory.u8string());
    }

    const std::string json = read_text_file(root / "model.json", kMaxManifestBytes);
    DataBind* raw_codec = nullptr;
    DataBindError bind_error = DATA_BIND_ERROR_INIT;
    if (data_bind_create_from_text(kManifestSchema, sizeof(kManifestSchema) - 1U,
                                   &raw_codec, &bind_error) != DATA_BIND_OK)
    {
        throw RuntimeError(RuntimeErrorCode::InvalidModelPackage,
                           std::string("internal model package schema failed: ") + bind_error.message);
    }
    std::unique_ptr<DataBind, DataBindDeleter> codec(raw_codec);

    DataBindRecord* raw_record = nullptr;
    if (data_bind_record_from_json(codec.get(), "Package", json.data(), json.size(),
                                   &raw_record, &bind_error) != DATA_BIND_OK)
    {
        throw RuntimeError(RuntimeErrorCode::InvalidModelPackage,
                           std::string("invalid model.json: ") + bind_error.message);
    }
    std::unique_ptr<DataBindRecord, RecordDeleter> record(raw_record);

    if (required_string(record.get(), "schema") != "kfcore.model/1")
    {
        throw_manifest("model.json schema must be 'kfcore.model/1'");
    }

    ModelPackage result;
    result.root_ = root;
    result.id_ = required_string(record.get(), "id");
    result.version_ = required_string(record.get(), "version");
    result.model_type_ = required_string(record.get(), "model_type");
    result.variant_ = optional_string(record.get(), "variant");

    DataBindListView list = DATA_BIND_LIST_VIEW_INIT;
    if (data_bind_record_get_list(record.get(), "artifacts", &list, &bind_error) != DATA_BIND_OK ||
        data_bind_list_view_count(&list) == 0U)
    {
        throw_manifest("model.json requires a non-empty artifacts list");
    }

    const std::size_t count = data_bind_list_view_count(&list);
    result.artifacts_.reserve(count);
    for (std::size_t index = 0U; index < count; ++index)
    {
        DataBindRecordField item = DATA_BIND_RECORD_FIELD_INIT;
        DataBindRecordView view = DATA_BIND_RECORD_VIEW_INIT;
        if (data_bind_list_view_at(&list, index, &item, &bind_error) != DATA_BIND_OK ||
            data_bind_record_field_get_object(&item, &view, &bind_error) != DATA_BIND_OK)
        {
            throw_manifest("model.json artifact entry is not an object");
        }

        ModelArtifact artifact;
        // RecordView has the same field API through find_field; use field-level getters.
        auto get_required = [&](const char* name) -> std::string {
            DataBindRecordField field = DATA_BIND_RECORD_FIELD_INIT;
            DataBindStringView text = DATA_BIND_STRING_VIEW_INIT;
            if (data_bind_record_view_find_field(&view, name, &field, &bind_error) != DATA_BIND_OK ||
                data_bind_record_field_get_string(&field, &text, &bind_error) != DATA_BIND_OK ||
                text.data == nullptr || text.length == 0U)
            {
                throw_manifest(std::string("artifact requires non-empty field '") + name + "'");
            }
            return std::string(text.data, text.length);
        };
        auto get_optional = [&](const char* name) -> std::string {
            DataBindRecordField field = DATA_BIND_RECORD_FIELD_INIT;
            DataBindStringView text = DATA_BIND_STRING_VIEW_INIT;
            if (data_bind_record_view_find_field(&view, name, &field, &bind_error) != DATA_BIND_OK)
            {
                return {};
            }
            if (data_bind_record_field_get_string(&field, &text, &bind_error) != DATA_BIND_OK)
            {
                throw_manifest(std::string("artifact field '") + name + "' must be string");
            }
            return text.data == nullptr ? std::string{} : std::string(text.data, text.length);
        };
        auto get_optional_u32 = [&](const char* name) -> std::uint32_t {
            DataBindRecordField field = DATA_BIND_RECORD_FIELD_INIT;
            if (data_bind_record_view_find_field(&view, name, &field, &bind_error) != DATA_BIND_OK)
            {
                return 0U;
            }
            std::uint32_t value = 0U;
            if (data_bind_record_field_get_u32(&field, &value, &bind_error) != DATA_BIND_OK)
            {
                throw_manifest(std::string("artifact field '") + name + "' must be uint32");
            }
            return value;
        };

        artifact.id = get_required("id");
        artifact.format = get_required("format");
        artifact.path = get_required("path");
        artifact.flavor = get_optional("flavor");
        artifact.sha256 = get_required("sha256");
        artifact.backend = get_required("backend");
        artifact.device = get_required("device");
        artifact.source_artifact = get_optional("source_artifact");
        artifact.source_sha256 = get_optional("source_sha256");
        artifact.runtime_major = get_optional_u32("runtime_major");
        artifact.compute_capability = get_optional("compute_capability");
        artifact.precision = get_optional("precision");
        artifact.profile = get_optional("profile");
        validate_artifact_identity(artifact);
        (void)normalize_artifact_path(result.root_, artifact.path);

        for (const auto& existing : result.artifacts_)
        {
            if (existing.id == artifact.id)
            {
                throw_manifest("duplicate artifact id: " + artifact.id);
            }
        }
        result.artifacts_.push_back(std::move(artifact));
    }

    for (const auto& artifact : result.artifacts_)
    {
        if (!artifact.source_artifact.empty())
        {
            const auto& source = result.artifact(artifact.source_artifact);
            if (artifact.source_sha256.empty() || artifact.source_sha256 != source.sha256)
            {
                throw_manifest("artifact '" + artifact.id +
                               "' source_sha256 does not match source artifact declaration");
            }
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
    for (const auto& value : artifacts_)
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
    return normalize_artifact_path(root_, artifact.path);
}

void verify_model_artifact(const ModelPackage& package, const ModelArtifact& artifact)
{
    const auto path = package.artifact_path(artifact);
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
    {
        throw RuntimeError(RuntimeErrorCode::FileIo,
                           "model artifact is not a readable regular file: " + path.u8string());
    }
    const std::string actual = sha256_file(path);
    if (actual != artifact.sha256)
    {
        throw RuntimeError(RuntimeErrorCode::ArtifactIntegrity,
                           "SHA-256 mismatch for model artifact '" + artifact.id + "'");
    }

    if (!artifact.source_artifact.empty())
    {
        const auto& source = package.artifact(artifact.source_artifact);
        if (source.sha256 != artifact.source_sha256)
        {
            throw RuntimeError(RuntimeErrorCode::ArtifactIntegrity,
                               "source provenance mismatch for model artifact '" + artifact.id + "'");
        }
        const auto source_path = package.artifact_path(source);
        if (!std::filesystem::is_regular_file(source_path, error) || error ||
            sha256_file(source_path) != source.sha256)
        {
            throw RuntimeError(RuntimeErrorCode::ArtifactIntegrity,
                               "source artifact integrity failure for '" + artifact.id + "'");
        }
    }
}

} // namespace kfcore::runtime
