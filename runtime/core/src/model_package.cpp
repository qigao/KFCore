#include "kfcore/runtime/model_package.hpp"

#include "kfcore/runtime/error.hpp"

#include <data_bind.h>
#include <salts/crypto.h>

#include <array>
#include <cctype>
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
    "optional string runtime_version; optional string platform; "
    "optional string hardware_compatibility; optional string device_name; "
    "optional string compute_capability; optional string precision; optional string profile; } "
    "message Package { group<Artifact> artifacts; [name(\"schema\")] string schema_id; "
    "string id; string version; string model_type; optional string variant; "
    "optional string predicate_order_sha256; }";

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

bool valid_compute_capability(std::string_view value)
{
    const std::size_t dot = value.find('.');
    if (dot == std::string_view::npos || dot == 0U || dot + 1U >= value.size() ||
        value.find('.', dot + 1U) != std::string_view::npos)
    {
        return false;
    }
    for (std::size_t i = 0U; i < value.size(); ++i)
    {
        if (i == dot)
        {
            continue;
        }
        if (!std::isdigit(static_cast<unsigned char>(value[i])))
        {
            return false;
        }
    }
    return value.front() != '0';
}

bool valid_runtime_version(std::string_view value)
{
    std::size_t component = 0U;
    std::size_t start = 0U;
    while (start <= value.size())
    {
        const std::size_t end = value.find('.', start);
        const std::size_t stop = end == std::string_view::npos ? value.size() : end;
        if (stop == start || component >= 4U)
        {
            return false;
        }
        bool nonzero = false;
        for (std::size_t index = start; index < stop; ++index)
        {
            const unsigned char c = static_cast<unsigned char>(value[index]);
            if (!std::isdigit(c))
            {
                return false;
            }
            nonzero = nonzero || value[index] != '0';
        }
        if (component == 0U && !nonzero)
        {
            return false;
        }
        ++component;
        if (end == std::string_view::npos)
        {
            break;
        }
        start = end + 1U;
    }
    return component == 4U;
}

bool valid_platform(std::string_view value)
{
    return value == "windows-x86_64" || value == "windows-aarch64" ||
           value == "linux-x86_64" || value == "linux-aarch64";
}

bool valid_hardware_compatibility(std::string_view value)
{
    return value == "exact-device" || value == "same-compute-capability";
}

bool valid_yolo_flavor(std::string_view value)
{
    return value == "raw-yolo" || value == "raw-yolox" ||
           value == "compact-nms" || value == "efficient-nms";
}

bool cuda_device_constraint(std::string_view value)
{
    return value == "cuda" || value.rfind("cuda:", 0U) == 0U;
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

    if (artifact.format == "tensorrt-engine")
    {
        if (artifact.backend != "tensorrt")
        {
            invalid_package("TensorRT artifact '" + artifact.id +
                            "' must declare backend='tensorrt'");
        }
        if (!cuda_device_constraint(artifact.device))
        {
            invalid_package("TensorRT artifact '" + artifact.id +
                            "' must declare a CUDA device constraint");
        }
        if (artifact.source_artifact.empty() || artifact.source_sha256.empty())
        {
            invalid_package("TensorRT artifact '" + artifact.id +
                            "' requires source_artifact and source_sha256 provenance");
        }
        if (!valid_runtime_version(artifact.runtime_version))
        {
            invalid_package("TensorRT artifact '" + artifact.id +
                            "' requires runtime_version in major.minor.patch.build form");
        }
        if (!valid_platform(artifact.platform))
        {
            invalid_package("TensorRT artifact '" + artifact.id +
                            "' requires a supported platform identifier");
        }
        if (!valid_hardware_compatibility(artifact.hardware_compatibility))
        {
            invalid_package("TensorRT artifact '" + artifact.id +
                            "' requires hardware_compatibility='exact-device' or "
                            "'same-compute-capability'");
        }
        if (artifact.hardware_compatibility == "exact-device" && artifact.device_name.empty())
        {
            invalid_package("TensorRT artifact '" + artifact.id +
                            "' with exact-device compatibility requires device_name");
        }
        if (!valid_compute_capability(artifact.compute_capability))
        {
            invalid_package("TensorRT artifact '" + artifact.id +
                            "' requires compute_capability in major.minor form");
        }
        if (artifact.precision.empty())
        {
            invalid_package("TensorRT artifact '" + artifact.id +
                            "' requires precision metadata");
        }
    }
}

} // namespace

std::string compute_model_artifact_sha256(
    const std::filesystem::path& artifact_path)
{
    return sha256_file(artifact_path);
}

ModelPackage ModelPackage::load(const std::filesystem::path& package_directory)
{
    std::error_code filesystem_error;
    const auto input = std::filesystem::weakly_canonical(package_directory, filesystem_error);
    if (filesystem_error)
    {
        throw RuntimeError(RuntimeErrorCode::InvalidModelPackage,
                           "invalid model package path: " + package_directory.u8string());
    }

    std::filesystem::path root;
    std::filesystem::path manifest_path;
    if (std::filesystem::is_directory(input, filesystem_error))
    {
        root = input;
        manifest_path = root / "model.json";
    }
    else if (std::filesystem::is_regular_file(input, filesystem_error) &&
             input.extension() == ".json")
    {
        manifest_path = input;
        root = input.parent_path();
    }
    else
    {
        throw RuntimeError(RuntimeErrorCode::InvalidModelPackage,
                           "invalid model package path: " + package_directory.u8string());
    }

    const std::string json = read_manifest(manifest_path);
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

    if (required_string(record.get(), "schema_id") != "kfcore.model/1")
    {
        invalid_package("model.json schema must be 'kfcore.model/1'");
    }

    ModelPackage result;
    result.root_ = root;
    result.id_ = required_string(record.get(), "id");
    result.version_ = required_string(record.get(), "version");
    result.model_type_ = required_string(record.get(), "model_type");
    result.variant_ = optional_string(record.get(), "variant");
    result.predicate_order_sha256_ =
        optional_string(record.get(), "predicate_order_sha256");
    if (!result.predicate_order_sha256_.empty() &&
        !valid_sha256(result.predicate_order_sha256_))
    {
        invalid_package("predicate_order_sha256 must be a lowercase SHA-256 digest");
    }

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
        artifact.runtime_version = optional_string(view, "runtime_version", error);
        artifact.platform = optional_string(view, "platform", error);
        artifact.hardware_compatibility = optional_string(view, "hardware_compatibility", error);
        artifact.device_name = optional_string(view, "device_name", error);
        artifact.compute_capability = optional_string(view, "compute_capability", error);
        artifact.precision = optional_string(view, "precision", error);
        artifact.profile = optional_string(view, "profile", error);
        validate_artifact(artifact);
        if (result.model_type_ == "yolo-detection" && !valid_yolo_flavor(artifact.flavor))
        {
            invalid_package("YOLO artifact '" + artifact.id +
                            "' requires flavor raw-yolo, raw-yolox, compact-nms, or efficient-nms");
        }
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
        if (artifact.source_artifact == artifact.id)
        {
            invalid_package("artifact '" + artifact.id + "' cannot derive from itself");
        }
        const ModelArtifact& source = result.artifact(artifact.source_artifact);
        if (artifact.source_sha256.empty() || artifact.source_sha256 != source.sha256)
        {
            invalid_package("artifact '" + artifact.id +
                            "' source_sha256 does not match its declared source artifact");
        }
        if (artifact.format == "tensorrt-engine" && source.format != "onnx")
        {
            invalid_package("TensorRT artifact '" + artifact.id +
                            "' must derive from an ONNX source artifact");
        }
    }
    return result;
}

const std::filesystem::path& ModelPackage::root() const noexcept { return root_; }
const std::string& ModelPackage::id() const noexcept { return id_; }
const std::string& ModelPackage::version() const noexcept { return version_; }
const std::string& ModelPackage::model_type() const noexcept { return model_type_; }
const std::string& ModelPackage::variant() const noexcept { return variant_; }
const std::string& ModelPackage::predicate_order_sha256() const noexcept
{
    return predicate_order_sha256_;
}
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
        error.clear();
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
