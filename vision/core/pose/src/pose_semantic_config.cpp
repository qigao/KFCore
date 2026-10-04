#include "pose_semantic_config.hpp"

#include "kfcore/pose/error.hpp"
#include "kfcore/runtime/error.hpp"

#include <data_bind.h>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace kfcore::pose::detail
{
namespace
{

constexpr std::size_t kMaxSemanticConfigBytes = 64U * 1024U;

constexpr char kPoseSemanticSchema[] =
    "message PoseSemantic { "
    "bool decode_visibility; "
    "optional double visibility_beta; "
    "optional double visibility_sigma_x; "
    "optional double visibility_sigma_y; "
    "[name(\"schema\")] string schema_id; "
    "string codec; "
    "}";

struct DataBindDeleter
{
    void operator()(DataBind* value) const noexcept
    {
        data_bind_free(value);
    }
};

struct RecordDeleter
{
    void operator()(DataBindRecord* value) const noexcept
    {
        data_bind_record_free(value);
    }
};

[[noreturn]] void throw_contract(std::string detail)
{
    throw PoseError(
        PoseErrorCode::ModelContractMismatch,
        "Pose semantic config: " + std::move(detail));
}

std::string read_config(const std::filesystem::path& path)
{
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    if (error || bytes == 0U || bytes > kMaxSemanticConfigBytes)
    {
        throw_contract(
            "file is missing, empty, or exceeds the 64 KiB limit");
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw_contract("cannot open " + path.u8string());
    }

    std::string text(static_cast<std::size_t>(bytes), '\0');
    stream.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!stream)
    {
        throw_contract("cannot read complete file");
    }
    return text;
}

std::string required_string(const DataBindRecord* record,
                            const char* name)
{
    DataBindStringView value = DATA_BIND_STRING_VIEW_INIT;
    DataBindError error = DATA_BIND_ERROR_INIT;
    if (data_bind_record_get_string(
            record, name, &value, &error) != DATA_BIND_OK ||
        value.data == nullptr || value.length == 0U)
    {
        throw_contract(
            std::string("requires non-empty string '") + name + "'");
    }
    return std::string(value.data, value.length);
}

bool has_field(const DataBindRecord* record, const char* name)
{
    DataBindRecordField field = DATA_BIND_RECORD_FIELD_INIT;
    DataBindError error = DATA_BIND_ERROR_INIT;
    return data_bind_record_find_field(
               record, name, &field, &error) == DATA_BIND_OK;
}

double required_double(const DataBindRecord* record,
                       const char* name)
{
    DataBindRecordField field = DATA_BIND_RECORD_FIELD_INIT;
    DataBindError error = DATA_BIND_ERROR_INIT;
    double value = 0.0;
    if (data_bind_record_find_field(
            record, name, &field, &error) != DATA_BIND_OK ||
        data_bind_record_field_get_double(
            &field, &value, &error) != DATA_BIND_OK)
    {
        throw_contract(
            std::string("requires numeric field '") + name + "'");
    }
    return value;
}

float checked_positive_float(double value, const char* name)
{
    if (!std::isfinite(value) || value <= 0.0 ||
        value > static_cast<double>(
                    (std::numeric_limits<float>::max)()))
    {
        throw_contract(
            std::string("'") + name +
            "' must be finite, positive, and representable as float");
    }
    return static_cast<float>(value);
}

} // namespace

PoseSemanticConfig
load_pose_semantic_config(const runtime::ModelPackage& package)
{
    PoseSemanticConfig result;
    if (!package.has_semantic_config())
    {
        return result;
    }

    try
    {
        runtime::verify_model_semantic_config(package);
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_contract(
            "integrity verification failed: " +
            std::string(error.what()));
    }

    const std::string json =
        read_config(package.semantic_config_path());

    DataBind* raw_codec = nullptr;
    DataBindError error = DATA_BIND_ERROR_INIT;
    if (data_bind_create_from_text(
            kPoseSemanticSchema,
            sizeof(kPoseSemanticSchema) - 1U,
            &raw_codec,
            &error) != DATA_BIND_OK)
    {
        throw_contract(
            std::string("internal schema is invalid: ") +
            error.message);
    }
    std::unique_ptr<DataBind, DataBindDeleter> codec(raw_codec);

    DataBindRecord* raw_record = nullptr;
    if (data_bind_record_from_json(
            codec.get(),
            "PoseSemantic",
            json.data(),
            json.size(),
            &raw_record,
            &error) != DATA_BIND_OK)
    {
        throw_contract(
            std::string("invalid JSON: ") + error.message);
    }
    std::unique_ptr<DataBindRecord, RecordDeleter>
        record(raw_record);

    if (required_string(record.get(), "schema_id") !=
        "kfcore.pose-semantic/1")
    {
        throw_contract(
            "schema must be 'kfcore.pose-semantic/1'");
    }
    if (required_string(record.get(), "codec") != "simcc")
    {
        throw_contract(
            "RTMW semantic config requires codec='simcc'");
    }

    int decode_visibility = 0;
    error = DATA_BIND_ERROR_INIT;
    if (data_bind_record_get_bool(
            record.get(),
            "decode_visibility",
            &decode_visibility,
            &error) != DATA_BIND_OK)
    {
        throw_contract(
            "requires boolean field 'decode_visibility'");
    }
    result.decode_visibility = decode_visibility != 0;

    const bool has_beta =
        has_field(record.get(), "visibility_beta");
    const bool has_sigma_x =
        has_field(record.get(), "visibility_sigma_x");
    const bool has_sigma_y =
        has_field(record.get(), "visibility_sigma_y");

    if (!result.decode_visibility)
    {
        if (has_beta || has_sigma_x || has_sigma_y)
        {
            throw_contract(
                "visibility parameters must be absent when "
                "decode_visibility=false");
        }
        return result;
    }

    if (!(has_beta && has_sigma_x && has_sigma_y))
    {
        throw_contract(
            "decode_visibility=true requires beta and both axis sigma values");
    }

    result.visibility.beta = checked_positive_float(
        required_double(record.get(), "visibility_beta"),
        "visibility_beta");
    result.visibility.sigma_x = checked_positive_float(
        required_double(record.get(), "visibility_sigma_x"),
        "visibility_sigma_x");
    result.visibility.sigma_y = checked_positive_float(
        required_double(record.get(), "visibility_sigma_y"),
        "visibility_sigma_y");
    return result;
}

} // namespace kfcore::pose::detail
