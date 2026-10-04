#include "kfcore/pose/error.hpp"
#include "kfcore/pose/rtmw.hpp"
#include "kfcore/pose/schema.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/runtime.hpp"
#include "tinytest.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace kfcore;

namespace
{

struct TempPackage final
{
    TempPackage()
        : directory(tt_make_temp_dir("kfcore-rtmw-contract"))
    {
        check_not_null(directory);
    }

    ~TempPackage()
    {
        if (directory != nullptr)
        {
            (void)tt_remove_tree(directory);
            std::free(directory);
        }
    }

    TempPackage(const TempPackage&) = delete;
    TempPackage& operator=(const TempPackage&) = delete;

    char* directory = nullptr;
};

runtime::ModelPackage write_package(
    TempPackage& package,
    const std::string& semantic_contract,
    const std::string& semantic_version,
    const std::string& semantic_config_json = {})
{
    const std::filesystem::path root(package.directory);
    const std::filesystem::path artifact = root / "rtmw.onnx";
    const std::filesystem::path manifest = root / "model.json";
    const std::filesystem::path semantic_config = root / "pose.json";

    std::string semantic_config_sha;
    if (!semantic_config_json.empty())
    {
        check_true(tt_write_file(
                       semantic_config.string().c_str(),
                       semantic_config_json.data(),
                       semantic_config_json.size()) == 0);
        semantic_config_sha =
            runtime::compute_model_artifact_sha256(semantic_config);
    }

    std::string json =
        "{\"schema\":\"kfcore.model/1\",\"id\":\"rtmw\","
        "\"version\":\"1\",\"model_type\":\"pose.rtmw\"";
    if (!semantic_contract.empty())
    {
        json += ",\"semantic_contract\":\"" + semantic_contract + "\"";
    }
    if (!semantic_version.empty())
    {
        json += ",\"semantic_version\":\"" + semantic_version + "\"";
    }
    if (!semantic_config_json.empty())
    {
        json +=
            ",\"semantic_config\":\"pose.json\","
            "\"semantic_config_sha256\":\"" +
            semantic_config_sha + "\"";
    }
    json +=
        ",\"artifacts\":[{\"id\":\"ort-cpu\",\"format\":\"onnx\","
        "\"path\":\"rtmw.onnx\","
        "\"sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\","
        "\"backend\":\"onnxruntime\",\"device\":\"cpu\"}]}";

    check_true(tt_write_file(artifact.string().c_str(), "", 0U) == 0);
    check_true(tt_write_file(
                   manifest.string().c_str(), json.data(), json.size()) == 0);

    return runtime::ModelPackage::load(root);
}

struct LoadError
{
    pose::PoseErrorCode code = pose::PoseErrorCode::InvalidArgument;
    std::string message;
};

LoadError load_error(const runtime::ModelPackage& package)
{
    runtime::Runtime runtime;
    try
    {
        (void)pose::Rtmw::load(
            runtime,
            package,
            runtime::ExecutionPolicy::exact("onnxruntime", "cpu"));
    }
    catch (const pose::PoseError& error)
    {
        return {error.code(), error.what()};
    }
    check(false);
    return {};
}

pose::PoseErrorCode load_error_code(const runtime::ModelPackage& package)
{
    return load_error(package).code;
}

} // namespace

spec("RTMW semantic model contract")
{
    it("rejects a missing semantic contract before backend resolution")
    {
        TempPackage temp;
        const auto package = write_package(temp, "", "");
        check(load_error_code(package) ==
              pose::PoseErrorCode::ModelContractMismatch);
    }

    it("rejects an unknown semantic contract before backend resolution")
    {
        TempPackage temp;
        const auto package = write_package(temp, "pose.unknown", "1");
        check(load_error_code(package) ==
              pose::PoseErrorCode::ModelContractMismatch);
    }

    it("accepts WholeBody133 semantics and then reaches backend resolution")
    {
        TempPackage temp;
        const auto package = write_package(
            temp,
            std::string(pose::kCocoWholeBody133SemanticContract),
            std::string(pose::kCocoWholeBody133SemanticVersion));
        check(load_error_code(package) ==
              pose::PoseErrorCode::RuntimeFailure);
    }

    it("accepts an explicit confidence-only SimCC semantic config")
    {
        TempPackage temp;
        const auto package = write_package(
            temp,
            std::string(pose::kCocoWholeBody133SemanticContract),
            std::string(pose::kCocoWholeBody133SemanticVersion),
            "{\"schema\":\"kfcore.pose-semantic/1\","
            "\"codec\":\"simcc\","
            "\"decode_visibility\":false}");
        check(load_error_code(package) ==
              pose::PoseErrorCode::RuntimeFailure);
    }

    it("accepts an explicit SimCC visibility config before backend resolution")
    {
        TempPackage temp;
        const auto package = write_package(
            temp,
            std::string(pose::kCocoWholeBody133SemanticContract),
            std::string(pose::kCocoWholeBody133SemanticVersion),
            "{\"schema\":\"kfcore.pose-semantic/1\","
            "\"codec\":\"simcc\","
            "\"decode_visibility\":true,"
            "\"visibility_beta\":150.0,"
            "\"visibility_sigma_x\":6.0,"
            "\"visibility_sigma_y\":6.93}");
        const auto error = load_error(package);
        if (error.code != pose::PoseErrorCode::RuntimeFailure)
        {
            std::fprintf(stderr,
                         "unexpected visibility RTMW load error: %s\n",
                         error.message.c_str());
        }
        check(error.code == pose::PoseErrorCode::RuntimeFailure);
    }

    it("rejects incomplete visibility semantics before backend resolution")
    {
        TempPackage temp;
        const auto package = write_package(
            temp,
            std::string(pose::kCocoWholeBody133SemanticContract),
            std::string(pose::kCocoWholeBody133SemanticVersion),
            "{\"schema\":\"kfcore.pose-semantic/1\","
            "\"codec\":\"simcc\","
            "\"decode_visibility\":true,"
            "\"visibility_beta\":150.0}");
        check(load_error_code(package) ==
              pose::PoseErrorCode::ModelContractMismatch);
    }

    it("rejects visibility parameters when decoding is disabled")
    {
        TempPackage temp;
        const auto package = write_package(
            temp,
            std::string(pose::kCocoWholeBody133SemanticContract),
            std::string(pose::kCocoWholeBody133SemanticVersion),
            "{\"schema\":\"kfcore.pose-semantic/1\","
            "\"codec\":\"simcc\","
            "\"decode_visibility\":false,"
            "\"visibility_beta\":150.0,"
            "\"visibility_sigma_x\":6.0,"
            "\"visibility_sigma_y\":6.93}");
        check(load_error_code(package) ==
              pose::PoseErrorCode::ModelContractMismatch);
    }

    it("rejects a tampered semantic config before backend resolution")
    {
        TempPackage temp;
        const auto package = write_package(
            temp,
            std::string(pose::kCocoWholeBody133SemanticContract),
            std::string(pose::kCocoWholeBody133SemanticVersion),
            "{\"schema\":\"kfcore.pose-semantic/1\","
            "\"codec\":\"simcc\","
            "\"decode_visibility\":false}");

        const std::filesystem::path config =
            std::filesystem::path(temp.directory) / "pose.json";
        static constexpr char kTampered[] = "{}";
        check_true(tt_write_file(
                       config.string().c_str(),
                       kTampered,
                       sizeof(kTampered) - 1U) == 0);

        check(load_error_code(package) ==
              pose::PoseErrorCode::ModelContractMismatch);
    }
}
