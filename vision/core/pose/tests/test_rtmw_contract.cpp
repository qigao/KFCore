#include "kfcore/pose/error.hpp"
#include "kfcore/pose/rtmw.hpp"
#include "kfcore/pose/schema.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/runtime.hpp"
#include "tinytest.hpp"

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

runtime::ModelPackage write_package(const std::string& semantic_contract,
                                    const std::string& semantic_version)
{
    TempPackage package;
    const std::filesystem::path root(package.directory);
    const std::filesystem::path artifact = root / "rtmw.onnx";
    const std::filesystem::path manifest = root / "model.json";

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

pose::PoseErrorCode load_error_code(const runtime::ModelPackage& package)
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
        return error.code();
    }
    check(false);
    return pose::PoseErrorCode::InvalidArgument;
}

} // namespace

spec("RTMW semantic model contract")
{
    it("rejects a missing semantic contract before backend resolution")
    {
        const auto package = write_package("", "");
        check(load_error_code(package) ==
              pose::PoseErrorCode::ModelContractMismatch);
    }

    it("rejects an unknown semantic contract before backend resolution")
    {
        const auto package = write_package("pose.unknown", "1");
        check(load_error_code(package) ==
              pose::PoseErrorCode::ModelContractMismatch);
    }

    it("accepts WholeBody133 semantics and then reaches backend resolution")
    {
        const auto package = write_package(
            std::string(pose::kCocoWholeBody133SemanticContract),
            std::string(pose::kCocoWholeBody133SemanticVersion));
        check(load_error_code(package) ==
              pose::PoseErrorCode::RuntimeFailure);
    }
}
