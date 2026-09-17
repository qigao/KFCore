#include "kfcore/runtime/model_package.hpp"

#include "tinytest.hpp"

#include <cstdlib>
#include <filesystem>
#include <string>

namespace
{

struct TempPackage final
{
    TempPackage()
        : directory(tt_make_temp_dir("kfcore-model-package"))
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

} // namespace

spec("runtime model package")
{
    it("loads the public schema field through the internal DataBind schema")
    {
        TempPackage package;
        const std::filesystem::path root(package.directory);
        const std::filesystem::path artifact = root / "model.onnx";
        const std::filesystem::path manifest = root / "model.json";
        static constexpr char kManifest[] =
            "{\"schema\":\"kfcore.model/1\",\"id\":\"test-package\","
            "\"version\":\"1.0.0\",\"model_type\":\"gesture.temporal-gru\","
            "\"artifacts\":[{\"id\":\"onnx-cpu\",\"format\":\"onnx\","
            "\"path\":\"model.onnx\",\"flavor\":\"causal-gru-v1\","
            "\"sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\","
            "\"backend\":\"onnxruntime\",\"device\":\"cpu\"}]}";

        check_true(tt_write_file(artifact.string().c_str(), "", 0U) == 0);
        check_true(tt_write_file(manifest.string().c_str(), kManifest,
                                 sizeof(kManifest) - 1U) == 0);

        const auto loaded = kfcore::runtime::ModelPackage::load(root);
        check_true(loaded.id() == "test-package");
        check_true(loaded.model_type() == "gesture.temporal-gru");
        check_true(loaded.artifacts().size() == 1U);
        check_nothrow(kfcore::runtime::verify_model_artifact(
            loaded, loaded.artifacts().front()));
    }
}
