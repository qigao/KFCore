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
    it("computes the lowercase SHA-256 used by package manifests")
    {
        TempPackage package;
        const std::filesystem::path artifact =
            std::filesystem::path(package.directory) / "payload.bin";
        static constexpr char kPayload[] = "abc";
        check_true(tt_write_file(artifact.string().c_str(), kPayload,
                                 sizeof(kPayload) - 1U) == 0);

        check_true(kfcore::runtime::compute_model_artifact_sha256(artifact) ==
                   "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

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

    it("accepts the raw-yolox detector flavor")
    {
        TempPackage package;
        const std::filesystem::path root(package.directory);
        const std::filesystem::path artifact = root / "yolox_tiny.onnx";
        const std::filesystem::path manifest = root / "model.json";
        static constexpr char kManifest[] =
            "{\"schema\":\"kfcore.model/1\",\"id\":\"yolox-tiny\","
            "\"version\":\"0.1.1rc0\",\"model_type\":\"yolo-detection\","
            "\"artifacts\":[{\"id\":\"ort-cpu\",\"format\":\"onnx\","
            "\"path\":\"yolox_tiny.onnx\",\"flavor\":\"raw-yolox\","
            "\"sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\","
            "\"backend\":\"onnxruntime\",\"device\":\"cpu\"}]}";

        check_true(tt_write_file(artifact.string().c_str(), "", 0U) == 0);
        check_true(tt_write_file(manifest.string().c_str(), kManifest,
                                 sizeof(kManifest) - 1U) == 0);

        const auto loaded = kfcore::runtime::ModelPackage::load(root);
        check_true(loaded.model_type() == "yolo-detection");
        check_true(loaded.artifacts().front().flavor == "raw-yolox");
        check_nothrow(kfcore::runtime::verify_model_artifact(
            loaded, loaded.artifacts().front()));
    }

    it("loads a flat manifest and sibling ONNX artifact")
    {
        TempPackage package;
        const std::filesystem::path root(package.directory);
        const std::filesystem::path artifact = root / "2dfan4.onnx";
        const std::filesystem::path manifest = root / "2dfan4.json";
        static constexpr char kManifest[] =
            "{\"schema\":\"kfcore.model/1\",\"id\":\"face68\","
            "\"version\":\"1.0.0\",\"model_type\":\"face.face68\","
            "\"artifacts\":[{\"id\":\"runtime\",\"format\":\"onnx\","
            "\"path\":\"2dfan4.onnx\","
            "\"sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\","
            "\"backend\":\"onnxruntime\",\"device\":\"any\"}]}";

        check_true(tt_write_file(artifact.string().c_str(), "", 0U) == 0);
        check_true(tt_write_file(manifest.string().c_str(), kManifest,
                                 sizeof(kManifest) - 1U) == 0);

        const auto loaded = kfcore::runtime::ModelPackage::load(manifest);
        check_true(loaded.root() == std::filesystem::weakly_canonical(root));
        check_true(loaded.id() == "face68");
        check_true(loaded.model_type() == "face.face68");
        check_true(loaded.artifact_path(loaded.artifacts().front()) == artifact);
    }

    it("loads paired semantic contract metadata")
    {
        TempPackage package;
        const std::filesystem::path root(package.directory);
        const std::filesystem::path artifact = root / "rtmw.onnx";
        const std::filesystem::path manifest = root / "model.json";
        static constexpr char kManifest[] =
            "{\"schema\":\"kfcore.model/1\",\"id\":\"rtmw\","
            "\"version\":\"1\",\"model_type\":\"pose.rtmw\","
            "\"semantic_contract\":\"pose.coco-wholebody-133\","
            "\"semantic_version\":\"1\","
            "\"artifacts\":[{\"id\":\"ort-cpu\",\"format\":\"onnx\","
            "\"path\":\"rtmw.onnx\","
            "\"sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\","
            "\"backend\":\"onnxruntime\",\"device\":\"cpu\"}]}";

        check_true(tt_write_file(artifact.string().c_str(), "", 0U) == 0);
        check_true(tt_write_file(manifest.string().c_str(), kManifest,
                                 sizeof(kManifest) - 1U) == 0);

        const auto loaded = kfcore::runtime::ModelPackage::load(root);
        check_true(loaded.semantic_contract() == "pose.coco-wholebody-133");
        check_true(loaded.semantic_version() == "1");
    }

    it("keeps semantic identity absent for legacy non-semantic packages")
    {
        TempPackage package;
        const std::filesystem::path root(package.directory);
        const std::filesystem::path artifact = root / "model.onnx";
        const std::filesystem::path manifest = root / "model.json";
        static constexpr char kManifest[] =
            "{\"schema\":\"kfcore.model/1\",\"id\":\"legacy\","
            "\"version\":\"1\",\"model_type\":\"gesture.temporal-gru\","
            "\"artifacts\":[{\"id\":\"ort-cpu\",\"format\":\"onnx\","
            "\"path\":\"model.onnx\","
            "\"sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\","
            "\"backend\":\"onnxruntime\",\"device\":\"cpu\"}]}";

        check_true(tt_write_file(artifact.string().c_str(), "", 0U) == 0);
        check_true(tt_write_file(manifest.string().c_str(), kManifest,
                                 sizeof(kManifest) - 1U) == 0);

        const auto loaded = kfcore::runtime::ModelPackage::load(root);
        check_true(loaded.semantic_contract().empty());
        check_true(loaded.semantic_version().empty());
    }

    it("rejects explicitly empty semantic contract metadata")
    {
        TempPackage package;
        const std::filesystem::path root(package.directory);
        const std::filesystem::path artifact = root / "rtmw.onnx";
        const std::filesystem::path manifest = root / "model.json";
        static constexpr char kManifest[] =
            "{\"schema\":\"kfcore.model/1\",\"id\":\"rtmw\","
            "\"version\":\"1\",\"model_type\":\"pose.rtmw\","
            "\"semantic_contract\":\"\",\"semantic_version\":\"1\","
            "\"artifacts\":[{\"id\":\"ort-cpu\",\"format\":\"onnx\","
            "\"path\":\"rtmw.onnx\","
            "\"sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\","
            "\"backend\":\"onnxruntime\",\"device\":\"cpu\"}]}";

        check_true(tt_write_file(artifact.string().c_str(), "", 0U) == 0);
        check_true(tt_write_file(manifest.string().c_str(), kManifest,
                                 sizeof(kManifest) - 1U) == 0);

        check_throws_as(
            kfcore::runtime::ModelPackage::load(root),
            kfcore::runtime::RuntimeError);
    }

    it("rejects unpaired semantic contract metadata")
    {
        TempPackage package;
        const std::filesystem::path root(package.directory);
        const std::filesystem::path artifact = root / "rtmw.onnx";
        const std::filesystem::path manifest = root / "model.json";
        static constexpr char kManifest[] =
            "{\"schema\":\"kfcore.model/1\",\"id\":\"rtmw\","
            "\"version\":\"1\",\"model_type\":\"pose.rtmw\","
            "\"semantic_contract\":\"pose.coco-wholebody-133\","
            "\"artifacts\":[{\"id\":\"ort-cpu\",\"format\":\"onnx\","
            "\"path\":\"rtmw.onnx\","
            "\"sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\","
            "\"backend\":\"onnxruntime\",\"device\":\"cpu\"}]}";

        check_true(tt_write_file(artifact.string().c_str(), "", 0U) == 0);
        check_true(tt_write_file(manifest.string().c_str(), kManifest,
                                 sizeof(kManifest) - 1U) == 0);

        check_throws_as(
            kfcore::runtime::ModelPackage::load(root),
            kfcore::runtime::RuntimeError);
    }
}
