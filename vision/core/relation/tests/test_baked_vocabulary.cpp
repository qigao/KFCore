#include "kfcore/relation/error.hpp"
#include "kfcore/relation/relate_anything.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/resolver.hpp"
#include "kfcore/runtime/runtime.hpp"
#include "tinytest.hpp"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct TempPackage final
{
    TempPackage()
        : directory(tt_make_temp_dir("kfcore-baked-vocabulary"))
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

    char* directory = nullptr;
};

kfcore::runtime::ModelPackage load_package(
    const std::filesystem::path& root, const std::string& predicate_hash)
{
    const std::string manifest =
        "{\"schema\":\"kfcore.model/1\",\"id\":\"fixed-relation\","
        "\"version\":\"1.0.0\",\"model_type\":\"relation.relate-anything\","
        "\"predicate_order_sha256\":\"" + predicate_hash + "\","
        "\"artifacts\":[{\"id\":\"ort-cpu\",\"format\":\"onnx\","
        "\"path\":\"model.onnx\","
        "\"sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\","
        "\"backend\":\"onnxruntime\",\"device\":\"cpu\"}]}";
    const auto path = root / "model.json";
    check(tt_write_file(path.string().c_str(), manifest.data(),
                        manifest.size()) == 0);
    return kfcore::runtime::ModelPackage::load(root);
}

kfcore::relation::RelationErrorCode load_error(
    const kfcore::runtime::ModelPackage& package,
    const std::vector<std::string>& names)
{
    kfcore::runtime::Runtime runtime;
    auto options = kfcore::relation::RelateAnythingOptions{};
    options.predicates = names;
    try
    {
        (void)kfcore::relation::RelateAnything::load(
            runtime, package,
            kfcore::runtime::ExecutionPolicy::exact("onnxruntime", "cpu"),
            options);
    }
    catch (const kfcore::relation::RelationError& error)
    {
        return error.code();
    }
    throw std::runtime_error("expected relation load to fail without a backend");
}

} // namespace

spec("baked relation vocabulary contract")
{
    it("rejects a reordered vocabulary before loading the backend")
    {
        TempPackage temp;
        const auto package = load_package(
            temp.directory,
            "132cf909c75fd7c994666eeb63109cea4cfdea9f4be050a4ef182b56b1dbec39");
        check(package.predicate_order_sha256() ==
              "132cf909c75fd7c994666eeb63109cea4cfdea9f4be050a4ef182b56b1dbec39");
        check(load_error(package, {"beside", "on"}) ==
              kfcore::relation::RelationErrorCode::ModelContractMismatch);
        check(load_error(package, {"on", "beside"}) ==
              kfcore::relation::RelationErrorCode::RuntimeFailure);
    }

    it("rejects a fixed-vocabulary package without the required binding")
    {
        TempPackage temp;
        const auto package = load_package(temp.directory, "");
        check(load_error(package, {"on", "beside"}) ==
              kfcore::relation::RelationErrorCode::ModelContractMismatch);
    }
}
