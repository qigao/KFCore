#include "kfcore/relation/open_vocabulary_relation.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

kfcore::relation::PredicateVocabulary vocabulary(
    const std::vector<std::string>& names)
{
    kfcore::relation::PredicateVocabulary value;
    value.predicates = names;
    value.embedding_dim = 512U;
    value.embeddings.assign(names.size() * value.embedding_dim, 0.0F);
    value.spatial_weights.assign(names.size(), 0.0F);
    for (std::size_t row = 0U; row < names.size(); ++row)
    {
        value.embeddings[row * value.embedding_dim + row] = 1.0F;
        if (row == 1U)
        {
            value.spatial_weights[row] = 0.5F;
        }
        else if (row == 2U)
        {
            value.spatial_weights[row] = 1.0F;
        }
    }
    return value;
}

void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

bool valid_backend_id(const std::string& value)
{
    return value == "onnxruntime" || value == "tensorrt";
}

bool valid_device_id(const std::string& value)
{
    if (value == "cpu" || value == "cuda")
    {
        return true;
    }
    if (value.rfind("cuda:", 0U) != 0U || value.size() <= 5U)
    {
        return false;
    }
    for (std::size_t index = 5U; index < value.size(); ++index)
    {
        if (value[index] < '0' || value[index] > '9')
        {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 4 && argc != 6)
        {
            std::cerr
                << "usage: released_relsgg_qualification "
                << "<plugin> <package-dir> <report.json> "
                   "[<backend-id> <device-id>]\n";
            return 2;
        }

        const std::filesystem::path plugin_path(argv[1]);
        const std::filesystem::path package_path(argv[2]);
        const std::filesystem::path report_path(argv[3]);
        const std::string backend_id =
            argc == 6 ? std::string(argv[4]) : "onnxruntime";
        const std::string device_id =
            argc == 6 ? std::string(argv[5]) : "cpu";
        require(
            valid_backend_id(backend_id),
            "released qualification backend must be onnxruntime or tensorrt");
        require(
            valid_device_id(device_id),
            "released qualification device must be cpu, cuda, or cuda:N");
        if (backend_id == "tensorrt")
        {
            require(
                device_id == "cuda" ||
                    device_id.rfind("cuda:", 0U) == 0U,
                "TensorRT qualification requires an explicit CUDA device");
        }

        kfcore::runtime::Runtime runtime;
        const auto backend = runtime.load_backend(plugin_path);
        require(
            backend->id() == backend_id,
            "released qualification loaded wrong backend");

        const auto package =
            kfcore::runtime::ModelPackage::load(package_path);
        require(
            package.model_type() ==
                kfcore::relation::kDynamicOpenVocabularyRelationModelType,
            "released package is not relation.open-vocabulary");

        kfcore::relation::OpenVocabularyRelationOptions options;
        options.input_size = 448;
        options.max_boxes = 32U;
        options.max_pairs = 128U;
        options.query_dim = 512U;
        options.threshold = 0.0F;
        options.pair_weight = 1.0F;
        options.calibration_a = 0.5651F;
        options.calibration_b = -1.9623F;
        options.top_k = 20U;
        options.weight_ranking_by_detector_score = false;

        auto relation =
            kfcore::relation::OpenVocabularyRelation::load(
                runtime,
                package,
                kfcore::runtime::ExecutionPolicy::exact(
                    backend_id,
                    device_id),
                options);

        require(
            relation->backend_scoring(),
            "released graph did not select backend scoring");
        require(
            relation->execution_route().backend_id == backend_id &&
                relation->execution_route().device_id == device_id,
            "released graph did not resolve to requested backend/device");

        std::vector<std::uint8_t> pixels(
            448U * 448U * 3U,
            0U);
        const kfcore::image::ImageView image {
            pixels.data(),
            pixels.size(),
            448,
            448,
            448U * 3U,
            kfcore::image::PixelFormat::Bgr8,
            kfcore::image::MemoryKind::Host,
        };
        const std::vector<kfcore::relation::Region> regions {
            {40.0F, 40.0F, 180.0F, 280.0F, 1.0F, std::nullopt},
            {220.0F, 80.0F, 400.0F, 360.0F, 1.0F, std::nullopt},
        };

        relation->set_vocabulary(
            vocabulary({"holding"}));
        const auto v1 = relation->infer_timed(image, regions);
        require(
            relation->vocabulary_version() == 1U,
            "released V=1 version mismatch");

        relation->set_vocabulary(
            vocabulary({"holding", "beside", "behind"}));
        const auto v3 = relation->infer_timed(image, regions);
        require(
            relation->vocabulary_version() == 2U,
            "released V=3 version mismatch");
        require(
            v1.timing.predicate_count == 1U &&
                v3.timing.predicate_count == 3U,
            "released predicate counts differ");
        require(
            v1.timing.region_count == 2U &&
                v3.timing.region_count == 2U,
            "released region count differs");
        require(
            v1.timing.total_ms >= 0.0 &&
                v3.timing.total_ms >= 0.0,
            "released timing is negative");

        std::ofstream stream(report_path);
        require(
            static_cast<bool>(stream),
            "cannot create released qualification report");
        stream
            << "{\n"
            << "  \"schema\": "
               "\"kfcore.released-relsgg-runtime-qualification/1\",\n"
            << "  \"passed\": true,\n"
            << "  \"provider\": \"" << backend_id << "\",\n"
            << "  \"device\": \"" << device_id << "\",\n"
            << "  \"backend_scoring\": true,\n"
            << "  \"object_labels_enter_relation_inference\": false,\n"
            << "  \"vocabulary_versions\": [1, 2],\n"
            << "  \"vocabulary_sizes\": [1, 3],\n"
            << "  \"v1_total_ms\": " << v1.timing.total_ms << ",\n"
            << "  \"v3_total_ms\": " << v3.timing.total_ms << "\n"
            << "}\n";

        std::cout
            << "released relsgg " << backend_id << "/" << device_id
            << " qualification: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "released relsgg qualification: FAIL: "
            << error.what() << '\n';
        return 1;
    }
}
