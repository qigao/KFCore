#include "kfcore/pipelines/scene_graph.hpp"
#include "kfcore/relation/open_vocabulary_relation.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using kfcore::relation::PredicateVocabulary;
using kfcore::relation::RelationEdge;

struct VocabularyCase
{
    std::string label;
    PredicateVocabulary vocabulary;
};

struct ReferenceEdge
{
    std::size_t subject = 0U;
    std::size_t object = 0U;
    std::size_t predicate = 0U;
    double score = 0.0;
};

std::vector<std::string> split_tabs(const std::string& line)
{
    std::vector<std::string> fields;
    std::size_t begin = 0U;
    for (;;)
    {
        const std::size_t end = line.find('\t', begin);
        fields.push_back(line.substr(
            begin,
            end == std::string::npos
                ? std::string::npos
                : end - begin));
        if (end == std::string::npos)
        {
            break;
        }
        begin = end + 1U;
    }
    return fields;
}

std::vector<VocabularyCase>
load_vocabularies(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error(
            "cannot open vocabulary fixture: " +
            path.string());
    }

    std::vector<VocabularyCase> result;
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            continue;
        }
        const auto fields = split_tabs(line);
        if (fields.size() != 7U)
        {
            throw std::runtime_error(
                "vocabulary fixture row must have seven fields");
        }

        auto match = std::find_if(
            result.begin(),
            result.end(),
            [&](const VocabularyCase& value)
            {
                return value.label == fields[0];
            });
        if (match == result.end())
        {
            VocabularyCase item;
            item.label = fields[0];
            item.vocabulary.embedding_dim = 4U;
            result.push_back(std::move(item));
            match = std::prev(result.end());
        }

        match->vocabulary.predicates.push_back(fields[1]);
        for (std::size_t index = 2U; index < 6U; ++index)
        {
            match->vocabulary.embeddings.push_back(
                std::stof(fields[index]));
        }
        match->vocabulary.spatial_weights.push_back(
            std::stof(fields[6]));
    }

    if (result.size() != 4U)
    {
        throw std::runtime_error(
            "qualification requires four vocabulary cases");
    }
    const std::array<std::string, 4U> expected {
        "v1", "v3", "default", "v1-repeat"
    };
    for (std::size_t index = 0U;
         index < result.size(); ++index)
    {
        if (result[index].label != expected[index])
        {
            throw std::runtime_error(
                "unexpected vocabulary case order");
        }
    }
    return result;
}

std::map<std::string, std::vector<ReferenceEdge>>
load_reference(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error(
            "cannot open ORT reference fixture: " +
            path.string());
    }

    std::map<std::string, std::vector<ReferenceEdge>> result;
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            continue;
        }
        const auto fields = split_tabs(line);
        if (fields.size() != 5U)
        {
            throw std::runtime_error(
                "reference row must have five fields");
        }
        ReferenceEdge edge;
        edge.subject =
            static_cast<std::size_t>(
                std::stoull(fields[1]));
        edge.object =
            static_cast<std::size_t>(
                std::stoull(fields[2]));
        edge.predicate =
            static_cast<std::size_t>(
                std::stoull(fields[3]));
        edge.score = std::stod(fields[4]);
        result[fields[0]].push_back(edge);
    }
    return result;
}

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void compare_edges(
    const std::vector<RelationEdge>& actual,
    const std::vector<ReferenceEdge>& expected)
{
    require(
        actual.size() == expected.size(),
        "C++ edge count differs from Python/ORT reference");
    for (std::size_t index = 0U;
         index < actual.size(); ++index)
    {
        require(
            actual[index].subject_index ==
                expected[index].subject,
            "subject index differs from Python/ORT reference");
        require(
            actual[index].object_index ==
                expected[index].object,
            "object index differs from Python/ORT reference");
        require(
            actual[index].predicate_index ==
                expected[index].predicate,
            "predicate index differs from Python/ORT reference");
        require(
            std::fabs(
                static_cast<double>(
                    actual[index].score) -
                expected[index].score) <= 1.0e-6,
            "relation score differs from Python/ORT reference");
        require(
            !actual[index].subject_track_id.has_value() &&
                !actual[index].object_track_id.has_value(),
            "GT-box relation edge unexpectedly acquired a track id");
    }
}

void compare_relation_edges(
    const std::vector<RelationEdge>& actual,
    const std::vector<RelationEdge>& expected)
{
    require(
        actual.size() == expected.size(),
        "backend/host edge count differs");
    for (std::size_t index = 0U;
         index < actual.size(); ++index)
    {
        require(
            actual[index].subject_index ==
                    expected[index].subject_index &&
                actual[index].object_index ==
                    expected[index].object_index &&
                actual[index].predicate_index ==
                    expected[index].predicate_index,
            "backend/host relation key differs");
        require(
            std::fabs(
                actual[index].score -
                expected[index].score) <= 1.0e-6F,
            "backend/host relation score differs");
    }
}

std::string json_escape(const std::string& value)
{
    std::ostringstream stream;
    for (const unsigned char ch : value)
    {
        switch (ch)
        {
        case '"': stream << "\\\""; break;
        case '\\': stream << "\\\\"; break;
        case '\n': stream << "\\n"; break;
        case '\r': stream << "\\r"; break;
        case '\t': stream << "\\t"; break;
        default:
            if (ch < 0x20U)
            {
                stream << "\\u"
                       << std::hex
                       << std::setw(4)
                       << std::setfill('0')
                       << static_cast<unsigned int>(ch)
                       << std::dec;
            }
            else
            {
                stream << static_cast<char>(ch);
            }
            break;
        }
    }
    return stream.str();
}

void write_case_array(
    std::ofstream& stream,
    const char* key,
    const std::vector<VocabularyCase>& vocabularies,
    const std::vector<kfcore::relation::TimedRelationFrame>& runs,
    bool backend_scoring)
{
    stream << "  \"" << key << "\": [\n";
    for (std::size_t index = 0U;
         index < runs.size(); ++index)
    {
        const auto& value = runs[index];
        stream << "    {\n";
        stream << "      \"label\": \""
               << json_escape(vocabularies[index].label)
               << "\",\n";
        stream << "      \"predicate_count\": "
               << value.timing.predicate_count << ",\n";
        stream << "      \"vocabulary_version\": "
               << value.frame.vocabulary_version << ",\n";
        stream << "      \"region_count\": "
               << value.timing.region_count << ",\n";
        stream << "      \"valid_pair_count\": "
               << value.timing.valid_pair_count << ",\n";
        stream << "      \"edge_count\": "
               << value.timing.edge_count << ",\n";
        stream << "      \"preprocess_ms\": "
               << value.timing.preprocess_ms << ",\n";
        stream << "      \"backend_ms\": "
               << value.timing.backend_ms << ",\n";
        stream << "      \"predicate_score_ms\": "
               << value.timing.predicate_score_ms << ",\n";
        stream << "      \"backbone_context_ms\": ";
        if (backend_scoring)
        {
            stream << "null,\n";
        }
        else
        {
            stream << value.timing.backend_ms << ",\n";
        }
        stream << "      \"predicate_scoring_ms\": ";
        if (backend_scoring)
        {
            stream << "null,\n";
        }
        else
        {
            stream << value.timing.predicate_score_ms << ",\n";
        }
        stream << "      \"predicate_scoring_in_backend\": "
               << (backend_scoring ? "true" : "false")
               << ",\n";
        stream << "      \"runtime_ms\": "
               << value.timing.runtime_ms << ",\n";
        stream << "      \"decode_ms\": "
               << value.timing.decode_ms << ",\n";
        stream << "      \"total_ms\": "
               << value.timing.total_ms << ",\n";
        stream << "      \"pair_keys\": [";
        for (std::size_t edge_index = 0U;
             edge_index < value.frame.edges.size();
             ++edge_index)
        {
            if (edge_index != 0U)
            {
                stream << ", ";
            }
            const auto& edge = value.frame.edges[edge_index];
            stream << "[" << edge.subject_index
                   << ", " << edge.object_index << "]";
        }
        stream << "]\n";
        stream << "    }"
               << (index + 1U == runs.size()
                       ? "\n"
                       : ",\n");
    }
    stream << "  ],\n";
}

void write_report(
    const std::filesystem::path& path,
    const std::string& model_sha,
    const std::string& package_sha,
    const std::string& host_model_sha,
    const std::string& host_package_sha,
    const std::string& vocabulary_sha,
    const std::string& reference_sha,
    const std::string& plugin_sha,
    const std::vector<VocabularyCase>& vocabularies,
    const std::vector<kfcore::relation::TimedRelationFrame>& runs,
    const std::vector<kfcore::relation::TimedRelationFrame>& host_runs,
    const kfcore::runtime::ExecutionRoute& route)
{
    if (path.empty())
    {
        throw std::runtime_error(
            "qualification report path must not be empty");
    }
    if (std::filesystem::exists(path))
    {
        throw std::runtime_error(
            "qualification report path already exists");
    }

    std::ofstream stream(path);
    if (!stream)
    {
        throw std::runtime_error(
            "cannot create qualification report");
    }

    stream << std::fixed << std::setprecision(6);
    stream << "{\n";
    stream << "  \"schema\": \"kfcore.relation-ort-cpu-cpp-qualification/1\",\n";
    stream << "  \"passed\": true,\n";
    stream << "  \"runtime_language\": \"C++17\",\n";
    stream << "  \"backend_scoring\": true,\n";
    stream << "  \"host_fallback_reference\": true,\n";
    stream << "  \"backend_host_pair_keyed_parity\": true,\n";
    stream << "  \"provider\": \""
           << json_escape(route.backend_id)
           << "\",\n";
    stream << "  \"device\": \""
           << json_escape(route.device_id)
           << "\",\n";
    stream << "  \"image_size\": 8,\n";
    stream << "  \"vocabulary_source\": \"precomputed\",\n";
    stream << "  \"text_encoder_sha256\": null,\n";
    stream << "  \"tokenizer_sha256\": null,\n";
    stream << "  \"model_sha256\": \""
           << model_sha << "\",\n";
    stream << "  \"package_sha256\": \""
           << package_sha << "\",\n";
    stream << "  \"host_model_sha256\": \""
           << host_model_sha << "\",\n";
    stream << "  \"host_package_sha256\": \""
           << host_package_sha << "\",\n";
    stream << "  \"vocabulary_fixture_sha256\": \""
           << vocabulary_sha << "\",\n";
    stream << "  \"python_ort_reference_sha256\": \""
           << reference_sha << "\",\n";
    stream << "  \"backend_plugin_sha256\": \""
           << plugin_sha << "\",\n";
    stream << "  \"model_load_count\": 1,\n";
    stream << "  \"host_model_load_count\": 1,\n";
    stream << "  \"same_model_reused_across_vocabularies\": true,\n";
    stream << "  \"same_host_model_reused_across_vocabularies\": true,\n";
    stream << "  \"object_labels_enter_relation_inference\": false,\n";

    write_case_array(
        stream, "cases", vocabularies, runs, true);
    write_case_array(
        stream, "host_cases", vocabularies, host_runs, false);

    stream << "  \"vocabulary_version_sequence\": [";
    for (std::size_t index = 0U;
         index < runs.size(); ++index)
    {
        if (index != 0U)
        {
            stream << ", ";
        }
        stream << runs[index].frame.vocabulary_version;
    }
    stream << "],\n";
    stream << "  \"relation_frame_scene_graph_composition\": true,\n";
    stream << "  \"python_ort_edge_parity\": true\n";
    stream << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 6)
        {
            std::cerr
                << "usage: relation_ort_cpu_qualification "
                << "<plugin> <package-dir> <vocab.tsv> "
                << "<reference.tsv> <report.json>\n";
            return 2;
        }

        const std::filesystem::path plugin_path(argv[1]);
        const std::filesystem::path package_path(argv[2]);
        const std::filesystem::path vocab_path(argv[3]);
        const std::filesystem::path reference_path(argv[4]);
        const std::filesystem::path report_path(argv[5]);

        const auto vocabularies =
            load_vocabularies(vocab_path);
        const auto references =
            load_reference(reference_path);

        kfcore::runtime::Runtime runtime;
        const auto backend =
            runtime.load_backend(plugin_path);
        require(
            backend->id() == "onnxruntime",
            "qualification loaded the wrong backend");

        const auto package =
            kfcore::runtime::ModelPackage::load(
                package_path);
        require(
            package.model_type() ==
                kfcore::relation::
                    kDynamicOpenVocabularyRelationModelType,
            "fixture is not relation.open-vocabulary");

        const auto host_package_path =
            package_path / "host";
        const auto host_package =
            kfcore::runtime::ModelPackage::load(
                host_package_path);
        require(
            host_package.model_type() ==
                kfcore::relation::
                    kOpenVocabularyRelationModelType,
            "host fixture is not relation.open-vocabulary-encoder");

        auto options =
            kfcore::relation::
                OpenVocabularyRelationOptions {};
        options.input_size = 8;
        options.max_boxes = 3U;
        options.max_pairs = 4U;
        options.query_dim = 4U;
        options.logit_scale = 1.0F;
        options.logit_bias = 0.0F;
        options.threshold = 0.0F;
        options.top_k = 4U;
        options.weight_ranking_by_detector_score = false;

        auto relation =
            kfcore::relation::OpenVocabularyRelation::load(
                runtime,
                package,
                kfcore::runtime::ExecutionPolicy::exact(
                    "onnxruntime",
                    "cpu"),
                options);

        auto host_relation =
            kfcore::relation::OpenVocabularyRelation::load(
                runtime,
                host_package,
                kfcore::runtime::ExecutionPolicy::exact(
                    "onnxruntime",
                    "cpu"),
                options);

        require(
            relation->backend_scoring(),
            "qualification did not select backend scoring");
        require(
            !host_relation->backend_scoring(),
            "qualification did not select host query/scorer fallback");
        require(
            relation->execution_route().backend_id ==
                    "onnxruntime" &&
                relation->execution_route().device_id ==
                    "cpu",
            "qualification route is not ORT CPU");
        require(
            host_relation->execution_route().backend_id ==
                    "onnxruntime" &&
                host_relation->execution_route().device_id ==
                    "cpu",
            "host qualification route is not ORT CPU");

        std::array<std::uint8_t, 8U * 8U * 3U> pixels {};
        const kfcore::image::ImageView image {
            pixels.data(),
            pixels.size(),
            8,
            8,
            8U * 3U,
            kfcore::image::PixelFormat::Bgr8,
            kfcore::image::MemoryKind::Host,
        };
        const std::vector<kfcore::relation::Region> regions {
            {1.0F, 1.0F, 3.0F, 3.0F, 1.0F, std::nullopt},
            {5.0F, 1.0F, 7.0F, 3.0F, 1.0F, std::nullopt},
            {3.0F, 5.0F, 5.0F, 7.0F, 1.0F, std::nullopt},
        };

        std::vector<
            kfcore::relation::TimedRelationFrame>
            runs;
        std::vector<
            kfcore::relation::TimedRelationFrame>
            host_runs;
        runs.reserve(vocabularies.size());
        host_runs.reserve(vocabularies.size());

        for (std::size_t index = 0U;
             index < vocabularies.size(); ++index)
        {
            relation->set_vocabulary(
                vocabularies[index].vocabulary);
            host_relation->set_vocabulary(
                vocabularies[index].vocabulary);
            require(
                relation->predicate_count() ==
                    vocabularies[index]
                        .vocabulary.predicates.size(),
                "runtime predicate count differs from vocabulary");
            require(
                relation->vocabulary_version() ==
                    index + 1U,
                "vocabulary version is not positive/monotonic");
            require(
                host_relation->vocabulary_version() ==
                    index + 1U,
                "host vocabulary version is not positive/monotonic");

            auto timed =
                relation->infer_timed(image, regions);
            auto host_timed =
                host_relation->infer_timed(image, regions);
            require(
                timed.frame.vocabulary_version ==
                    relation->vocabulary_version(),
                "RelationFrame vocabulary version drifted");
            require(
                timed.timing.region_count ==
                    regions.size(),
                "timing region count drifted");
            require(
                timed.timing.predicate_count ==
                    relation->predicate_count(),
                "timing predicate count drifted");
            require(
                timed.timing.preprocess_ms >= 0.0 &&
                    timed.timing.backend_ms >= 0.0 &&
                    timed.timing.predicate_score_ms >= 0.0 &&
                    timed.timing.runtime_ms >= 0.0 &&
                    timed.timing.decode_ms >= 0.0 &&
                    timed.timing.total_ms >= 0.0,
                "negative backend-scoring timing value");
            require(
                host_timed.timing.preprocess_ms >= 0.0 &&
                    host_timed.timing.backend_ms >= 0.0 &&
                    host_timed.timing.predicate_score_ms >= 0.0 &&
                    host_timed.timing.runtime_ms >= 0.0 &&
                    host_timed.timing.decode_ms >= 0.0 &&
                    host_timed.timing.total_ms >= 0.0,
                "negative host-scoring timing value");
            require(
                std::fabs(
                    timed.timing.runtime_ms -
                    (timed.timing.backend_ms +
                     timed.timing.predicate_score_ms)) <= 1.0e-3,
                "backend timing split does not sum to runtime");
            require(
                std::fabs(
                    host_timed.timing.runtime_ms -
                    (host_timed.timing.backend_ms +
                     host_timed.timing.predicate_score_ms)) <= 1.0e-3,
                "host timing split does not sum to runtime");
            require(
                timed.timing.edge_count ==
                    timed.frame.edges.size(),
                "timing edge count drifted");

            const auto reference =
                references.find(
                    vocabularies[index].label);
            require(
                reference != references.end(),
                "missing Python/ORT reference case");
            compare_edges(
                timed.frame.edges,
                reference->second);
            compare_edges(
                host_timed.frame.edges,
                reference->second);
            compare_relation_edges(
                timed.frame.edges,
                host_timed.frame.edges);

            kfcore::pipelines::SceneGraphFrame scene;
            scene.objects.image_width = 8;
            scene.objects.image_height = 8;
            scene.objects.detections = {
                {{{1.0F, 1.0F, 3.0F, 3.0F}, 1.0F, 1001}, std::nullopt},
                {{{5.0F, 1.0F, 7.0F, 3.0F}, 1.0F, 1002}, std::nullopt},
                {{{3.0F, 5.0F, 5.0F, 7.0F}, 1.0F, 1003}, std::nullopt},
            };
            scene.relations = timed.frame;
            require(
                scene.relations.vocabulary_version ==
                    timed.frame.vocabulary_version,
                "SceneGraphFrame lost vocabulary version");
            for (const auto& edge :
                 scene.relations.edges)
            {
                require(
                    edge.subject_index <
                            scene.objects.detections.size() &&
                        edge.object_index <
                            scene.objects.detections.size(),
                    "SceneGraphFrame relation/object index drift");
                require(
                    !edge.subject_track_id.has_value() &&
                        !edge.object_track_id.has_value(),
                    "GT-box SceneGraphFrame unexpectedly has track ids");
            }

            runs.push_back(std::move(timed));
            host_runs.push_back(std::move(host_timed));
        }

        require(
            runs.front().frame.edges.size() ==
                runs.back().frame.edges.size(),
            "repeated V=1 changed edge count");
        for (std::size_t index = 0U;
             index < runs.front().frame.edges.size();
             ++index)
        {
            const auto& first =
                runs.front().frame.edges[index];
            const auto& repeated =
                runs.back().frame.edges[index];
            require(
                first.subject_index ==
                        repeated.subject_index &&
                    first.object_index ==
                        repeated.object_index &&
                    first.predicate_index ==
                        repeated.predicate_index &&
                    std::fabs(
                        first.score -
                        repeated.score) <= 1.0e-6F,
                "repeated V=1 changed relation semantics");
        }

        const auto& artifact =
            package.artifact("ort-cpu");
        const std::string model_sha =
            kfcore::runtime::
                compute_model_artifact_sha256(
                    package.artifact_path(artifact));
        require(
            model_sha == artifact.sha256,
            "executed ONNX hash differs from package manifest");

        write_report(
            report_path,
            model_sha,
            kfcore::runtime::
                compute_model_artifact_sha256(
                    package_path / "model.json"),
            kfcore::runtime::
                compute_model_artifact_sha256(
                    vocab_path),
            kfcore::runtime::
                compute_model_artifact_sha256(
                    reference_path),
            kfcore::runtime::
                compute_model_artifact_sha256(
                    plugin_path),
            vocabularies,
            runs,
            relation->execution_route());

        std::cout
            << "ORT CPU C++ relation qualification: PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "ORT CPU C++ relation qualification: FAIL: "
            << error.what() << '\n';
        return 1;
    }
}
