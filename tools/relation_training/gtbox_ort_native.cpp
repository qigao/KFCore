#include "kfcore/pipelines/scene_graph.hpp"
#include "kfcore/relation/open_vocabulary_relation.hpp"
#include "kfcore/runtime/plugin.hpp"
#include "kfcore/runtime/resolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using Config = std::unordered_map<std::string, std::string>;

std::string read_text(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error(
            "cannot open text file: " + path.string());
    }
    std::ostringstream output;
    output << stream.rdbuf();
    if (!stream.good() && !stream.eof())
    {
        throw std::runtime_error(
            "failed while reading text file: " + path.string());
    }
    return output.str();
}

std::vector<std::uint8_t>
read_bytes(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error(
            "cannot open binary file: " + path.string());
    }
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (end < 0)
    {
        throw std::runtime_error(
            "cannot determine binary file size: " + path.string());
    }
    const auto size = static_cast<std::size_t>(end);
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> result(size);
    if (size != 0U)
    {
        stream.read(
            reinterpret_cast<char*>(result.data()),
            static_cast<std::streamsize>(size));
        if (!stream)
        {
            throw std::runtime_error(
                "failed while reading binary file: " + path.string());
        }
    }
    return result;
}

std::vector<float>
read_float32(const std::filesystem::path& path)
{
    const auto bytes = read_bytes(path);
    if (bytes.size() % sizeof(float) != 0U)
    {
        throw std::runtime_error(
            "float32 file size is not aligned: " + path.string());
    }
    std::vector<float> result(
        bytes.size() / sizeof(float));
    if (!bytes.empty())
    {
        std::memcpy(
            result.data(),
            bytes.data(),
            bytes.size());
    }
    return result;
}

Config read_config(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error(
            "cannot open config: " + path.string());
    }
    Config result;
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            continue;
        }
        const auto tab = line.find('\t');
        if (tab == std::string::npos ||
            tab == 0U ||
            tab + 1U >= line.size())
        {
            throw std::runtime_error(
                "invalid config line: " + line);
        }
        const std::string key = line.substr(0U, tab);
        const std::string value = line.substr(tab + 1U);
        if (!result.emplace(key, value).second)
        {
            throw std::runtime_error(
                "duplicate config key: " + key);
        }
    }
    return result;
}

const std::string& require(
    const Config& config,
    const char* key)
{
    const auto found = config.find(key);
    if (found == config.end())
    {
        throw std::runtime_error(
            std::string("missing config key: ") + key);
    }
    return found->second;
}

std::size_t parse_size(
    const Config& config,
    const char* key)
{
    const std::string& text = require(config, key);
    std::size_t consumed = 0U;
    const unsigned long long value =
        std::stoull(text, &consumed);
    if (consumed != text.size() ||
        value >
            static_cast<unsigned long long>(
                (std::numeric_limits<std::size_t>::max)()))
    {
        throw std::runtime_error(
            std::string("invalid size config: ") + key);
    }
    return static_cast<std::size_t>(value);
}

std::int32_t parse_i32(
    const Config& config,
    const char* key)
{
    const std::string& text = require(config, key);
    std::size_t consumed = 0U;
    const long long value =
        std::stoll(text, &consumed);
    if (consumed != text.size() ||
        value < (std::numeric_limits<std::int32_t>::min)() ||
        value > (std::numeric_limits<std::int32_t>::max)())
    {
        throw std::runtime_error(
            std::string("invalid int32 config: ") + key);
    }
    return static_cast<std::int32_t>(value);
}

float parse_float(
    const Config& config,
    const char* key)
{
    const std::string& text = require(config, key);
    std::size_t consumed = 0U;
    const float value = std::stof(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value))
    {
        throw std::runtime_error(
            std::string("invalid float config: ") + key);
    }
    return value;
}

std::vector<kfcore::relation::Region>
read_regions(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error(
            "cannot open regions: " + path.string());
    }
    std::vector<kfcore::relation::Region> result;
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            continue;
        }
        std::istringstream row(line);
        kfcore::relation::Region region;
        if (!(row >> region.left
                  >> region.top
                  >> region.right
                  >> region.bottom
                  >> region.detector_score))
        {
            throw std::runtime_error(
                "invalid region line: " + line);
        }
        std::string extra;
        if (row >> extra)
        {
            throw std::runtime_error(
                "region line has extra fields: " + line);
        }
        result.push_back(region);
    }
    if (result.empty())
    {
        throw std::runtime_error(
            "GT region fixture must not be empty");
    }
    return result;
}

std::vector<std::string>
read_names(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error(
            "cannot open predicate names: " + path.string());
    }
    std::vector<std::string> result;
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            throw std::runtime_error(
                "predicate names must not contain empty rows");
        }
        result.push_back(line);
    }
    if (result.empty())
    {
        throw std::runtime_error(
            "predicate vocabulary must not be empty");
    }
    return result;
}

std::string json_escape(const std::string& value)
{
    std::string result;
    result.reserve(value.size() + 8U);
    for (const unsigned char c : value)
    {
        switch (c)
        {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\b': result += "\\b"; break;
        case '\f': result += "\\f"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (c < 0x20U)
            {
                const char hex[] = "0123456789abcdef";
                result += "\\u00";
                result += hex[(c >> 4U) & 0x0fU];
                result += hex[c & 0x0fU];
            }
            else
            {
                result.push_back(
                    static_cast<char>(c));
            }
        }
    }
    return result;
}

std::vector<std::filesystem::path>
vocabulary_cases(const std::filesystem::path& root)
{
    std::vector<std::filesystem::path> result;
    for (const auto& entry :
         std::filesystem::directory_iterator(root))
    {
        if (entry.is_directory())
        {
            result.push_back(entry.path());
        }
    }
    std::sort(result.begin(), result.end());
    if (result.empty())
    {
        throw std::runtime_error(
            "qualification fixture contains no vocabulary cases");
    }
    return result;
}

kfcore::relation::PredicateVocabulary
read_vocabulary(
    const std::filesystem::path& directory,
    std::size_t query_dim)
{
    kfcore::relation::PredicateVocabulary vocabulary;
    vocabulary.predicates =
        read_names(directory / "names.txt");
    vocabulary.embeddings =
        read_float32(directory / "W.f32");
    vocabulary.spatial_weights =
        read_float32(directory / "alpha.f32");
    vocabulary.embedding_dim = query_dim;
    vocabulary.text_encoder_provenance =
        "qualification-precomputed-W";
    vocabulary.routing_gate_provenance =
        "qualification-precomputed-alpha";

    const std::size_t expected =
        vocabulary.predicates.size() * query_dim;
    if (vocabulary.embeddings.size() != expected)
    {
        throw std::runtime_error(
            "W.f32 size does not match V*D");
    }
    if (vocabulary.spatial_weights.size() !=
        vocabulary.predicates.size())
    {
        throw std::runtime_error(
            "alpha.f32 size does not match V");
    }
    return vocabulary;
}

kfcore::yolo::TrackFrame
gt_object_table(
    std::int32_t width,
    std::int32_t height,
    const std::vector<kfcore::relation::Region>& regions)
{
    kfcore::yolo::TrackFrame frame;
    frame.image_width = width;
    frame.image_height = height;
    frame.detections.reserve(regions.size());
    for (const auto& region : regions)
    {
        kfcore::yolo::Detection detection {
            {
                region.left,
                region.top,
                region.right,
                region.bottom,
            },
            region.detector_score,
            -1,
        };
        frame.detections.push_back({
            detection,
            std::nullopt,
        });
    }
    return frame;
}

bool all_track_ids_null(
    const kfcore::pipelines::SceneGraphFrame& scene)
{
    for (const auto& object : scene.objects.detections)
    {
        if (object.track_id.has_value())
        {
            return false;
        }
    }
    for (const auto& edge : scene.relations.edges)
    {
        if (edge.subject_track_id.has_value() ||
            edge.object_track_id.has_value())
        {
            return false;
        }
    }
    return true;
}

void validate_scene(
    const kfcore::pipelines::SceneGraphFrame& scene)
{
    if (scene.objects.image_width !=
            scene.relations.image_width ||
        scene.objects.image_height !=
            scene.relations.image_height)
    {
        throw std::runtime_error(
            "SceneGraphFrame image geometry drifted");
    }
    for (const auto& edge : scene.relations.edges)
    {
        if (edge.subject_index >=
                scene.objects.detections.size() ||
            edge.object_index >=
                scene.objects.detections.size())
        {
            throw std::runtime_error(
                "relation edge references unavailable GT object");
        }
    }
    if (!all_track_ids_null(scene))
    {
        throw std::runtime_error(
            "GT-box qualification must remain track-id-free");
    }
}

void write_case(
    std::ostream& output,
    const std::string& case_name,
    const kfcore::relation::TimedRelationFrame& timed,
    std::size_t v,
    const std::string& backend,
    const std::string& device,
    bool backend_scoring,
    std::size_t scene_object_count,
    bool track_ids_null)
{
    output
        << std::setprecision(12)
        << "{"
        << "\"case\":\"" << json_escape(case_name) << "\","
        << "\"v\":" << v << ","
        << "\"vocabulary_version\":"
        << timed.frame.vocabulary_version << ","
        << "\"backend\":\"" << json_escape(backend) << "\","
        << "\"device\":\"" << json_escape(device) << "\","
        << "\"model_load_count\":1,"
        << "\"backend_scoring\":"
        << (backend_scoring ? "true" : "false") << ","
        << "\"scene_object_count\":"
        << scene_object_count << ","
        << "\"all_track_ids_null\":"
        << (track_ids_null ? "true" : "false") << ","
        << "\"valid_pair_count\":"
        << timed.timing.selected_pair_count << ","
        << "\"edges\":[";

    for (std::size_t index = 0U;
         index < timed.frame.edges.size();
         ++index)
    {
        if (index != 0U)
        {
            output << ",";
        }
        const auto& edge = timed.frame.edges[index];
        output
            << "{"
            << "\"subject\":" << edge.subject_index << ","
            << "\"object\":" << edge.object_index << ","
            << "\"predicate\":" << edge.predicate_index << ","
            << "\"score\":" << edge.score
            << "}";
    }

    output
        << "],"
        << "\"timing\":{"
        << "\"preprocess_ms\":"
        << timed.timing.preprocess_ms << ","
        << "\"runtime_ms\":"
        << timed.timing.runtime_ms << ","
        << "\"scoring_ms\":"
        << timed.timing.scoring_ms << ","
        << "\"decode_ms\":"
        << timed.timing.decode_ms << ","
        << "\"total_ms\":"
        << timed.timing.total_ms
        << "}"
        << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 5)
        {
            throw std::runtime_error(
                "usage: gtbox_ort_native <backend.so> <model.onnx> "
                "<fixture-dir> <output.jsonl>");
        }

        const std::filesystem::path backend_path = argv[1];
        const std::filesystem::path model_path = argv[2];
        const std::filesystem::path fixture = argv[3];
        const std::filesystem::path output_path = argv[4];

        const Config config =
            read_config(fixture / "config.tsv");
        const std::int32_t width =
            parse_i32(config, "width");
        const std::int32_t height =
            parse_i32(config, "height");
        const std::size_t query_dim =
            parse_size(config, "query_dim");

        const auto image_bytes =
            read_bytes(fixture / "image.bgr");
        const std::size_t expected_image_bytes =
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height) * 3U;
        if (image_bytes.size() != expected_image_bytes)
        {
            throw std::runtime_error(
                "image.bgr size does not match width*height*3");
        }
        const auto regions =
            read_regions(fixture / "regions.tsv");

        auto backend =
            kfcore::runtime::BackendPlugin::load(
                backend_path);
        if (backend->id() != "onnxruntime")
        {
            throw std::runtime_error(
                "qualification backend is not onnxruntime");
        }

        const auto devices = backend->devices();
        const auto cpu = std::find_if(
            devices.begin(),
            devices.end(),
            [](const kfcore::runtime::BackendDevice& device)
            {
                return device.id == "cpu";
            });
        if (cpu == devices.end())
        {
            throw std::runtime_error(
                "ONNX Runtime backend did not expose cpu device");
        }

        kfcore::runtime::ModelLoadRequest request;
        request.artifact_path = model_path;
        request.artifact_format = "onnx";
        request.device_id = "cpu";
        if (!backend->can_load(request))
        {
            throw std::runtime_error(
                "ONNX Runtime backend rejected relation ONNX");
        }

        kfcore::runtime::ResolvedModel resolved;
        resolved.route.backend_id = backend->id();
        resolved.route.device_id = "cpu";
        resolved.route.artifact.id = "gtbox-qualification";
        resolved.route.artifact.format = "onnx";
        resolved.route.artifact.path = model_path;
        resolved.route.artifact.backend = backend->id();
        resolved.route.artifact.device = "cpu";
        resolved.model = backend->load_model(request);
        if (!resolved.model)
        {
            throw std::runtime_error(
                "ONNX Runtime backend returned no executable model");
        }

        kfcore::relation::OpenVocabularyRelationOptions options;
        options.input_size =
            parse_i32(config, "input_size");
        options.max_boxes =
            parse_size(config, "max_boxes");
        options.max_pairs =
            parse_size(config, "max_pairs");
        options.query_dim = query_dim;
        options.threshold =
            parse_float(config, "threshold");
        options.pair_weight =
            parse_float(config, "pair_weight");
        options.calibration_a =
            parse_float(config, "calibration_a");
        options.calibration_b =
            parse_float(config, "calibration_b");
        options.top_k =
            parse_size(config, "top_k");

        auto relation =
            kfcore::relation::OpenVocabularyRelation::load_resolved(
                std::move(resolved),
                kfcore::relation::kDynamicOpenVocabularyRelationModelType,
                options);
        if (!relation->backend_scoring())
        {
            throw std::runtime_error(
                "qualification relation did not select backend scoring");
        }

        const kfcore::image::ImageView image {
            image_bytes.data(),
            image_bytes.size(),
            width,
            height,
            static_cast<std::size_t>(width) * 3U,
            kfcore::image::PixelFormat::Bgr8,
            kfcore::image::MemoryKind::Host,
        };

        std::ofstream output(output_path);
        if (!output)
        {
            throw std::runtime_error(
                "cannot open native qualification output");
        }

        std::uint64_t expected_version = 0U;
        const auto cases =
            vocabulary_cases(fixture / "vocabs");
        for (const auto& case_dir : cases)
        {
            const std::string case_name =
                case_dir.filename().string();
            auto vocabulary =
                read_vocabulary(case_dir, query_dim);
            const std::size_t v =
                vocabulary.predicates.size();

            relation->set_vocabulary(
                std::move(vocabulary));
            ++expected_version;
            if (relation->vocabulary_version() !=
                expected_version)
            {
                throw std::runtime_error(
                    "vocabulary version did not increment exactly once");
            }

            auto timed =
                relation->infer_timed(
                    image,
                    regions);
            if (timed.frame.vocabulary_version !=
                expected_version)
            {
                throw std::runtime_error(
                    "RelationFrame vocabulary version drifted");
            }

            kfcore::pipelines::SceneGraphFrame scene;
            scene.objects = gt_object_table(
                width,
                height,
                regions);
            scene.relations = timed.frame;
            validate_scene(scene);

            write_case(
                output,
                case_name,
                timed,
                v,
                relation->execution_route().backend_id,
                relation->execution_route().device_id,
                relation->backend_scoring(),
                scene.objects.detections.size(),
                all_track_ids_null(scene));
        }

        output.flush();
        if (!output)
        {
            throw std::runtime_error(
                "failed to write native qualification output");
        }

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
