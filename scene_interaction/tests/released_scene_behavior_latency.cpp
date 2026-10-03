#include "kfcore/relation/open_vocabulary_relation.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/runtime.hpp"
#include "kfcore/yolo/detector.hpp"
#include "kfcore/pipelines/scene_graph.hpp"
#include "kfcore/scene_interaction/pipeline.hpp"
#include "kfcore/scene_interaction/latency_report.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr std::array<char, 8> kVocabMagic{
    'K','F','R','E','L','V','O','1'
};

std::uint32_t read_u32(std::istream& stream)
{
    std::array<unsigned char, 4> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), 4);
    if (!stream)
    {
        throw std::runtime_error("unexpected EOF while reading u32");
    }
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

float read_f32(std::istream& stream)
{
    const std::uint32_t bits = read_u32(stream);
    float value = 0.0F;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

kfcore::relation::PredicateVocabulary
load_vocabulary(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("cannot open vocabulary binary");
    }
    std::array<char, 8> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!stream || magic != kVocabMagic)
    {
        throw std::runtime_error("invalid vocabulary binary magic");
    }

    const std::uint32_t count = read_u32(stream);
    const std::uint32_t dim = read_u32(stream);
    if (count == 0U || dim != 512U)
    {
        throw std::runtime_error("invalid vocabulary dimensions");
    }

    kfcore::relation::PredicateVocabulary value;
    value.embedding_dim = dim;
    value.predicates.reserve(count);
    value.embeddings.reserve(
        static_cast<std::size_t>(count) * dim);
    value.spatial_weights.reserve(count);

    for (std::uint32_t row = 0; row < count; ++row)
    {
        const std::uint32_t name_size = read_u32(stream);
        if (name_size == 0U || name_size > 65535U)
        {
            throw std::runtime_error("invalid predicate name length");
        }
        std::string name(name_size, '\0');
        stream.read(name.data(), static_cast<std::streamsize>(name_size));
        if (!stream)
        {
            throw std::runtime_error("unexpected EOF in predicate name");
        }
        value.predicates.push_back(std::move(name));
        value.spatial_weights.push_back(read_f32(stream));
        for (std::uint32_t column = 0; column < dim; ++column)
        {
            value.embeddings.push_back(read_f32(stream));
        }
    }

    char trailing = 0;
    if (stream.read(&trailing, 1))
    {
        throw std::runtime_error("vocabulary binary has trailing bytes");
    }
    return value;
}

struct FrameSpec
{
    std::filesystem::path relative;
    std::int32_t width = 0;
    std::int32_t height = 0;
    std::string image_id;
};

std::vector<FrameSpec>
load_frames(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("cannot open frames.tsv");
    }
    std::vector<FrameSpec> result;
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            continue;
        }
        std::istringstream fields(line);
        std::string relative;
        std::string width_text;
        std::string height_text;
        std::string image_id;
        if (!std::getline(fields, relative, '\t') ||
            !std::getline(fields, width_text, '\t') ||
            !std::getline(fields, height_text, '\t') ||
            !std::getline(fields, image_id))
        {
            throw std::runtime_error("invalid frames.tsv row");
        }
        const int width = std::stoi(width_text);
        const int height = std::stoi(height_text);
        if (width <= 0 || height <= 0)
        {
            throw std::runtime_error("frame dimensions must be positive");
        }
        result.push_back({
            std::filesystem::path(relative),
            static_cast<std::int32_t>(width),
            static_cast<std::int32_t>(height),
            image_id,
        });
    }
    if (result.empty())
    {
        throw std::runtime_error("frames.tsv is empty");
    }
    return result;
}

std::vector<std::uint8_t>
load_frame_bytes(const std::filesystem::path& path,
                 std::int32_t width,
                 std::int32_t height)
{
    const std::uintmax_t expected =
        static_cast<std::uintmax_t>(width) *
        static_cast<std::uintmax_t>(height) * 3U;
    if (expected >
        static_cast<std::uintmax_t>(
            (std::numeric_limits<std::size_t>::max)()))
    {
        throw std::runtime_error("frame byte count overflows size_t");
    }

    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
    {
        throw std::runtime_error("cannot open raw BGR frame");
    }
    const auto size = stream.tellg();
    if (size < 0 || static_cast<std::uintmax_t>(size) != expected)
    {
        throw std::runtime_error("raw BGR frame size mismatch");
    }
    stream.seekg(0);
    std::vector<std::uint8_t> bytes(
        static_cast<std::size_t>(expected));
    stream.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!stream)
    {
        throw std::runtime_error("cannot read raw BGR frame");
    }
    return bytes;
}

kfcore::scene_interaction::SceneBehaviorModel
make_behavior_model(
    std::size_t predicate_count,
    std::uint64_t vocabulary_version)
{
    using namespace kfcore::scene_interaction;
    if (predicate_count == 0U)
    {
        throw std::runtime_error("behavior model needs predicates");
    }
    SceneBehaviorModel model;
    model.predicate_count = predicate_count;
    model.vocabulary_version = vocabulary_version;
    model.reservoir_size = 1U;
    model.leak_rate = 1.0F;
    model.neutral_index = 0U;
    model.labels = {"none", "active"};
    model.input_weights.assign(
        predicate_count + kSceneGeometryFeatureCount,
        0.0F);
    model.input_weights[0] = 1.0F;
    model.recurrent_weights = {0.0F};
    model.reservoir_bias = {0.0F};
    model.output_weights = {0.0F, 1.0F};
    model.output_bias = {0.0F, 0.0F};
    return model;
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

bool same_plugin_path(const std::filesystem::path& left,
                      const std::filesystem::path& right)
{
    std::error_code left_error;
    std::error_code right_error;
    const auto left_canonical =
        std::filesystem::canonical(left, left_error);
    const auto right_canonical =
        std::filesystem::canonical(right, right_error);
    return !left_error && !right_error &&
        left_canonical == right_canonical;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 7 && argc != 12)
        {
            std::cerr
                << "usage: released_scene_behavior_latency "
                << "<ort-plugin> <detector-package> <relation-package> "
                   "<vocabulary.bin> <frames.tsv> <timing.jsonl>\n"
                << "   or: released_scene_behavior_latency "
                   "<detector-plugin> <relation-plugin> "
                   "<detector-package> <relation-package> "
                   "<vocabulary.bin> <frames.tsv> <timing.jsonl> "
                   "<detector-backend> <detector-device> "
                   "<relation-backend> <relation-device>\n";
            return 2;
        }

        const bool explicit_routes = argc == 12;
        const std::filesystem::path detector_plugin_path(argv[1]);
        const std::filesystem::path relation_plugin_path(
            explicit_routes ? argv[2] : argv[1]);
        const std::filesystem::path detector_package_path(
            explicit_routes ? argv[3] : argv[2]);
        const std::filesystem::path relation_package_path(
            explicit_routes ? argv[4] : argv[3]);
        const std::filesystem::path vocabulary_path(
            explicit_routes ? argv[5] : argv[4]);
        const std::filesystem::path frames_path(
            explicit_routes ? argv[6] : argv[5]);
        const std::filesystem::path output_path(
            explicit_routes ? argv[7] : argv[6]);
        const std::string detector_backend_id =
            explicit_routes ? std::string(argv[8]) : "onnxruntime";
        const std::string detector_device_id =
            explicit_routes ? std::string(argv[9]) : "cpu";
        const std::string relation_backend_id =
            explicit_routes ? std::string(argv[10]) : "onnxruntime";
        const std::string relation_device_id =
            explicit_routes ? std::string(argv[11]) : "cpu";

        require(
            valid_backend_id(detector_backend_id) &&
                valid_backend_id(relation_backend_id),
            "scene behavior backend must be onnxruntime or tensorrt");
        require(
            valid_device_id(detector_device_id) &&
                valid_device_id(relation_device_id),
            "scene behavior device must be cpu, cuda, or cuda:N");

        if (std::filesystem::exists(output_path))
        {
            throw std::runtime_error(
                "timing output already exists");
        }

        kfcore::runtime::Runtime runtime;
        const auto detector_backend =
            runtime.load_backend(detector_plugin_path);
        require(
            detector_backend->id() == detector_backend_id,
            "latency qualification loaded wrong detector backend");

        if (relation_backend_id == detector_backend_id)
        {
            require(
                same_plugin_path(
                    detector_plugin_path,
                    relation_plugin_path),
                "same backend id requires the same plugin path");
        }
        else
        {
            const auto relation_backend =
                runtime.load_backend(relation_plugin_path);
            require(
                relation_backend->id() == relation_backend_id,
                "latency qualification loaded wrong relation backend");
        }

        const auto detector_package =
            kfcore::runtime::ModelPackage::load(
                detector_package_path);
        const auto relation_package =
            kfcore::runtime::ModelPackage::load(
                relation_package_path);

        kfcore::yolo::YoloDetectorOptions detector_options;
        detector_options.score_threshold = 0.1F;
        detector_options.iou_threshold = 0.45F;
        detector_options.max_detections = 32U;

        auto detector =
            kfcore::yolo::YoloDetector::load(
                runtime,
                detector_package,
                kfcore::runtime::ExecutionPolicy::exact(
                    detector_backend_id,
                    detector_device_id),
                detector_options);
        require(
            detector->execution_route().backend_id ==
                    detector_backend_id &&
                detector->execution_route().device_id ==
                    detector_device_id,
            "detector did not resolve to requested backend/device");

        kfcore::relation::OpenVocabularyRelationOptions relation_options;
        relation_options.input_size = 448;
        relation_options.max_boxes = 32U;
        relation_options.max_pairs = 128U;
        relation_options.query_dim = 512U;
        relation_options.pair_weight = 1.0F;
        relation_options.calibration_a = 0.5651F;
        relation_options.calibration_b = -1.9623F;
        relation_options.threshold = 0.0F;
        relation_options.top_k = 100U;
        relation_options.weight_ranking_by_detector_score = false;

        auto relation =
            kfcore::relation::OpenVocabularyRelation::load(
                runtime,
                relation_package,
                kfcore::runtime::ExecutionPolicy::exact(
                    relation_backend_id,
                    relation_device_id),
                relation_options);
        require(
            relation->execution_route().backend_id ==
                    relation_backend_id &&
                relation->execution_route().device_id ==
                    relation_device_id,
            "relation did not resolve to requested backend/device");

        auto vocabulary =
            load_vocabulary(vocabulary_path);
        relation->set_vocabulary(vocabulary);
        require(
            relation->vocabulary_version() == 1U,
            "relation vocabulary version did not initialize to 1");

        auto scene_graph =
            kfcore::pipelines::SceneGraphPipeline::create(
                std::move(detector),
                std::move(relation));

        kfcore::scene_interaction::SceneInteractionOptions interaction_options;
        interaction_options.minimum_score = 0.0F;
        interaction_options.minimum_margin = 0.0F;
        interaction_options.max_pair_states = 128U;

        auto behavior =
            kfcore::scene_interaction::SceneBehaviorPipeline::create(
                std::move(scene_graph),
                interaction_options);

        const auto model = make_behavior_model(
            vocabulary.predicates.size(),
            behavior->vocabulary_version());
        (void)behavior->configure_model(model, 0.0);

        const auto frames = load_frames(frames_path);
        const auto frame_root = frames_path.parent_path();

        std::ofstream output(output_path);
        if (!output)
        {
            throw std::runtime_error("cannot create timing JSONL");
        }

        double seconds = 0.0;
        std::size_t samples = 0U;
        for (const auto& frame : frames)
        {
            auto bytes = load_frame_bytes(
                frame_root / frame.relative,
                frame.width,
                frame.height);
            const kfcore::image::ImageView image {
                bytes.data(),
                bytes.size(),
                frame.width,
                frame.height,
                static_cast<std::size_t>(frame.width) * 3U,
                kfcore::image::PixelFormat::Bgr8,
                kfcore::image::MemoryKind::Host,
            };

            // HICO is a set of independent still images, not one video.
            // Reset identity/temporal state for each image, then use one
            // identical warm observation so ByteTrack's second observation
            // owns stable track IDs. Only the second pass is measured.
            (void)behavior->reset_tracking(seconds);
            (void)behavior->process(image, seconds);
            seconds += 1.0 / 30.0;

            const auto timed =
                behavior->process_timed(image, seconds);
            output
                << kfcore::scene_interaction::
                    scene_behavior_timing_json(
                        timed.timing)
                << '\n';
            ++samples;
            seconds += 1.0 / 30.0;
        }

        require(
            samples == frames.size(),
            "timing sample count mismatch");
        std::cout
            << "released SceneBehavior latency samples="
            << samples
            << " detector=" << detector_backend_id
            << "/" << detector_device_id
            << " relation=" << relation_backend_id
            << "/" << relation_device_id
            << "\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "released SceneBehavior latency: FAIL: "
            << error.what() << '\n';
        return 1;
    }
}
