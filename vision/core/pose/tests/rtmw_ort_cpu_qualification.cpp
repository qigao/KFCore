#include "rtmw_preprocess.hpp"
#include "simcc_decode.hpp"

#include "kfcore/image_processor/types.hpp"
#include "kfcore/pose/rtmw.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using kfcore::pose::RectF;

constexpr std::size_t kKeypoints = 133U;
constexpr std::size_t kInputWidth = 288U;
constexpr std::size_t kInputHeight = 384U;
constexpr std::size_t kXbins = 576U;
constexpr std::size_t kYbins = 768U;

struct Case
{
    std::size_t index = 0U;
    std::string label;
    RectF box;
};

struct ReferencePoint
{
    float model_x = -1.0F;
    float model_y = -1.0F;
    float source_x = -1.0F;
    float source_y = -1.0F;
    float confidence = 0.0F;
};

struct Metrics
{
    double preprocess_max_abs = 0.0;
    double preprocess_mean_abs = 0.0;
    double simcc_max_abs = 0.0;
    double model_coord_max_abs = 0.0;
    double model_score_max_abs = 0.0;
    double source_coord_max_abs = 0.0;
    double source_score_max_abs = 0.0;
};

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

std::vector<std::string> split_tabs(const std::string& line)
{
    std::vector<std::string> result;
    std::size_t begin = 0U;
    for (;;)
    {
        const std::size_t end = line.find('\t', begin);
        result.push_back(line.substr(
            begin,
            end == std::string::npos ? std::string::npos : end - begin));
        if (end == std::string::npos)
        {
            break;
        }
        begin = end + 1U;
    }
    return result;
}

std::vector<Case> load_cases(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    require(static_cast<bool>(stream), "cannot open cases.tsv");
    std::vector<Case> result;
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            continue;
        }
        const auto fields = split_tabs(line);
        require(fields.size() == 6U, "case row must have six fields");
        Case item;
        item.index = static_cast<std::size_t>(std::stoull(fields[0]));
        item.label = fields[1];
        item.box.x = std::stof(fields[2]);
        item.box.y = std::stof(fields[3]);
        item.box.width = std::stof(fields[4]);
        item.box.height = std::stof(fields[5]);
        require(item.index == result.size(), "case indexes must be contiguous");
        result.push_back(std::move(item));
    }
    require(!result.empty(), "qualification requires at least one case");
    return result;
}

std::vector<std::vector<ReferencePoint>>
load_reference(const std::filesystem::path& path, std::size_t case_count)
{
    std::ifstream stream(path);
    require(static_cast<bool>(stream), "cannot open reference.tsv");
    std::vector<std::vector<ReferencePoint>> result(
        case_count, std::vector<ReferencePoint>(kKeypoints));
    std::vector<std::size_t> seen(case_count, 0U);
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            continue;
        }
        const auto fields = split_tabs(line);
        require(fields.size() == 7U, "reference row must have seven fields");
        const std::size_t case_index =
            static_cast<std::size_t>(std::stoull(fields[0]));
        const std::size_t keypoint =
            static_cast<std::size_t>(std::stoull(fields[1]));
        require(case_index < case_count, "reference case index out of range");
        require(keypoint < kKeypoints, "reference keypoint out of range");
        ReferencePoint point;
        point.model_x = std::stof(fields[2]);
        point.model_y = std::stof(fields[3]);
        point.source_x = std::stof(fields[4]);
        point.source_y = std::stof(fields[5]);
        point.confidence = std::stof(fields[6]);
        result[case_index][keypoint] = point;
        ++seen[case_index];
    }
    for (const std::size_t count : seen)
    {
        require(count == kKeypoints, "reference must contain all keypoints");
    }
    return result;
}

std::vector<float> read_f32(const std::filesystem::path& path,
                            std::size_t expected_count)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(stream), "cannot open " + path.string());
    const std::streamsize bytes = stream.tellg();
    require(bytes >= 0, "cannot query binary fixture size");
    require(static_cast<std::size_t>(bytes) ==
                expected_count * sizeof(float),
            "binary fixture has unexpected size: " + path.string());
    stream.seekg(0);
    std::vector<float> result(expected_count);
    stream.read(reinterpret_cast<char*>(result.data()), bytes);
    require(static_cast<bool>(stream), "cannot read " + path.string());
    return result;
}

kfcore::image::BgrImage synthetic_image()
{
    kfcore::image::BgrImage image;
    image.width = 640;
    image.height = 480;
    image.pixels.resize(
        static_cast<std::size_t>(image.width) *
        static_cast<std::size_t>(image.height) * 3U);
    for (std::int32_t y = 0; y < image.height; ++y)
    {
        for (std::int32_t x = 0; x < image.width; ++x)
        {
            const std::size_t offset =
                (static_cast<std::size_t>(y) * image.width +
                 static_cast<std::size_t>(x)) * 3U;
            image.pixels[offset + 0U] = static_cast<std::uint8_t>(
                16 + (x * 96) / (image.width - 1) +
                (y * 48) / (image.height - 1));
            image.pixels[offset + 1U] = static_cast<std::uint8_t>(
                24 + (x * 48) / (image.width - 1) +
                (y * 96) / (image.height - 1));
            image.pixels[offset + 2U] = static_cast<std::uint8_t>(
                32 + (x * 80) / (image.width - 1) +
                (y * 80) / (image.height - 1));
        }
    }
    return image;
}

double compare_vectors(const std::vector<float>& actual,
                       const std::vector<float>& expected,
                       double tolerance,
                       const std::string& subject)
{
    require(actual.size() == expected.size(), subject + " size differs");
    double max_abs = 0.0;
    for (std::size_t i = 0U; i < actual.size(); ++i)
    {
        const double delta = std::fabs(
            static_cast<double>(actual[i]) -
            static_cast<double>(expected[i]));
        max_abs = (std::max)(max_abs, delta);
    }
    require(
        max_abs <= tolerance,
        subject + " max abs delta " + std::to_string(max_abs) +
            " exceeds " + std::to_string(tolerance));
    return max_abs;
}

double mean_abs_delta(const std::vector<float>& actual,
                      const std::vector<float>& expected)
{
    require(actual.size() == expected.size(), "mean delta size differs");
    double sum = 0.0;
    for (std::size_t i = 0U; i < actual.size(); ++i)
    {
        sum += std::fabs(
            static_cast<double>(actual[i]) -
            static_cast<double>(expected[i]));
    }
    return actual.empty() ? 0.0 : sum / static_cast<double>(actual.size());
}

const kfcore::runtime::TensorDescriptor&
find_tensor(const std::vector<kfcore::runtime::TensorDescriptor>& tensors,
            const std::string& name,
            bool is_input)
{
    const auto it = std::find_if(
        tensors.begin(), tensors.end(),
        [&](const auto& tensor)
        {
            return tensor.name == name && tensor.is_input == is_input;
        });
    require(it != tensors.end(), "missing tensor " + name);
    require(it->data_type == kfcore::runtime::DataType::Float32,
            name + " must be FP32 for ORT CPU qualification");
    return *it;
}

void write_report(const std::filesystem::path& path,
                  const Metrics& metrics,
                  std::size_t cases)
{
    std::ofstream stream(path);
    require(static_cast<bool>(stream), "cannot create qualification report");
    stream << std::setprecision(10)
           << "{\n"
           << "  \"schema\": \"kfcore.rtmw-ort-cpu-cpp-qualification/1\",\n"
           << "  \"passed\": true,\n"
           << "  \"provider\": \"onnxruntime\",\n"
           << "  \"device\": \"cpu\",\n"
           << "  \"case_count\": " << cases << ",\n"
           << "  \"preprocess_max_abs\": " << metrics.preprocess_max_abs << ",\n"
           << "  \"preprocess_mean_abs\": " << metrics.preprocess_mean_abs << ",\n"
           << "  \"simcc_max_abs\": " << metrics.simcc_max_abs << ",\n"
           << "  \"model_coord_max_abs\": " << metrics.model_coord_max_abs << ",\n"
           << "  \"model_score_max_abs\": " << metrics.model_score_max_abs << ",\n"
           << "  \"source_coord_max_abs\": " << metrics.source_coord_max_abs << ",\n"
           << "  \"source_score_max_abs\": " << metrics.source_score_max_abs << "\n"
           << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        require(argc == 4,
                "usage: rtmw_ort_cpu_qualification <plugin> <fixture-dir> <report>");
        const std::filesystem::path plugin_path = argv[1];
        const std::filesystem::path fixture = argv[2];
        const std::filesystem::path report = argv[3];

        const auto cases = load_cases(fixture / "cases.tsv");
        const auto reference =
            load_reference(fixture / "reference.tsv", cases.size());
        const auto image = synthetic_image();

        kfcore::runtime::Runtime runtime;
        const auto backend = runtime.load_backend(plugin_path);
        require(backend->id() == "onnxruntime",
                "qualification loaded the wrong backend");

        const auto package =
            kfcore::runtime::ModelPackage::load(fixture);
        require(package.model_type() == kfcore::pose::kRtmwModelType,
                "fixture model_type must be pose.rtmw");

        const auto policy =
            kfcore::runtime::ExecutionPolicy::exact("onnxruntime", "cpu");
        auto resolved = runtime.load_model(package, policy);
        require(resolved.route.backend_id == "onnxruntime" &&
                    resolved.route.device_id == "cpu",
                "raw qualification route must be ORT CPU");

        const auto tensors = resolved.model->tensors();
        const auto& input_desc = find_tensor(tensors, "input", true);
        const auto& x_desc = find_tensor(tensors, "simcc_x", false);
        const auto& y_desc = find_tensor(tensors, "simcc_y", false);
        auto context = resolved.model->create_context();

        kfcore::pose::RtmwOptions options;
        options.input_width = static_cast<std::int32_t>(kInputWidth);
        options.input_height = static_cast<std::int32_t>(kInputHeight);
        auto rtmw = kfcore::pose::Rtmw::load(
            runtime, package, policy, options);

        Metrics metrics;
        double preprocess_sum = 0.0;
        std::size_t preprocess_values = 0U;

        for (const Case& item : cases)
        {
            const auto expected_preprocess = read_f32(
                fixture / ("preprocess-" + std::to_string(item.index) + ".f32"),
                3U * kInputHeight * kInputWidth);
            const auto expected_x = read_f32(
                fixture / ("simcc-x-" + std::to_string(item.index) + ".f32"),
                kKeypoints * kXbins);
            const auto expected_y = read_f32(
                fixture / ("simcc-y-" + std::to_string(item.index) + ".f32"),
                kKeypoints * kYbins);

            const auto actual_preprocess =
                kfcore::pose::detail::preprocess_rtmw(
                    image, item.box, options,
                    static_cast<std::int32_t>(kInputWidth),
                    static_cast<std::int32_t>(kInputHeight));

            const double preprocess_max = compare_vectors(
                actual_preprocess.nchw, expected_preprocess, 0.020F,
                "G1 preprocess/" + item.label);
            metrics.preprocess_max_abs =
                (std::max)(metrics.preprocess_max_abs, preprocess_max);
            for (std::size_t i = 0U; i < expected_preprocess.size(); ++i)
            {
                preprocess_sum += std::fabs(
                    static_cast<double>(actual_preprocess.nchw[i]) -
                    static_cast<double>(expected_preprocess[i]));
            }
            preprocess_values += expected_preprocess.size();

            std::vector<float> actual_x(kKeypoints * kXbins);
            std::vector<float> actual_y(kKeypoints * kYbins);
            kfcore::runtime::TensorView input {
                input_desc.name,
                kfcore::runtime::DataType::Float32,
                {1, 3, static_cast<std::int64_t>(kInputHeight),
                 static_cast<std::int64_t>(kInputWidth)},
                expected_preprocess.data(),
                expected_preprocess.size() * sizeof(float),
                kfcore::runtime::MemoryKind::Host,
                {},
            };
            std::vector<kfcore::runtime::MutableTensorView> outputs {
                {
                    x_desc.name,
                    kfcore::runtime::DataType::Float32,
                    {1, static_cast<std::int64_t>(kKeypoints),
                     static_cast<std::int64_t>(kXbins)},
                    actual_x.data(),
                    actual_x.size() * sizeof(float),
                    kfcore::runtime::MemoryKind::Host,
                    {},
                },
                {
                    y_desc.name,
                    kfcore::runtime::DataType::Float32,
                    {1, static_cast<std::int64_t>(kKeypoints),
                     static_cast<std::int64_t>(kYbins)},
                    actual_y.data(),
                    actual_y.size() * sizeof(float),
                    kfcore::runtime::MemoryKind::Host,
                    {},
                },
            };
            context->run({input}, outputs);

            metrics.simcc_max_abs = (std::max)(
                metrics.simcc_max_abs,
                compare_vectors(
                    actual_x, expected_x, 1.0e-4,
                    "G2 simcc_x/" + item.label));
            metrics.simcc_max_abs = (std::max)(
                metrics.simcc_max_abs,
                compare_vectors(
                    actual_y, expected_y, 1.0e-4,
                    "G2 simcc_y/" + item.label));

            std::vector<kfcore::pose::detail::DecodedSimccKeypoint>
                decoded(kKeypoints);
            kfcore::pose::detail::decode_simcc(
                {expected_x.data(),
                 kfcore::pose::detail::SimccElementType::Float32,
                 kKeypoints, kXbins},
                {expected_y.data(),
                 kfcore::pose::detail::SimccElementType::Float32,
                 kKeypoints, kYbins},
                2.0F, decoded.data(), decoded.size());

            for (std::size_t keypoint = 0U; keypoint < kKeypoints; ++keypoint)
            {
                const ReferencePoint& expected =
                    reference[item.index][keypoint];
                const auto& actual = decoded[keypoint];
                metrics.model_coord_max_abs = (std::max)(
                    metrics.model_coord_max_abs,
                    static_cast<double>((std::max)(
                        std::fabs(actual.x - expected.model_x),
                        std::fabs(actual.y - expected.model_y))));
                metrics.model_score_max_abs = (std::max)(
                    metrics.model_score_max_abs,
                    static_cast<double>(
                        std::fabs(actual.confidence - expected.confidence)));
            }
            require(metrics.model_coord_max_abs <= 1.0e-6,
                    "G3 decoded model coordinate differs from Python reference");
            require(metrics.model_score_max_abs <= 1.0e-5,
                    "G3 decoded score differs from Python reference");

            const auto pose = rtmw->infer(image.view(), item.box);
            for (std::size_t keypoint = 0U; keypoint < kKeypoints; ++keypoint)
            {
                const ReferencePoint& expected =
                    reference[item.index][keypoint];
                const auto& actual = pose.keypoints[keypoint];
                metrics.source_coord_max_abs = (std::max)(
                    metrics.source_coord_max_abs,
                    static_cast<double>((std::max)(
                        std::fabs(actual.x - expected.source_x),
                        std::fabs(actual.y - expected.source_y))));
                metrics.source_score_max_abs = (std::max)(
                    metrics.source_score_max_abs,
                    static_cast<double>(
                        std::fabs(actual.score - expected.confidence)));
            }
        }

        metrics.preprocess_mean_abs =
            preprocess_values == 0U
                ? 0.0
                : preprocess_sum / static_cast<double>(preprocess_values);

        require(metrics.preprocess_mean_abs <= 0.0015,
                "G1 preprocess mean abs delta exceeds 0.0015");
        require(metrics.source_coord_max_abs <= 0.25,
                "G4 source coordinate max delta exceeds 0.25 px");
        require(metrics.source_score_max_abs <= 0.01,
                "G4 source confidence max delta exceeds 0.01");

        write_report(report, metrics, cases.size());
        std::cout << std::fixed << std::setprecision(8)
                  << "G1 preprocess max=" << metrics.preprocess_max_abs
                  << " mean=" << metrics.preprocess_mean_abs << '\n'
                  << "G2 simcc max=" << metrics.simcc_max_abs << '\n'
                  << "G3 model coord max=" << metrics.model_coord_max_abs
                  << " score=" << metrics.model_score_max_abs << '\n'
                  << "G4 source coord max=" << metrics.source_coord_max_abs
                  << " score=" << metrics.source_score_max_abs << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "RTMW ORT CPU qualification failed: "
                  << error.what() << '\n';
        return 1;
    }
}
