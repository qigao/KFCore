#include "kfcore/face_models/tensorrt.hpp"
#include "tinytest.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kfcore::face_models;
using kfcore::tensorrt::DataType;
using kfcore::tensorrt::Engine;
using kfcore::tensorrt::MemoryKind;
using kfcore::tensorrt::TensorDescriptor;
using kfcore::tensorrt::TensorIoMode;
using kfcore::tensorrt::TensorShape;
using kfcore::tensorrt::TensorView;

namespace
{

enum class SelectedModel
{
    ArcFace,
    AgeGender,
    Face68,
    InSwapper,
    GfpGan
};

std::optional<SelectedModel> selected_model()
{
    constexpr char kVariable[] = "KFCORE_FACE_MODEL_TEST_KIND";
    const char* value = std::getenv(kVariable);
    if (value == nullptr || *value == '\0')
    {
        info("%s must be arcface, age_gender, face68, inswapper, or gfpgan", kVariable);
        check(false);
        return std::nullopt;
    }

    const std::string kind(value);
    if (kind == "arcface")
    {
        return SelectedModel::ArcFace;
    }
    if (kind == "age_gender")
    {
        return SelectedModel::AgeGender;
    }
    if (kind == "face68")
    {
        return SelectedModel::Face68;
    }
    if (kind == "inswapper")
    {
        return SelectedModel::InSwapper;
    }
    if (kind == "gfpgan")
    {
        return SelectedModel::GfpGan;
    }

    info("%s has unsupported value: %s", kVariable, value);
    check(false);
    return std::nullopt;
}

std::filesystem::path required_engine_path(const char* variable)
{
    const char* value = std::getenv(variable);
    if (value == nullptr || *value == '\0')
    {
        throw std::runtime_error(std::string(variable) +
                                 " must name a trusted TensorRT engine");
    }
    return std::filesystem::path(value);
}

std::vector<float> zero_input(std::int64_t extent)
{
    return std::vector<float>(static_cast<std::size_t>(kFaceModelInputChannels * extent * extent),
                              0.0F);
}

TensorView prepared_input(const std::string& name, std::int64_t extent,
                          const std::vector<float>& storage)
{
    return { name,
             DataType::Float32,
             { 1, kFaceModelInputChannels, extent, extent },
             storage.data(),
             storage.size() * sizeof(float),
             MemoryKind::Host };
}

template <std::size_t N>
void check_finite(const std::array<float, N>& values)
{
    check(values.size() == N);
    for (float value : values)
    {
        check_true(std::isfinite(value));
    }
}

template <std::size_t N>
void print_summary(const char* model, const std::array<float, N>& values)
{
    const auto range = std::minmax_element(values.begin(), values.end());
    double sum = 0.0;
    for (float value : values)
    {
        sum += static_cast<double>(value);
    }
    std::cout << "[integration] " << model << " batch=1 result_width=" << values.size()
              << " min=" << *range.first << " max=" << *range.second << " sum=" << sum
              << '\n';
}

const TensorDescriptor& required_heatmap(const std::vector<TensorDescriptor>& tensors,
                                         const std::string& name)
{
    for (const TensorDescriptor& tensor : tensors)
    {
        if (tensor.name == name)
        {
            return tensor;
        }
    }
    throw std::runtime_error("trusted Face68 engine does not expose its required heatmap output");
}

void run_arcface_integration()
{
    constexpr char kEngineVariable[] =
        "KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_ARCFACE";
    const ArcFaceOptions options;
    auto model = TensorRtArcFace::load(required_engine_path(kEngineVariable), options);
    const std::vector<float> input = zero_input(kArcFaceInputExtent);

    const std::vector<ArcFaceResult> results =
        model->infer(prepared_input(options.input_name, kArcFaceInputExtent, input));

    check(results.size() == std::size_t { 1 });
    if (results.size() != std::size_t { 1 })
    {
        return;
    }
    check_finite(results.front());
    print_summary("ArcFace adapter", results.front());
}

void run_age_gender_integration()
{
    constexpr char kEngineVariable[] =
        "KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_AGE_GENDER";
    const AgeGenderOptions options;
    auto model = TensorRtAgeGender::load(required_engine_path(kEngineVariable), options);
    const std::vector<float> input = zero_input(kAgeGenderInputExtent);

    const std::vector<AgeGenderResult> results =
        model->infer(prepared_input(options.input_name, kAgeGenderInputExtent, input));

    check(results.size() == std::size_t { 1 });
    if (results.size() != std::size_t { 1 })
    {
        return;
    }
    check_finite(results.front());
    print_summary("AgeGender adapter", results.front());
}

void run_face68_integration()
{
    constexpr char kEngineVariable[] =
        "KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_FACE68";
    const std::filesystem::path engine_path = required_engine_path(kEngineVariable);
    const Face68Options options;
    {
        const auto engine = Engine::load(engine_path, options.engine);
        const TensorDescriptor& heatmap =
            required_heatmap(engine->tensors(), options.heatmap_output_name);
        check(heatmap.mode == TensorIoMode::Output);
        check(heatmap.data_type == DataType::Float32);
        check(heatmap.declared_shape.size() == std::size_t { 4 });
        if (heatmap.declared_shape.size() != std::size_t { 4 })
        {
            throw std::runtime_error("trusted Face68 heatmap output must have rank four");
        }
        check(heatmap.declared_shape[1] ==
              static_cast<std::int64_t>(kFace68LandmarkCount));
        check(heatmap.declared_shape[2] == kFace68HeatmapExtent);
        check(heatmap.declared_shape[3] == kFace68HeatmapExtent);
    }

    auto model = TensorRtFace68::load(engine_path, options);
    const std::vector<float> input = zero_input(kFace68InputExtent);
    const std::vector<Face68Result> results =
        model->infer(prepared_input(options.input_name, kFace68InputExtent, input));

    check(results.size() == std::size_t { 1 });
    if (results.size() != std::size_t { 1 })
    {
        return;
    }
    check(results.front().size() == kFace68LandmarkCount);
    float minimum = results.front().front().x;
    float maximum = minimum;
    double sum = 0.0;
    for (const Face68Landmark& landmark : results.front())
    {
        check_true(std::isfinite(landmark.x));
        check_true(std::isfinite(landmark.y));
        check_true(std::isfinite(landmark.score));
        minimum = (std::min)({ minimum, landmark.x, landmark.y, landmark.score });
        maximum = (std::max)({ maximum, landmark.x, landmark.y, landmark.score });
        sum += static_cast<double>(landmark.x) + static_cast<double>(landmark.y) +
               static_cast<double>(landmark.score);
    }
    std::cout << "[integration] Face68 adapter batch=1 landmarks="
              << results.front().size() << " heatmap_bound=1 min=" << minimum
              << " max=" << maximum << " sum=" << sum << '\n';
}

void check_finite_vector(const std::vector<float>& values, std::size_t expected_size)
{
    check(values.size() == expected_size);
    for (float value : values)
    {
        check_true(std::isfinite(value));
    }
}

void run_inswapper_integration()
{
    constexpr char kEngineVariable[] =
        "KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_INSWAPPER";
    const InSwapperOptions options;
    auto model = TensorRtInSwapper::load(required_engine_path(kEngineVariable), options);
    const std::vector<float> target = zero_input(kInSwapperInputExtent);
    std::vector<float> source(kInSwapperEmbeddingLength, 0.0F);
    source.front() = 1.0F;
    const TensorView source_view { options.source_input_name,
                                   DataType::Float32,
                                   { 1, static_cast<std::int64_t>(source.size()) },
                                   source.data(),
                                   source.size() * sizeof(float),
                                   MemoryKind::Host };

    const InSwapperResult first = model->infer(
        prepared_input(options.target_input_name, kInSwapperInputExtent, target), source_view);
    const InSwapperResult second = model->infer(
        prepared_input(options.target_input_name, kInSwapperInputExtent, target), source_view);
    check_finite_vector(first.values, kInSwapperOutputElementCount);
    check(first.values == second.values);
}

void run_gfpgan_integration()
{
    constexpr char kEngineVariable[] = "KFCORE_TENSORRT_RUNTIME_TEST_ENGINE_GFPGAN";
    const GfpGanOptions options;
    auto model = TensorRtGfpGan::load(required_engine_path(kEngineVariable), options);
    const std::vector<float> input = zero_input(kGfpGanInputExtent);

    const GfpGanResult first =
        model->infer(prepared_input(options.input_name, kGfpGanInputExtent, input));
    const GfpGanResult second =
        model->infer(prepared_input(options.input_name, kGfpGanInputExtent, input));
    check_finite_vector(first.values, kGfpGanOutputElementCount);
    check(first.values == second.values);
}

} // namespace

spec("TensorRT face model real-engine integration")
{
    it("executes exactly one adapter selected by an exact model kind")
    {
        const std::optional<SelectedModel> model = selected_model();
        if (!model.has_value())
        {
            return;
        }

        switch (*model)
        {
        case SelectedModel::ArcFace:
            run_arcface_integration();
            break;
        case SelectedModel::AgeGender:
            run_age_gender_integration();
            break;
        case SelectedModel::Face68:
            run_face68_integration();
            break;
        case SelectedModel::InSwapper:
            run_inswapper_integration();
            break;
        case SelectedModel::GfpGan:
            run_gfpgan_integration();
            break;
        }
    }
}
