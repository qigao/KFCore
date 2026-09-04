#include "kfcore/face_models/cpu.hpp"
#include "tinytest.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kfcore::face_models;

namespace
{

enum class SelectedModel
{
    Face68,
    ArcFace,
    AgeGender,
    InSwapper,
    GfpGan,
};

std::optional<SelectedModel> selected_model()
{
    constexpr char kVariable[] = "KFCORE_CPU_FACE_MODEL_TEST_KIND";
    const char* value = std::getenv(kVariable);
    if (value == nullptr || *value == '\0')
    {
        info("%s is required", kVariable);
        check(false);
        return std::nullopt;
    }
    const std::string kind(value);
    if (kind == "face68") return SelectedModel::Face68;
    if (kind == "arcface") return SelectedModel::ArcFace;
    if (kind == "age_gender") return SelectedModel::AgeGender;
    if (kind == "inswapper") return SelectedModel::InSwapper;
    if (kind == "gfpgan") return SelectedModel::GfpGan;
    info("unsupported CPU face model kind: %s", value);
    check(false);
    return std::nullopt;
}

std::filesystem::path required_model_path()
{
    constexpr char kVariable[] = "KFCORE_CPU_FACE_MODEL_TEST_PATH";
    const char* value = std::getenv(kVariable);
    if (value == nullptr || *value == '\0')
    {
        throw std::runtime_error(std::string(kVariable) + " must name a trusted ONNX model");
    }
    return std::filesystem::path(value);
}

std::vector<float> zero_image_tensor(std::int64_t extent)
{
    const std::size_t elements = static_cast<std::size_t>(
        kFaceModelInputChannels * extent * extent);
    return std::vector<float>(elements, 0.0F);
}

CpuTensorView view(const std::vector<float>& values)
{
    return { values.data(), values.size() };
}

template <typename Range>
void check_finite(const Range& values)
{
    for (float value : values)
    {
        check_true(std::isfinite(value));
    }
}

void run_face68()
{
    auto model = CpuFace68::load(required_model_path());
    const std::vector<float> input = zero_image_tensor(kFace68InputExtent);
    const Face68Result result = model->infer(view(input));
    check(result.size() == kFace68LandmarkCount);
    for (const Face68Landmark& landmark : result)
    {
        check_true(std::isfinite(landmark.x));
        check_true(std::isfinite(landmark.y));
        check_true(std::isfinite(landmark.score));
    }
}

void run_arcface()
{
    auto model = CpuArcFace::load(required_model_path());
    const std::vector<float> input = zero_image_tensor(kArcFaceInputExtent);
    const ArcFaceResult result = model->infer(view(input));
    check(result.size() == kArcFaceEmbeddingLength);
    check_finite(result);
}

void run_age_gender()
{
    auto model = CpuAgeGender::load(required_model_path());
    const std::vector<float> input = zero_image_tensor(kAgeGenderInputExtent);
    const AgeGenderResult result = model->infer(view(input));
    check(result.size() == kAgeGenderLogitCount);
    check_finite(result);
}

void run_inswapper()
{
    auto model = CpuInSwapper::load(required_model_path());
    const std::vector<float> target = zero_image_tensor(kInSwapperInputExtent);
    std::vector<float> source(kInSwapperEmbeddingLength, 0.0F);
    source.front() = 1.0F;
    const InSwapperResult first = model->infer(view(target), view(source));
    const InSwapperResult second = model->infer(view(target), view(source));
    check(first.values.size() == kInSwapperOutputElementCount);
    check_finite(first.values);
    check(first.values == second.values);
}

void run_gfpgan()
{
    auto model = CpuGfpGan::load(required_model_path());
    const std::vector<float> input = zero_image_tensor(kGfpGanInputExtent);
    const GfpGanResult first = model->infer(view(input));
    const GfpGanResult second = model->infer(view(input));
    check(first.values.size() == kGfpGanOutputElementCount);
    check_finite(first.values);
    check(first.values == second.values);
}

} // namespace

spec("ONNX Runtime CPU face model integration")
{
    it("executes one explicitly selected real model adapter")
    {
        const std::optional<SelectedModel> model = selected_model();
        if (!model)
        {
            return;
        }
        switch (*model)
        {
        case SelectedModel::Face68: run_face68(); break;
        case SelectedModel::ArcFace: run_arcface(); break;
        case SelectedModel::AgeGender: run_age_gender(); break;
        case SelectedModel::InSwapper: run_inswapper(); break;
        case SelectedModel::GfpGan: run_gfpgan(); break;
        }
    }
}
