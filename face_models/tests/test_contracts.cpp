#include "contracts.hpp"

#include "kfcore/face_models/error.hpp"
#include "kfcore/face_models/tensorrt.hpp"
#include "tinytest.hpp"

#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using namespace kfcore::face_models;
using namespace kfcore::tensorrt;

namespace
{

TensorDescriptor input(std::string name, TensorShape declared, TensorShape minimum,
                       TensorShape optimum, TensorShape maximum)
{
    return { std::move(name), TensorIoMode::Input, DataType::Float32, std::move(declared),
             TensorProfile { std::move(minimum), std::move(optimum), std::move(maximum) } };
}

TensorDescriptor output(std::string name, TensorShape declared)
{
    return { std::move(name), TensorIoMode::Output, DataType::Float32, std::move(declared),
             std::nullopt };
}

std::vector<TensorDescriptor> face68_metadata(bool heatmaps = true)
{
    std::vector<TensorDescriptor> tensors = {
        input("input", { -1, 3, 256, 256 }, { 1, 3, 256, 256 }, { 2, 3, 256, 256 },
              { 4, 3, 256, 256 }),
        output("landmarks_xyscore", { -1, 68, 3 }),
    };
    if (heatmaps)
    {
        tensors.push_back(output("heatmaps", { -1, 68, 64, 64 }));
    }
    return tensors;
}

std::vector<TensorDescriptor> arcface_metadata()
{
    return {
        input("input.1", { -1, 3, 112, 112 }, { 1, 3, 112, 112 }, { 2, 3, 112, 112 },
              { 8, 3, 112, 112 }),
        output("683", { -1, 512 }),
    };
}

std::vector<TensorDescriptor> age_gender_metadata()
{
    return {
        input("pixel_values", { -1, 3, 224, 224 }, { 1, 3, 224, 224 },
              { 2, 3, 224, 224 }, { 8, 3, 224, 224 }),
        output("logits", { -1, 2 }),
    };
}

void expect_error(const std::function<void()>& operation, FaceModelErrorCode code,
                  const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const FaceModelError& error)
    {
        threw = true;
        check(error.code() == code);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

spec("strict TensorRT face model contracts")
{
    it("rejects invalid adapter limits and tensor names before engine loading")
    {
        Face68Options face68;
        face68.input_name = "";
        expect_error([&] { detail::validate_face68_options(face68); },
                     FaceModelErrorCode::InvalidArgument, "empty");

        ArcFaceOptions arcface;
        arcface.output_name = arcface.input_name;
        expect_error([&] { detail::validate_arcface_options(arcface); },
                     FaceModelErrorCode::InvalidArgument, "distinct");

        AgeGenderOptions age_gender;
        age_gender.max_batch = 0;
        expect_error([&] { detail::validate_age_gender_options(age_gender); },
                     FaceModelErrorCode::ResourceLimitExceeded, "max_batch");
    }

    it("accepts Face68 with or without its exact optional heatmap output")
    {
        Face68Options options;
        options.max_batch = 4;
        const detail::Face68Contract with_heatmaps =
            detail::validate_face68_contract(face68_metadata(), options);
        const detail::Face68Contract without_heatmaps =
            detail::validate_face68_contract(face68_metadata(false), options);

        check_true(with_heatmaps.has_heatmaps);
        check_false(without_heatmaps.has_heatmaps);
        check(with_heatmaps.batch.minimum == std::size_t { 1 });
        check(with_heatmaps.batch.maximum == std::size_t { 4 });
    }

    it("accepts only exact explicitly configured tensor names")
    {
        ArcFaceOptions arcface;
        arcface.max_batch = 4;
        auto renamed = arcface_metadata();
        renamed[0].name = "images";
        expect_error([&] { (void)detail::validate_arcface_contract(renamed, arcface); },
                     FaceModelErrorCode::ModelContractMismatch, "input.1");

        arcface.input_name = "images";
        const detail::SingleOutputContract contract =
            detail::validate_arcface_contract(renamed, arcface);
        check(contract.input_name == "images");
    }

    it("rejects missing duplicate and extra tensors without alias fallback")
    {
        Face68Options options;
        options.max_batch = 4;
        auto missing = face68_metadata();
        missing.erase(missing.begin() + 1);
        expect_error([&] { (void)detail::validate_face68_contract(missing, options); },
                     FaceModelErrorCode::ModelContractMismatch, "landmarks_xyscore");

        auto duplicate = face68_metadata();
        duplicate.push_back(duplicate[1]);
        expect_error([&] { (void)detail::validate_face68_contract(duplicate, options); },
                     FaceModelErrorCode::ModelContractMismatch, "duplicate");

        auto extra = face68_metadata();
        extra.push_back(output("other", { -1, 1 }));
        expect_error([&] { (void)detail::validate_face68_contract(extra, options); },
                     FaceModelErrorCode::ModelContractMismatch, "other");
    }

    it("rejects wrong modes scalar types ranks and fixed dimensions")
    {
        AgeGenderOptions options;
        options.max_batch = 4;

        auto wrong_mode = age_gender_metadata();
        wrong_mode[0].mode = TensorIoMode::Output;
        expect_error([&] { (void)detail::validate_age_gender_contract(wrong_mode, options); },
                     FaceModelErrorCode::ModelContractMismatch, "mode");

        auto wrong_type = age_gender_metadata();
        wrong_type[1].data_type = DataType::Float16;
        expect_error([&] { (void)detail::validate_age_gender_contract(wrong_type, options); },
                     FaceModelErrorCode::ModelContractMismatch, "FP32");

        auto wrong_rank = age_gender_metadata();
        wrong_rank[1].declared_shape = { -1, 1, 2 };
        expect_error([&] { (void)detail::validate_age_gender_contract(wrong_rank, options); },
                     FaceModelErrorCode::ModelContractMismatch, "rank");

        auto wrong_width = age_gender_metadata();
        wrong_width[0].declared_shape[3] = 256;
        expect_error([&] { (void)detail::validate_age_gender_contract(wrong_width, options); },
                     FaceModelErrorCode::ModelContractMismatch, "224");
    }

    it("accepts fixed or dynamic batch and intersects profile and adapter bounds")
    {
        ArcFaceOptions options;
        options.max_batch = 3;
        const detail::SingleOutputContract dynamic =
            detail::validate_arcface_contract(arcface_metadata(), options);
        check(dynamic.batch.minimum == std::size_t { 1 });
        check(dynamic.batch.maximum == std::size_t { 3 });

        auto fixed = arcface_metadata();
        fixed[1].declared_shape[0] = 2;
        const detail::SingleOutputContract fixed_contract =
            detail::validate_arcface_contract(fixed, options);
        check(fixed_contract.batch.minimum == std::size_t { 2 });
        check(fixed_contract.batch.maximum == std::size_t { 2 });

        options.max_batch = 1;
        expect_error([&] { (void)detail::validate_arcface_contract(fixed, options); },
                     FaceModelErrorCode::ResourceLimitExceeded, "max_batch");
    }

    it("requires only batch to be dynamic and exact output declared shapes")
    {
        ArcFaceOptions options;
        options.max_batch = 4;
        auto dynamic_channel = arcface_metadata();
        dynamic_channel[0].declared_shape[1] = -1;
        expect_error([&] { (void)detail::validate_arcface_contract(dynamic_channel, options); },
                     FaceModelErrorCode::ModelContractMismatch, "channel");

        auto dynamic_width = arcface_metadata();
        dynamic_width[1].declared_shape[1] = -1;
        expect_error([&] { (void)detail::validate_arcface_contract(dynamic_width, options); },
                     FaceModelErrorCode::ModelContractMismatch, "512");
    }

    it("enforces profile maximum and bounded checked output storage")
    {
        Face68Options options;
        options.max_batch = 8;
        options.engine.max_output_bytes = 4U * 68U * 3U * sizeof(float);
        expect_error([&] { (void)detail::validate_face68_contract(face68_metadata(), options); },
                     FaceModelErrorCode::ResourceLimitExceeded, "output bytes");

        options.engine.max_output_bytes = 8U * 1024U * 1024U;
        const detail::Face68Contract contract =
            detail::validate_face68_contract(face68_metadata(), options);
        check(contract.batch.maximum == std::size_t { 4 });
        check(contract.landmark_float_capacity == std::size_t { 4U * 68U * 3U });
        check(contract.heatmap_float_capacity == std::size_t { 4U * 68U * 64U * 64U });
    }

    it("validates prepared input views before runtime execution")
    {
        ArcFaceOptions options;
        options.max_batch = 2;
        const detail::SingleOutputContract contract =
            detail::validate_arcface_contract(arcface_metadata(), options);
        float values[3 * 112 * 112] {};
        TensorView view { "input.1", DataType::Float32, { 1, 3, 112, 112 }, values,
                          sizeof(values), MemoryKind::Host };
        detail::validate_prepared_input(view, contract.input_name, contract.batch,
                                        { 3, 112, 112 }, "ArcFace");

        view.name = "input";
        expect_error(
            [&]
            {
                detail::validate_prepared_input(view, contract.input_name, contract.batch,
                                                { 3, 112, 112 }, "ArcFace");
            },
            FaceModelErrorCode::InvalidTensorView, "input.1");
    }

    it("classifies a non-positive prepared batch as an invalid caller view")
    {
        ArcFaceOptions options;
        options.max_batch = 2;
        const detail::SingleOutputContract contract =
            detail::validate_arcface_contract(arcface_metadata(), options);
        float values[3 * 112 * 112] {};
        TensorView view { "input.1", DataType::Float32, { 0, 3, 112, 112 }, values,
                          sizeof(values), MemoryKind::Host };

        expect_error(
            [&]
            {
                detail::validate_prepared_input(view, contract.input_name, contract.batch,
                                                { 3, 112, 112 }, "ArcFace");
            },
            FaceModelErrorCode::InvalidTensorView, "positive");
    }
}
