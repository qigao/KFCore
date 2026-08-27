#include "contracts.hpp"

#include "kfcore/face_models/error.hpp"
#include "kfcore/face_models/tensorrt.hpp"
#include "tinytest.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
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

std::vector<TensorDescriptor> inswapper_metadata()
{
    return {
        input("target", { 1, 3, 128, 128 }, { 1, 3, 128, 128 }, { 1, 3, 128, 128 },
              { 1, 3, 128, 128 }),
        input("source", { 1, 512 }, { 1, 512 }, { 1, 512 }, { 1, 512 }),
        output("output", { 1, 3, 128, 128 }),
    };
}

std::vector<TensorDescriptor> gfpgan_metadata()
{
    return {
        input("input", { 1, 3, 512, 512 }, { 1, 3, 512, 512 }, { 1, 3, 512, 512 },
              { 1, 3, 512, 512 }),
        output("output", { 1, 3, 512, 512 }),
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

void expect_prepared_view_error(const TensorView& input, const detail::BatchBounds& batch,
                                const std::string& message)
{
    bool threw = false;
    try
    {
        detail::validate_prepared_input(input, "input.1", batch, { 3, 112, 112 },
                                        "ArcFace");
    }
    catch (const FaceModelError& error)
    {
        threw = true;
        const std::string actual(error.what());
        check(error.code() == FaceModelErrorCode::InvalidTensorView);
        check(actual.find("ArcFace") != std::string::npos);
        check(actual.find("input validation stage") != std::string::npos);
        check(actual.find(message) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

spec("strict TensorRT face model contracts")
{
    it("clears borrowed input storage when its binding scope ends")
    {
        float source_value = 1.0f;
        TensorView source { "input", DataType::Float32, { 1, 1 }, &source_value,
                            sizeof(source_value), MemoryKind::CudaDevice };
        TensorView bound { "input", DataType::Float32, { 1, 1 }, nullptr, 0,
                           MemoryKind::Host };

        {
            detail::BorrowedInputGuard guard(bound, source);
            check(bound.data == source.data);
            check(bound.byte_size == source.byte_size);
            check(bound.memory_kind == MemoryKind::CudaDevice);
        }

        check_null(bound.data);
        check(bound.byte_size == std::size_t { 0 });
        check(bound.memory_kind == MemoryKind::Host);
    }

    it("clears borrowed input storage during exception unwinding")
    {
        float source_value = 1.0f;
        TensorView source { "input", DataType::Float32, { 1, 1 }, &source_value,
                            sizeof(source_value), MemoryKind::CudaDevice };
        TensorView bound { "input", DataType::Float32, { 1, 1 }, nullptr, 0,
                           MemoryKind::Host };

        try
        {
            detail::BorrowedInputGuard guard(bound, source);
            throw std::runtime_error("simulated inference failure");
        }
        catch (const std::runtime_error&)
        {
        }

        check_null(bound.data);
        check(bound.byte_size == std::size_t { 0 });
        check(bound.memory_kind == MemoryKind::Host);
    }

    it("restores owned output storage after a borrowed output binding")
    {
        float owned_value = 0.0F;
        float borrowed_value = 1.0F;
        MutableTensorView bound { "output", DataType::Float32, { 1, 1 }, &owned_value,
                                  sizeof(owned_value), MemoryKind::Host };
        const MutableTensorView borrowed { "output", DataType::Float32, { 1, 1 },
                                           &borrowed_value, sizeof(borrowed_value),
                                           MemoryKind::CudaDevice };
        {
            detail::BorrowedOutputGuard guard(bound, borrowed);
            check(bound.data == borrowed.data);
            check(bound.byte_size == borrowed.byte_size);
            check(bound.memory_kind == MemoryKind::CudaDevice);
        }
        check(bound.data == &owned_value);
        check(bound.byte_size == sizeof(owned_value));
        check(bound.memory_kind == MemoryKind::Host);
    }

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

        InSwapperOptions inswapper;
        inswapper.source_input_name = inswapper.target_input_name;
        expect_error([&] { detail::validate_inswapper_options(inswapper); },
                     FaceModelErrorCode::InvalidArgument, "distinct");

        GfpGanOptions gfpgan;
        gfpgan.engine.max_output_bytes = 0;
        expect_error([&] { detail::validate_gfpgan_options(gfpgan); },
                     FaceModelErrorCode::ResourceLimitExceeded, "max_output_bytes");
    }

    it("accepts exact fixed-batch InSwapper and GFPGAN contracts")
    {
        const detail::InSwapperContract inswapper =
            detail::validate_inswapper_contract(inswapper_metadata(), InSwapperOptions {});
        check(inswapper.batch.minimum == std::size_t { 1 });
        check(inswapper.batch.maximum == std::size_t { 1 });
        check(inswapper.output_float_capacity == std::size_t { 3U * 128U * 128U });

        const detail::SingleOutputContract gfpgan =
            detail::validate_gfpgan_contract(gfpgan_metadata(), GfpGanOptions {});
        check(gfpgan.batch.minimum == std::size_t { 1 });
        check(gfpgan.batch.maximum == std::size_t { 1 });
        check(gfpgan.output_float_capacity == std::size_t { 3U * 512U * 512U });
    }

    it("rejects InSwapper missing inputs wrong shapes types and output limits")
    {
        InSwapperOptions options;

        auto missing_source = inswapper_metadata();
        missing_source.erase(missing_source.begin() + 1);
        expect_error(
            [&] { (void)detail::validate_inswapper_contract(missing_source, options); },
            FaceModelErrorCode::ModelContractMismatch, "source");

        auto wrong_source_rank = inswapper_metadata();
        wrong_source_rank[1].declared_shape = { 1, 1, 512 };
        expect_error(
            [&] { (void)detail::validate_inswapper_contract(wrong_source_rank, options); },
            FaceModelErrorCode::ModelContractMismatch, "rank");

        auto wrong_target_type = inswapper_metadata();
        wrong_target_type[0].data_type = DataType::Float16;
        expect_error(
            [&] { (void)detail::validate_inswapper_contract(wrong_target_type, options); },
            FaceModelErrorCode::ModelContractMismatch, "FP32");

        auto dynamic_source_width = inswapper_metadata();
        dynamic_source_width[1].declared_shape[1] = -1;
        expect_error(
            [&] { (void)detail::validate_inswapper_contract(dynamic_source_width, options); },
            FaceModelErrorCode::ModelContractMismatch, "512");

        auto batch_two = inswapper_metadata();
        batch_two[0].declared_shape[0] = 2;
        batch_two[0].profile = TensorProfile { { 2, 3, 128, 128 }, { 2, 3, 128, 128 },
                                               { 2, 3, 128, 128 } };
        batch_two[1].declared_shape[0] = 2;
        batch_two[1].profile = TensorProfile { { 2, 512 }, { 2, 512 }, { 2, 512 } };
        batch_two[2].declared_shape[0] = 2;
        expect_error([&] { (void)detail::validate_inswapper_contract(batch_two, options); },
                     FaceModelErrorCode::ResourceLimitExceeded, "max_batch");

        options.engine.max_output_bytes = 3U * 128U * 128U * sizeof(float) - 1U;
        expect_error([&] { (void)detail::validate_inswapper_contract(inswapper_metadata(), options); },
                     FaceModelErrorCode::ResourceLimitExceeded, "output bytes");
    }

    it("rejects GFPGAN wrong rank dimensions dtype and dynamic non-batch dimensions")
    {
        GfpGanOptions options;

        auto wrong_output_rank = gfpgan_metadata();
        wrong_output_rank[1].declared_shape = { 1, 3, 512 };
        expect_error([&] { (void)detail::validate_gfpgan_contract(wrong_output_rank, options); },
                     FaceModelErrorCode::ModelContractMismatch, "rank");

        auto wrong_extent = gfpgan_metadata();
        wrong_extent[0].declared_shape[3] = 256;
        expect_error([&] { (void)detail::validate_gfpgan_contract(wrong_extent, options); },
                     FaceModelErrorCode::ModelContractMismatch, "512");

        auto wrong_output_type = gfpgan_metadata();
        wrong_output_type[1].data_type = DataType::Float16;
        expect_error([&] { (void)detail::validate_gfpgan_contract(wrong_output_type, options); },
                     FaceModelErrorCode::ModelContractMismatch, "FP32");

        auto dynamic_channel = gfpgan_metadata();
        dynamic_channel[1].declared_shape[1] = -1;
        expect_error([&] { (void)detail::validate_gfpgan_contract(dynamic_channel, options); },
                     FaceModelErrorCode::ModelContractMismatch, "3");

        options.engine.max_output_bytes = 3U * 512U * 512U * sizeof(float) - 1U;
        expect_error([&] { (void)detail::validate_gfpgan_contract(gfpgan_metadata(), options); },
                     FaceModelErrorCode::ResourceLimitExceeded, "output bytes");
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

    it("validates caller-owned output views before runtime execution")
    {
        float values[3 * 2 * 2] {};
        MutableTensorView output { "output", DataType::Float32, { 1, 3, 2, 2 }, values,
                                   sizeof(values), MemoryKind::CudaDevice };
        detail::validate_prepared_output(output, "output", 1, { 3, 2, 2 }, "TestModel");

        output.byte_size = sizeof(values) - 1U;
        expect_error(
            [&]
            {
                detail::validate_prepared_output(output, "output", 1, { 3, 2, 2 },
                                                 "TestModel");
            },
            FaceModelErrorCode::InvalidTensorView, "capacity");
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

    it("rejects every invalid prepared tensor field before runtime execution")
    {
        ArcFaceOptions options;
        options.max_batch = 2;
        const detail::SingleOutputContract contract =
            detail::validate_arcface_contract(arcface_metadata(), options);
        ArcFaceOptions profile_limited_options;
        profile_limited_options.max_batch = 16;
        const detail::SingleOutputContract profile_limited_contract =
            detail::validate_arcface_contract(arcface_metadata(), profile_limited_options);
        float values[3 * 112 * 112] {};
        const TensorView valid { "input.1", DataType::Float32, { 1, 3, 112, 112 }, values,
                                 sizeof(values), MemoryKind::Host };

        TensorView wrong_type = valid;
        wrong_type.data_type = DataType::Float16;
        TensorView wrong_memory = valid;
        wrong_memory.memory_kind = static_cast<MemoryKind>(99);
        TensorView null_data = valid;
        null_data.data = nullptr;
        TensorView wrong_rank = valid;
        wrong_rank.shape = { 1, 3, 112 };
        TensorView wrong_channel = valid;
        wrong_channel.shape[1] = 1;
        TensorView wrong_height = valid;
        wrong_height.shape[2] = 111;
        TensorView wrong_width = valid;
        wrong_width.shape[3] = 111;
        TensorView above_adapter_max = valid;
        above_adapter_max.shape[0] = 3;
        TensorView above_profile_max = valid;
        above_profile_max.shape[0] = 9;
        TensorView undersized = valid;
        undersized.byte_size = sizeof(values) - 1U;

        struct InvalidPreparedViewCase
        {
            const char*         name;
            TensorView          view;
            detail::BatchBounds batch;
            const char*         message;
        };
        const std::array<InvalidPreparedViewCase, 10> cases = {
            InvalidPreparedViewCase { "wrong data type", wrong_type, contract.batch, "FP32" },
            InvalidPreparedViewCase { "invalid memory kind", wrong_memory, contract.batch,
                                      "memory kind" },
            InvalidPreparedViewCase { "null data", null_data, contract.batch, "not be null" },
            InvalidPreparedViewCase { "wrong rank", wrong_rank, contract.batch, "rank" },
            InvalidPreparedViewCase { "wrong channel", wrong_channel, contract.batch,
                                      "fixed dimensions" },
            InvalidPreparedViewCase { "wrong height", wrong_height, contract.batch,
                                      "fixed dimensions" },
            InvalidPreparedViewCase { "wrong width", wrong_width, contract.batch,
                                      "fixed dimensions" },
            InvalidPreparedViewCase { "above adapter maximum", above_adapter_max,
                                      contract.batch, "adapter bounds" },
            InvalidPreparedViewCase { "above profile maximum", above_profile_max,
                                      profile_limited_contract.batch, "adapter bounds" },
            InvalidPreparedViewCase { "undersized byte capacity", undersized, contract.batch,
                                      "capacity" },
        };

        for (const InvalidPreparedViewCase& test_case : cases)
        {
            info("prepared-view case: %s", test_case.name);
            expect_prepared_view_error(test_case.view, test_case.batch, test_case.message);
        }
    }

    it("maps a cleanup-invalidated runtime error to a face runtime failure")
    {
        const TensorRtError runtime_error(
            TensorRtErrorCode::TensorRtFailure,
            "execution stage: executor is invalidated after tensor address cleanup stage: "
            "TensorRT rejected address reset");

        expect_error(
            [&] { detail::rethrow_tensorrt(runtime_error, "ArcFace", "inference"); },
            FaceModelErrorCode::RuntimeFailure, "tensor address cleanup stage");
    }
}
