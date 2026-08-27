#include "tensor_validation.hpp"
#include "kfcore/tensorrt/error.hpp"
#include "tinytest.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

using namespace kfcore::tensorrt;

namespace
{

TensorDescriptor input_descriptor(std::string name = "input")
{
    return { std::move(name),
             TensorIoMode::Input,
             DataType::Float32,
             { { 1, 3, 112, 112 }, { 2, 3, 112, 112 }, { 4, 3, 112, 112 } } };
}

TensorDescriptor output_descriptor(std::string name = "output")
{
    return { std::move(name),
             TensorIoMode::Output,
             DataType::Float32,
             { { 1, 512 }, { 2, 512 }, { 4, 512 } } };
}

void check_error(const std::function<void()>& operation, TensorRtErrorCode code,
                 const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const TensorRtError& error)
    {
        threw = true;
        check(error.code() == code);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

} // namespace

spec("TensorRT runtime tensor validation")
{
    it("reports the byte width of every supported scalar type")
    {
        check(detail::scalar_byte_size(DataType::Float32) == std::size_t { 4 });
        check(detail::scalar_byte_size(DataType::Float16) == std::size_t { 2 });
        check(detail::scalar_byte_size(DataType::Int8) == std::size_t { 1 });
        check(detail::scalar_byte_size(DataType::Int32) == std::size_t { 4 });
        check(detail::scalar_byte_size(DataType::Bool) == std::size_t { 1 });
        check(detail::scalar_byte_size(DataType::UInt8) == std::size_t { 1 });
        check(detail::scalar_byte_size(DataType::BFloat16) == std::size_t { 2 });
        check(detail::scalar_byte_size(DataType::Int64) == std::size_t { 8 });
    }

    it("computes tensor bytes with checked shape multiplication")
    {
        check(detail::checked_shape_byte_size({ 2, 3, 4 }, DataType::Float32, "shape") ==
              std::size_t { 96 });
        check_error(
            []
            {
                (void)detail::checked_shape_byte_size(
                    { (std::numeric_limits<std::int64_t>::max)(), 3 }, DataType::Int8, "shape");
            },
            TensorRtErrorCode::ResourceLimitExceeded, "overflow");
    }

    it("accepts only positive ordered profile bounds")
    {
        detail::validate_profile_bounds(input_descriptor().profile, "input");

        TensorProfile unordered = input_descriptor().profile;
        unordered.minimum[0]    = 3;
        check_error([&] { detail::validate_profile_bounds(unordered, "input"); },
                    TensorRtErrorCode::EngineContractMismatch, "min <= opt <= max");

        TensorProfile zero = input_descriptor().profile;
        zero.minimum[0]    = 0;
        check_error([&] { detail::validate_profile_bounds(zero, "input"); },
                    TensorRtErrorCode::EngineContractMismatch, "positive");
    }

    it("rejects missing duplicate and extra input views")
    {
        const std::vector<TensorDescriptor> expected = { input_descriptor("image"),
                                                         input_descriptor("aux") };
        float                               values[3 * 112 * 112] {};
        const TensorView                    image { "image", DataType::Float32, { 1, 3, 112, 112 },
                                 values,  sizeof(values),    MemoryKind::Host };

        check_error([&] { detail::validate_input_views(expected, { image }, 1U << 20U); },
                    TensorRtErrorCode::InvalidTensorView, "missing");
        check_error([&] { detail::validate_input_views(expected, { image, image }, 1U << 20U); },
                    TensorRtErrorCode::InvalidTensorView, "duplicate");
        TensorView extra = image;
        extra.name       = "other";
        check_error([&] { detail::validate_input_views(expected, { image, extra }, 1U << 20U); },
                    TensorRtErrorCode::InvalidTensorView, "unexpected");
    }

    it("enforces aggregate maximum input and output capacities")
    {
        EngineOptions options;
        options.max_serialized_engine_bytes = 1;
        options.max_tensor_count            = 2;
        options.max_input_bytes             = 100;
        options.max_output_bytes            = 100;

        check_error([&] { detail::validate_tensor_metadata({ input_descriptor() }, options); },
                    TensorRtErrorCode::ResourceLimitExceeded, "input bytes");
        check_error([&] { detail::validate_tensor_metadata({ output_descriptor() }, options); },
                    TensorRtErrorCode::ResourceLimitExceeded, "output bytes");
    }
}
