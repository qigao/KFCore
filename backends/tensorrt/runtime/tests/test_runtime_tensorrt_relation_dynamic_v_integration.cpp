#include "kfcore/tensorrt/runtime.hpp"
#include "tinytest.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kfcore::tensorrt;

namespace
{

std::filesystem::path required_engine_path()
{
    constexpr char kVariable[] =
        "KFCORE_RUNTIME_TENSORRT_TEST_ENGINE_RELATION";
    const char* value = std::getenv(kVariable);
    if (value == nullptr || *value == '\0')
    {
        throw std::runtime_error(
            std::string(kVariable) +
            " must name a trusted dynamic-vocabulary relation engine");
    }
    return std::filesystem::path(value);
}

const TensorDescriptor& tensor_named(
    const std::vector<TensorDescriptor>& tensors,
    const char* name,
    TensorIoMode mode)
{
    for (const auto& tensor : tensors)
    {
        if (tensor.name == name && tensor.mode == mode)
        {
            return tensor;
        }
    }
    throw std::runtime_error(
        std::string("relation engine lacks tensor: ") + name);
}

std::size_t element_count(const TensorShape& shape)
{
    std::size_t count = 1U;
    for (const auto extent : shape)
    {
        if (extent <= 0)
        {
            throw std::runtime_error(
                "runtime tensor shape must be positive");
        }
        count *= static_cast<std::size_t>(extent);
    }
    return count;
}

std::size_t scalar_bytes(DataType type)
{
    switch (type)
    {
    case DataType::Float32:
        return sizeof(float);
    case DataType::Int64:
        return sizeof(std::int64_t);
    case DataType::Bool:
        return sizeof(std::uint8_t);
    default:
        throw std::runtime_error(
            "relation integration test encountered unsupported scalar type");
    }
}

TensorShape fixed_shape(const TensorDescriptor& tensor)
{
    TensorShape result = tensor.declared_shape;
    for (auto extent : result)
    {
        if (extent <= 0)
        {
            throw std::runtime_error(
                "expected fixed relation tensor shape");
        }
    }
    return result;
}

std::vector<std::int64_t> vocabulary_sizes(
    const TensorDescriptor& bank)
{
    if (!bank.profile.has_value())
    {
        throw std::runtime_error(
            "dynamic W input is missing TensorRT profile bounds");
    }
    const auto& profile = *bank.profile;
    if (
        profile.minimum.size() != 2U ||
        profile.optimum.size() != 2U ||
        profile.maximum.size() != 2U)
    {
        throw std::runtime_error(
            "dynamic W profile must be rank 2");
    }

    std::vector<std::int64_t> result {
        profile.minimum[0],
        profile.optimum[0],
        profile.maximum[0],
    };
    std::sort(result.begin(), result.end());
    result.erase(
        std::unique(result.begin(), result.end()),
        result.end());

    if (
        result.empty() ||
        result.front() <= 0 ||
        result.back() < result.front())
    {
        throw std::runtime_error(
            "dynamic W profile has invalid vocabulary bounds");
    }
    return result;
}

} // namespace

spec("TensorRT dynamic-vocabulary relation integration")
{
    it("reuses one engine and executor across vocabulary profile sizes")
    {
        auto engine = Engine::load(required_engine_path());
        const auto& tensors = engine->tensors();

        const auto& image = tensor_named(
            tensors, "image", TensorIoMode::Input);
        const auto& boxes = tensor_named(
            tensors, "boxes", TensorIoMode::Input);
        const auto& box_counts = tensor_named(
            tensors, "box_counts", TensorIoMode::Input);
        const auto& bank = tensor_named(
            tensors, "W", TensorIoMode::Input);
        const auto& alpha = tensor_named(
            tensors, "alpha", TensorIoMode::Input);

        const auto& pred = tensor_named(
            tensors, "pred_logits", TensorIoMode::Output);
        const auto& pair = tensor_named(
            tensors, "pair_logits", TensorIoMode::Output);
        const auto& sub = tensor_named(
            tensors, "sub_idx", TensorIoMode::Output);
        const auto& obj = tensor_named(
            tensors, "obj_idx", TensorIoMode::Output);
        const auto& valid = tensor_named(
            tensors, "valid_mask", TensorIoMode::Output);

        check(image.data_type == DataType::Float32);
        check(boxes.data_type == DataType::Float32);
        check(box_counts.data_type == DataType::Int64);
        check(bank.data_type == DataType::Float32);
        check(alpha.data_type == DataType::Float32);
        check(pred.data_type == DataType::Float32);
        check(pair.data_type == DataType::Float32);
        check(sub.data_type == DataType::Int64);
        check(obj.data_type == DataType::Int64);
        check(valid.data_type == DataType::Bool);

        check(bank.declared_shape.size() == 2U);
        check(bank.declared_shape[0] == -1);
        check(bank.declared_shape[1] > 0);
        check(alpha.declared_shape == TensorShape{-1});
        check(pred.declared_shape.size() == 3U);
        check(pred.declared_shape[0] == 1);
        check(pred.declared_shape[1] > 0);
        check(pred.declared_shape[2] == -1);

        const TensorShape image_shape = fixed_shape(image);
        const TensorShape boxes_shape = fixed_shape(boxes);
        const TensorShape count_shape = fixed_shape(box_counts);
        const TensorShape pair_shape = fixed_shape(pair);

        check(image_shape.size() == 4U);
        check(image_shape[0] == 1);
        check(image_shape[1] == 3);
        check(boxes_shape.size() == 3U);
        check(boxes_shape[0] == 1);
        check(boxes_shape[2] == 4);
        check(count_shape == TensorShape{1});
        check(pair_shape.size() == 2U);
        check(pair_shape[0] == 1);
        check(pair_shape[1] == pred.declared_shape[1]);

        const std::int64_t query_dim = bank.declared_shape[1];
        const std::int64_t pair_budget = pred.declared_shape[1];
        const std::int64_t max_boxes = boxes_shape[1];

        std::vector<float> image_values(
            element_count(image_shape),
            0.0F);
        std::vector<float> box_values(
            element_count(boxes_shape),
            0.0F);
        if (max_boxes >= 2)
        {
            box_values[0] = 0.25F;
            box_values[1] = 0.25F;
            box_values[2] = 0.20F;
            box_values[3] = 0.20F;
            box_values[4] = 0.65F;
            box_values[5] = 0.25F;
            box_values[6] = 0.20F;
            box_values[7] = 0.20F;
        }
        const std::int64_t count_value =
            std::min<std::int64_t>(max_boxes, 2);

        auto executor = engine->create_executor();
        const auto sizes = vocabulary_sizes(bank);

        for (const auto vocabulary_size : sizes)
        {
            const TensorShape bank_shape {
                vocabulary_size,
                query_dim,
            };
            const TensorShape alpha_shape {
                vocabulary_size,
            };

            std::vector<float> bank_values(
                static_cast<std::size_t>(
                    vocabulary_size * query_dim),
                0.0F);
            std::vector<float> alpha_values(
                static_cast<std::size_t>(vocabulary_size),
                0.0F);

            for (std::int64_t row = 0;
                 row < vocabulary_size;
                 ++row)
            {
                const std::int64_t column =
                    row % query_dim;
                bank_values[
                    static_cast<std::size_t>(
                        row * query_dim + column)] = 1.0F;
                alpha_values[
                    static_cast<std::size_t>(row)] =
                    static_cast<float>(
                        row % 3) * 0.5F;
            }

            const std::vector<TensorView> inputs {
                {
                    image.name,
                    DataType::Float32,
                    image_shape,
                    image_values.data(),
                    image_values.size() * sizeof(float),
                    MemoryKind::Host,
                },
                {
                    boxes.name,
                    DataType::Float32,
                    boxes_shape,
                    box_values.data(),
                    box_values.size() * sizeof(float),
                    MemoryKind::Host,
                },
                {
                    box_counts.name,
                    DataType::Int64,
                    count_shape,
                    &count_value,
                    sizeof(count_value),
                    MemoryKind::Host,
                },
                {
                    bank.name,
                    DataType::Float32,
                    bank_shape,
                    bank_values.data(),
                    bank_values.size() * sizeof(float),
                    MemoryKind::Host,
                },
                {
                    alpha.name,
                    DataType::Float32,
                    alpha_shape,
                    alpha_values.data(),
                    alpha_values.size() * sizeof(float),
                    MemoryKind::Host,
                },
            };

            const std::size_t pred_bytes =
                static_cast<std::size_t>(
                    pair_budget * vocabulary_size)
                * sizeof(float);
            const std::size_t pair_bytes =
                element_count(pair_shape) * sizeof(float);
            const std::size_t index_bytes =
                element_count(pair_shape) * sizeof(std::int64_t);
            const std::size_t valid_bytes =
                element_count(pair_shape) * sizeof(std::uint8_t);

            const auto outputs = executor->run_dynamic(
                inputs,
                {
                    {
                        pred.name,
                        DataType::Float32,
                        pred_bytes,
                    },
                    {
                        pair.name,
                        DataType::Float32,
                        pair_bytes,
                    },
                    {
                        sub.name,
                        DataType::Int64,
                        index_bytes,
                    },
                    {
                        obj.name,
                        DataType::Int64,
                        index_bytes,
                    },
                    {
                        valid.name,
                        DataType::Bool,
                        valid_bytes,
                    },
                });

            check(outputs.size() == std::size_t{5U});
            check(
                outputs[0].shape ==
                TensorShape{
                    1,
                    pair_budget,
                    vocabulary_size,
                });
            check(
                outputs[1].shape ==
                TensorShape{1, pair_budget});
            check(
                outputs[2].shape ==
                TensorShape{1, pair_budget});
            check(
                outputs[3].shape ==
                TensorShape{1, pair_budget});
            check(
                outputs[4].shape ==
                TensorShape{1, pair_budget});

            const auto logits =
                outputs[0].float32_values();
            check(
                logits.size() ==
                static_cast<std::size_t>(
                    pair_budget * vocabulary_size));
            for (const float value : logits)
            {
                check_true(std::isfinite(value));
            }

            const auto pair_values =
                outputs[1].float32_values();
            for (const float value : pair_values)
            {
                check_true(std::isfinite(value));
            }
        }
    }
}
