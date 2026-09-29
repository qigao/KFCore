#include "kfcore/relation/relate_anything.hpp"

#include "decode.hpp"
#include "kfcore/image_processor/cpu.hpp"
#include "kfcore/image_processor/error.hpp"
#include "kfcore/relation/error.hpp"
#include "kfcore/runtime/error.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::relation
{
namespace
{

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw RelationError(RelationErrorCode::InvalidArgument,
                        "RelateAnything: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw RelationError(RelationErrorCode::ModelContractMismatch,
                        "RelateAnything model contract: " + detail);
}

[[noreturn]] void throw_runtime(const std::string& detail)
{
    throw RelationError(RelationErrorCode::RuntimeFailure,
                        "RelateAnything runtime: " + detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw RelationError(RelationErrorCode::ResourceLimitExceeded,
                        "RelateAnything resource limit: " + detail);
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char* subject)
{
    if (left != 0U &&
        right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_resource(std::string(subject) + " byte count overflow");
    }
    return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right,
                        const char* subject)
{
    if (right > (std::numeric_limits<std::size_t>::max)() - left)
    {
        throw_resource(std::string(subject) + " byte count overflow");
    }
    return left + right;
}

void validate_options(const RelateAnythingOptions& options)
{
    if (options.input_size < 0)
    {
        throw_invalid("input_size must be zero or positive");
    }
    if (options.max_boxes == 0U || options.max_pairs == 0U)
    {
        throw_invalid("max_boxes and max_pairs must be positive");
    }
    if (options.predicates.empty())
    {
        throw_invalid("baked-vocabulary inference requires predicate names");
    }
    for (const std::string& predicate : options.predicates)
    {
        if (predicate.empty())
        {
            throw_invalid("predicate names must not be empty");
        }
    }
    if (!std::isfinite(options.threshold) || options.threshold < 0.0F ||
        options.threshold > 1.0F)
    {
        throw_invalid("threshold must be finite within [0,1]");
    }
    if (!std::isfinite(options.pair_weight) ||
        !std::isfinite(options.calibration_a) ||
        !std::isfinite(options.calibration_b) ||
        options.calibration_a <= 0.0F)
    {
        throw_invalid("score fusion parameters must be finite and calibration_a positive");
    }
    if (options.top_k == 0U || options.top_k > options.max_pairs)
    {
        throw_invalid("top_k must be within [1,max_pairs]");
    }
    if (options.max_source_bytes == 0U || options.max_tensor_bytes == 0U ||
        options.max_output_bytes == 0U)
    {
        throw_resource("configured byte limits must be positive");
    }
}

const runtime::TensorDescriptor& require_tensor(
    const std::vector<runtime::TensorDescriptor>& tensors,
    const char* name, bool is_input)
{
    const auto match = std::find_if(tensors.begin(), tensors.end(),
        [&](const runtime::TensorDescriptor& tensor)
        {
            return tensor.name == name && tensor.is_input == is_input;
        });
    if (match == tensors.end())
    {
        throw_contract(std::string("missing ") +
                       (is_input ? "input '" : "output '") + name + "'");
    }
    return *match;
}

void require_dimension(std::int64_t declared, std::size_t expected,
                       const char* subject)
{
    if (declared != -1 &&
        declared != static_cast<std::int64_t>(expected))
    {
        throw_contract(std::string(subject) + " has incompatible extent");
    }
}

void require_batch_one(std::int64_t declared, const char* subject)
{
    if (declared != -1 && declared != 1)
    {
        throw_contract(std::string(subject) + " requires batch size 1");
    }
}

std::int32_t resolve_image_size(const runtime::TensorDescriptor& image,
                                std::int32_t requested)
{
    if (image.data_type != runtime::DataType::Float32 ||
        image.shape.size() != 4U)
    {
        throw_contract("image must be an FP32 NCHW rank-4 tensor");
    }
    require_batch_one(image.shape[0], "image");
    require_dimension(image.shape[1], 3U, "image channels");

    const std::int64_t declared_height = image.shape[2];
    const std::int64_t declared_width = image.shape[3];
    if (declared_height > 0 && declared_width > 0 &&
        declared_height != declared_width)
    {
        throw_contract("image input must be square");
    }

    std::int64_t resolved = 0;
    if (declared_height > 0)
    {
        resolved = declared_height;
    }
    if (declared_width > 0)
    {
        if (resolved != 0 && resolved != declared_width)
        {
            throw_contract("image input has conflicting spatial extents");
        }
        resolved = declared_width;
    }
    if (requested > 0)
    {
        if (resolved != 0 && resolved != requested)
        {
            throw_contract("configured input_size conflicts with graph shape");
        }
        resolved = requested;
    }
    if (resolved <= 0 ||
        resolved > (std::numeric_limits<std::int32_t>::max)())
    {
        throw_contract("dynamic image shape requires an explicit positive input_size");
    }
    return static_cast<std::int32_t>(resolved);
}

void validate_contract(const std::vector<runtime::TensorDescriptor>& tensors,
                       const RelateAnythingOptions& options,
                       std::int32_t& input_size)
{
    std::size_t input_count = 0U;
    std::size_t output_count = 0U;
    for (const runtime::TensorDescriptor& tensor : tensors)
    {
        tensor.is_input ? ++input_count : ++output_count;
    }
    if (input_count != 3U || output_count != 5U)
    {
        throw_contract(
            "PoC requires the baked-vocabulary graph with exactly 3 inputs and 5 outputs");
    }

    const auto& image = require_tensor(tensors, "image", true);
    const auto& boxes = require_tensor(tensors, "boxes", true);
    const auto& box_counts = require_tensor(tensors, "box_counts", true);
    const auto& pred = require_tensor(tensors, "pred_logits", false);
    const auto& pair = require_tensor(tensors, "pair_logits", false);
    const auto& sub = require_tensor(tensors, "sub_idx", false);
    const auto& obj = require_tensor(tensors, "obj_idx", false);
    const auto& valid = require_tensor(tensors, "valid_mask", false);

    input_size = resolve_image_size(image, options.input_size);

    if (boxes.data_type != runtime::DataType::Float32 ||
        boxes.shape.size() != 3U)
    {
        throw_contract("boxes must be FP32 [1,N,4]");
    }
    require_batch_one(boxes.shape[0], "boxes");
    require_dimension(boxes.shape[1], options.max_boxes, "boxes");
    require_dimension(boxes.shape[2], 4U, "boxes coordinate axis");

    if (box_counts.data_type != runtime::DataType::Int64 ||
        box_counts.shape.size() != 1U)
    {
        throw_contract("box_counts must be INT64 [1]");
    }
    require_dimension(box_counts.shape[0], 1U, "box_counts");

    if (pred.data_type != runtime::DataType::Float32 ||
        pred.shape.size() != 3U)
    {
        throw_contract("pred_logits must be FP32 [1,K,V]");
    }
    require_batch_one(pred.shape[0], "pred_logits");
    require_dimension(pred.shape[1], options.max_pairs, "pred_logits pair axis");
    require_dimension(pred.shape[2], options.predicates.size(),
                      "pred_logits predicate axis");

    const auto validate_pair_output =
        [&](const runtime::TensorDescriptor& tensor,
            runtime::DataType data_type, const char* name)
        {
            if (tensor.data_type != data_type || tensor.shape.size() != 2U)
            {
                throw_contract(std::string(name) + " has incompatible type or rank");
            }
            require_batch_one(tensor.shape[0], name);
            require_dimension(tensor.shape[1], options.max_pairs, name);
        };

    validate_pair_output(pair, runtime::DataType::Float32, "pair_logits");
    validate_pair_output(sub, runtime::DataType::Int64, "sub_idx");
    validate_pair_output(obj, runtime::DataType::Int64, "obj_idx");
    validate_pair_output(valid, runtime::DataType::Bool, "valid_mask");
}

void validate_regions(const std::vector<Region>& regions,
                      std::int32_t image_width, std::int32_t image_height,
                      std::size_t max_boxes)
{
    if (regions.size() > max_boxes)
    {
        throw_invalid("region count exceeds configured max_boxes");
    }

    const float width = static_cast<float>(image_width);
    const float height = static_cast<float>(image_height);
    for (const Region& region : regions)
    {
        if (!std::isfinite(region.left) || !std::isfinite(region.top) ||
            !std::isfinite(region.right) || !std::isfinite(region.bottom) ||
            region.left < 0.0F || region.top < 0.0F ||
            region.right > width || region.bottom > height ||
            region.right <= region.left || region.bottom <= region.top)
        {
            throw_invalid("regions must be finite positive boxes inside the source image");
        }
        if (!std::isfinite(region.detector_score) ||
            region.detector_score < 0.0F || region.detector_score > 1.0F)
        {
            throw_invalid("detector scores must be finite within [0,1]");
        }
    }
}

class UseGuard final
{
public:
    explicit UseGuard(std::atomic_flag& flag)
        : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw_invalid("calls on one relation model instance must not overlap");
        }
    }

    ~UseGuard()
    {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag& flag_;
};

} // namespace

struct RelateAnything::Impl final
{
    Impl(runtime::ResolvedModel resolved_value,
         RelateAnythingOptions options_value)
        : resolved(std::move(resolved_value))
        , options(std::move(options_value))
        , context(resolved.model->create_context())
    {
        validate_contract(resolved.model->tensors(), options, input_size_value);

        const std::size_t pred_elements =
            checked_multiply(options.max_pairs, options.predicates.size(),
                             "predicate output");

        std::size_t output_bytes =
            checked_multiply(pred_elements, sizeof(float), "predicate output");
        output_bytes = checked_add(
            output_bytes,
            checked_multiply(options.max_pairs, sizeof(float), "pair output"),
            "relation outputs");
        output_bytes = checked_add(
            output_bytes,
            checked_multiply(options.max_pairs, sizeof(std::int64_t) * 2U,
                             "index outputs"),
            "relation outputs");
        output_bytes = checked_add(
            output_bytes,
            checked_multiply(options.max_pairs, sizeof(std::uint8_t),
                             "valid output"),
            "relation outputs");
        if (output_bytes > options.max_output_bytes)
        {
            throw_resource("relation outputs exceed configured output byte limit");
        }

        pred_logits.resize(pred_elements);
        pair_logits.resize(options.max_pairs);
        subject_indices.resize(options.max_pairs);
        object_indices.resize(options.max_pairs);
        valid_mask.resize(options.max_pairs);
    }

    runtime::ResolvedModel resolved;
    RelateAnythingOptions options;
    std::unique_ptr<runtime::ExecutionContext> context;
    std::int32_t input_size_value = 0;
    std::vector<float> pred_logits;
    std::vector<float> pair_logits;
    std::vector<std::int64_t> subject_indices;
    std::vector<std::int64_t> object_indices;
    std::vector<std::uint8_t> valid_mask;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

RelateAnything::RelateAnything(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

RelateAnything::~RelateAnything() = default;

std::unique_ptr<RelateAnything>
RelateAnything::load(runtime::Runtime& runtime,
                     const runtime::ModelPackage& package,
                     const runtime::ExecutionPolicy& policy,
                     const RelateAnythingOptions& options)
{
    validate_options(options);
    if (package.model_type() != kRelateAnythingModelType)
    {
        throw_contract("ModelPackage model_type must be 'relation.relate-anything'");
    }

    try
    {
        runtime::ResolvedModel resolved = runtime.load_model(package, policy);
        return std::unique_ptr<RelateAnything>(
            new RelateAnything(
                std::make_unique<Impl>(std::move(resolved), options)));
    }
    catch (const RelationError&)
    {
        throw;
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
    catch (const image::ImageProcessorError& error)
    {
        if (error.code() ==
            image::ImageProcessorErrorCode::ResourceLimitExceeded)
        {
            throw_resource(error.what());
        }
        throw_invalid(error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("model allocation failed");
    }
}

RelationFrame RelateAnything::infer(const image::ImageView& image,
                                    const std::vector<Region>& regions)
{
    if (!impl_)
    {
        throw_invalid("model state is unavailable");
    }

    UseGuard guard(impl_->in_use);
    try
    {
        const image::BgrImage source =
            image::CpuImageProcessor::copy_bgr(
                image, impl_->options.max_source_bytes);
        validate_regions(regions, source.width, source.height,
                         impl_->options.max_boxes);

        const image::BgrImage resized =
            image::CpuImageProcessor::resize_bgr(
                source, impl_->input_size_value, impl_->input_size_value,
                impl_->options.max_source_bytes);

        image::PreprocessOptions preprocess;
        preprocess.output_format = image::PixelFormat::Rgb8;
        preprocess.mean = {0.0F, 0.0F, 0.0F};
        preprocess.stddev = {1.0F, 1.0F, 1.0F};
        const std::vector<float> image_tensor =
            image::CpuImageProcessor::to_nchw(
                resized, preprocess, impl_->options.max_tensor_bytes);

        const std::size_t expected_image_values = checked_multiply(
            checked_multiply(
                static_cast<std::size_t>(impl_->input_size_value),
                static_cast<std::size_t>(impl_->input_size_value),
                "image tensor"),
            3U, "image tensor");
        if (image_tensor.size() != expected_image_values)
        {
            throw_contract("preprocessed image tensor has unexpected size");
        }

        const std::size_t box_values =
            checked_multiply(impl_->options.max_boxes, 4U, "boxes tensor");
        std::vector<float> boxes(box_values, 0.0F);
        const float image_width = static_cast<float>(source.width);
        const float image_height = static_cast<float>(source.height);
        for (std::size_t index = 0U; index < regions.size(); ++index)
        {
            const Region& region = regions[index];
            const float left = region.left / image_width;
            const float top = region.top / image_height;
            const float right = region.right / image_width;
            const float bottom = region.bottom / image_height;
            boxes[index * 4U + 0U] = (left + right) * 0.5F;
            boxes[index * 4U + 1U] = (top + bottom) * 0.5F;
            boxes[index * 4U + 2U] = right - left;
            boxes[index * 4U + 3U] = bottom - top;
        }

        const std::int64_t box_count =
            static_cast<std::int64_t>(regions.size());
        const runtime::TensorShape image_shape {
            1, 3, impl_->input_size_value, impl_->input_size_value
        };
        const runtime::TensorShape boxes_shape {
            1, static_cast<std::int64_t>(impl_->options.max_boxes), 4
        };
        const runtime::TensorShape count_shape {1};
        const runtime::TensorShape pred_shape {
            1,
            static_cast<std::int64_t>(impl_->options.max_pairs),
            static_cast<std::int64_t>(impl_->options.predicates.size())
        };
        const runtime::TensorShape pair_shape {
            1, static_cast<std::int64_t>(impl_->options.max_pairs)
        };

        const std::size_t image_bytes =
            checked_multiply(image_tensor.size(), sizeof(float), "image tensor");
        const std::size_t boxes_bytes =
            checked_multiply(boxes.size(), sizeof(float), "boxes tensor");
        if (image_bytes > impl_->options.max_tensor_bytes ||
            boxes_bytes > impl_->options.max_tensor_bytes)
        {
            throw_resource("input tensor exceeds configured tensor byte limit");
        }

        std::vector<runtime::TensorView> inputs {
            {"image", runtime::DataType::Float32, image_shape,
             image_tensor.data(), image_bytes, runtime::MemoryKind::Host, {}},
            {"boxes", runtime::DataType::Float32, boxes_shape,
             boxes.data(), boxes_bytes, runtime::MemoryKind::Host, {}},
            {"box_counts", runtime::DataType::Int64, count_shape,
             &box_count, sizeof(box_count), runtime::MemoryKind::Host, {}},
        };
        std::vector<runtime::MutableTensorView> outputs {
            {"pred_logits", runtime::DataType::Float32, pred_shape,
             impl_->pred_logits.data(),
             impl_->pred_logits.size() * sizeof(float),
             runtime::MemoryKind::Host, {}},
            {"pair_logits", runtime::DataType::Float32, pair_shape,
             impl_->pair_logits.data(),
             impl_->pair_logits.size() * sizeof(float),
             runtime::MemoryKind::Host, {}},
            {"sub_idx", runtime::DataType::Int64, pair_shape,
             impl_->subject_indices.data(),
             impl_->subject_indices.size() * sizeof(std::int64_t),
             runtime::MemoryKind::Host, {}},
            {"obj_idx", runtime::DataType::Int64, pair_shape,
             impl_->object_indices.data(),
             impl_->object_indices.size() * sizeof(std::int64_t),
             runtime::MemoryKind::Host, {}},
            {"valid_mask", runtime::DataType::Bool, pair_shape,
             impl_->valid_mask.data(),
             impl_->valid_mask.size() * sizeof(std::uint8_t),
             runtime::MemoryKind::Host, {}},
        };
        impl_->context->run(inputs, outputs);

        detail::RawRelationOutputs raw;
        raw.pred_logits = impl_->pred_logits.data();
        raw.pair_logits = impl_->pair_logits.data();
        raw.subject_indices = impl_->subject_indices.data();
        raw.object_indices = impl_->object_indices.data();
        raw.valid_mask = impl_->valid_mask.data();
        raw.pair_count = impl_->options.max_pairs;
        raw.predicate_count = impl_->options.predicates.size();

        RelationFrame result;
        result.image_width = source.width;
        result.image_height = source.height;
        result.edges =
            detail::decode_relation_outputs(raw, regions, impl_->options);
        return result;
    }
    catch (const RelationError&)
    {
        throw;
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
    catch (const image::ImageProcessorError& error)
    {
        if (error.code() ==
            image::ImageProcessorErrorCode::ResourceLimitExceeded)
        {
            throw_resource(error.what());
        }
        throw_invalid(error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("inference allocation failed");
    }
}

std::int32_t RelateAnything::input_size() const noexcept
{
    return impl_ ? impl_->input_size_value : 0;
}

std::size_t RelateAnything::max_boxes() const noexcept
{
    return impl_ ? impl_->options.max_boxes : 0U;
}

std::size_t RelateAnything::max_pairs() const noexcept
{
    return impl_ ? impl_->options.max_pairs : 0U;
}

const std::vector<std::string>& RelateAnything::predicates() const noexcept
{
    static const std::vector<std::string> empty;
    return impl_ ? impl_->options.predicates : empty;
}

const runtime::ExecutionRoute& RelateAnything::execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty {};
    return impl_ ? impl_->resolved.route : empty;
}

} // namespace kfcore::relation
