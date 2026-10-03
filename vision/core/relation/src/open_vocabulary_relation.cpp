#include "kfcore/relation/open_vocabulary_relation.hpp"

#include "decode.hpp"
#include "vocabulary_score.hpp"
#include "kfcore/image_processor/cpu.hpp"
#include "kfcore/image_processor/error.hpp"
#include "kfcore/relation/error.hpp"
#include "kfcore/relation/relate_anything.hpp"
#include "kfcore/runtime/error.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kfcore::relation
{
namespace
{

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw RelationError(RelationErrorCode::InvalidArgument,
                        "OpenVocabularyRelation: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw RelationError(RelationErrorCode::ModelContractMismatch,
                        "OpenVocabularyRelation model contract: " + detail);
}

[[noreturn]] void throw_runtime(const std::string& detail)
{
    throw RelationError(RelationErrorCode::RuntimeFailure,
                        "OpenVocabularyRelation runtime: " + detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw RelationError(RelationErrorCode::ResourceLimitExceeded,
                        "OpenVocabularyRelation resource limit: " + detail);
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

void validate_options(const OpenVocabularyRelationOptions& options)
{
    if (options.input_size < 0)
    {
        throw_invalid("input_size must be zero or positive");
    }
    if (options.max_boxes == 0U || options.max_pairs == 0U ||
        options.query_dim == 0U)
    {
        throw_invalid("max_boxes, max_pairs and query_dim must be positive");
    }
    if (!std::isfinite(options.logit_scale) ||
        options.logit_scale <= 0.0F ||
        !std::isfinite(options.logit_bias))
    {
        throw_invalid("logit scale/bias contract is invalid");
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
        throw_invalid("score fusion parameters are invalid");
    }
    if (options.top_k == 0U || options.top_k > options.max_pairs)
    {
        throw_invalid("top_k must be within [1,max_pairs]");
    }
    if (options.max_source_bytes == 0U ||
        options.max_tensor_bytes == 0U ||
        options.max_output_bytes == 0U ||
        options.max_vocabulary_bytes == 0U)
    {
        throw_resource("configured byte limits must be positive");
    }
}

const runtime::TensorDescriptor& require_tensor(
    const std::vector<runtime::TensorDescriptor>& tensors,
    const char* name, bool is_input)
{
    const auto match = std::find_if(
        tensors.begin(), tensors.end(),
        [&](const runtime::TensorDescriptor& tensor)
        {
            return tensor.name == name && tensor.is_input == is_input;
        });
    if (match == tensors.end())
    {
        throw_contract(std::string("missing ") +
                       (is_input ? "input '" : "output '") +
                       name + "'");
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

std::int32_t resolve_image_size(
    const runtime::TensorDescriptor& image, std::int32_t requested)
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

    std::int64_t resolved = declared_height > 0 ? declared_height : 0;
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
        throw_contract(
            "dynamic image shape requires an explicit positive input_size");
    }
    return static_cast<std::int32_t>(resolved);
}

void validate_query_tensor(const runtime::TensorDescriptor& tensor,
                           const OpenVocabularyRelationOptions& options,
                           const char* name)
{
    if (tensor.data_type != runtime::DataType::Float32 ||
        tensor.shape.size() != 3U)
    {
        throw_contract(std::string(name) + " must be FP32 [1,K,D]");
    }
    require_batch_one(tensor.shape[0], name);
    require_dimension(tensor.shape[1], options.max_pairs, name);
    require_dimension(tensor.shape[2], options.query_dim, name);
}

void validate_pair_tensor(const runtime::TensorDescriptor& tensor,
                          const OpenVocabularyRelationOptions& options,
                          runtime::DataType type, const char* name)
{
    if (tensor.data_type != type || tensor.shape.size() != 2U)
    {
        throw_contract(std::string(name) + " has incompatible type or rank");
    }
    require_batch_one(tensor.shape[0], name);
    require_dimension(tensor.shape[1], options.max_pairs, name);
}

enum class ScoringMode
{
    HostQueries,
    BackendLogits,
};

void require_dynamic_dimension(
    std::int64_t declared,
    const char* subject)
{
    if (declared != -1)
    {
        throw_contract(
            std::string(subject) +
            " must use a dynamic vocabulary extent");
    }
}

void validate_common_inputs(
    const std::vector<runtime::TensorDescriptor>& tensors,
    const OpenVocabularyRelationOptions& options,
    std::int32_t& input_size)
{
    const auto& image = require_tensor(tensors, "image", true);
    const auto& boxes = require_tensor(tensors, "boxes", true);
    const auto& box_counts = require_tensor(tensors, "box_counts", true);

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
}

void validate_encoder_contract(
    const std::vector<runtime::TensorDescriptor>& tensors,
    const OpenVocabularyRelationOptions& options,
    std::int32_t& input_size)
{
    std::size_t input_count = 0U;
    std::size_t output_count = 0U;
    for (const auto& tensor : tensors)
    {
        tensor.is_input ? ++input_count : ++output_count;
    }
    if (input_count != 3U || output_count != 6U)
    {
        throw_contract(
            "encoder requires exactly 3 inputs and 6 outputs");
    }

    validate_common_inputs(tensors, options, input_size);

    const auto& semantic =
        require_tensor(tensors, "semantic_query", false);
    const auto& spatial =
        require_tensor(tensors, "spatial_query", false);
    const auto& pair = require_tensor(tensors, "pair_logits", false);
    const auto& sub = require_tensor(tensors, "sub_idx", false);
    const auto& obj = require_tensor(tensors, "obj_idx", false);
    const auto& valid = require_tensor(tensors, "valid_mask", false);

    validate_query_tensor(semantic, options, "semantic_query");
    validate_query_tensor(spatial, options, "spatial_query");
    validate_pair_tensor(
        pair, options, runtime::DataType::Float32, "pair_logits");
    validate_pair_tensor(
        sub, options, runtime::DataType::Int64, "sub_idx");
    validate_pair_tensor(
        obj, options, runtime::DataType::Int64, "obj_idx");
    validate_pair_tensor(
        valid, options, runtime::DataType::Bool, "valid_mask");
}

void validate_backend_contract(
    const std::vector<runtime::TensorDescriptor>& tensors,
    const OpenVocabularyRelationOptions& options,
    std::int32_t& input_size)
{
    std::size_t input_count = 0U;
    std::size_t output_count = 0U;
    for (const auto& tensor : tensors)
    {
        tensor.is_input ? ++input_count : ++output_count;
    }
    if (input_count != 5U || output_count != 5U)
    {
        throw_contract(
            "backend-scoring graph requires exactly 5 inputs and 5 outputs");
    }

    validate_common_inputs(tensors, options, input_size);

    const auto& bank = require_tensor(tensors, "W", true);
    const auto& alpha = require_tensor(tensors, "alpha", true);
    const auto& pred = require_tensor(tensors, "pred_logits", false);
    const auto& pair = require_tensor(tensors, "pair_logits", false);
    const auto& sub = require_tensor(tensors, "sub_idx", false);
    const auto& obj = require_tensor(tensors, "obj_idx", false);
    const auto& valid = require_tensor(tensors, "valid_mask", false);

    if (bank.data_type != runtime::DataType::Float32 ||
        bank.shape.size() != 2U)
    {
        throw_contract("W must be FP32 [V,D]");
    }
    require_dynamic_dimension(bank.shape[0], "W vocabulary axis");
    require_dimension(bank.shape[1], options.query_dim, "W embedding axis");

    if (alpha.data_type != runtime::DataType::Float32 ||
        alpha.shape.size() != 1U)
    {
        throw_contract("alpha must be FP32 [V]");
    }
    require_dynamic_dimension(alpha.shape[0], "alpha vocabulary axis");

    if (pred.data_type != runtime::DataType::Float32 ||
        pred.shape.size() != 3U)
    {
        throw_contract("pred_logits must be FP32 [1,K,V]");
    }
    require_batch_one(pred.shape[0], "pred_logits");
    require_dimension(pred.shape[1], options.max_pairs, "pred_logits pair axis");
    require_dynamic_dimension(pred.shape[2], "pred_logits vocabulary axis");

    validate_pair_tensor(
        pair, options, runtime::DataType::Float32, "pair_logits");
    validate_pair_tensor(
        sub, options, runtime::DataType::Int64, "sub_idx");
    validate_pair_tensor(
        obj, options, runtime::DataType::Int64, "obj_idx");
    validate_pair_tensor(
        valid, options, runtime::DataType::Bool, "valid_mask");
}


void validate_regions(const std::vector<Region>& regions,
                      std::int32_t image_width,
                      std::int32_t image_height,
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
        if (!std::isfinite(region.left) ||
            !std::isfinite(region.top) ||
            !std::isfinite(region.right) ||
            !std::isfinite(region.bottom) ||
            region.left < 0.0F || region.top < 0.0F ||
            region.right > width || region.bottom > height ||
            region.right <= region.left || region.bottom <= region.top)
        {
            throw_invalid(
                "regions must be finite positive boxes inside the source image");
        }
        if (!std::isfinite(region.detector_score) ||
            region.detector_score < 0.0F ||
            region.detector_score > 1.0F)
        {
            throw_invalid(
                "detector scores must be finite within [0,1]");
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

RelateAnythingOptions decode_options(
    const OpenVocabularyRelationOptions& options,
    const PredicateVocabulary& vocabulary)
{
    RelateAnythingOptions result;
    result.max_boxes = options.max_boxes;
    result.max_pairs = options.max_pairs;
    result.predicates = vocabulary.predicates;
    result.threshold = options.threshold;
    result.pair_weight = options.pair_weight;
    result.calibration_a = options.calibration_a;
    result.calibration_b = options.calibration_b;
    result.top_k = options.top_k;
    result.weight_ranking_by_detector_score =
        options.weight_ranking_by_detector_score;
    return result;
}

} // namespace

struct OpenVocabularyRelation::Impl final
{
    Impl(runtime::ResolvedModel resolved_value,
         OpenVocabularyRelationOptions options_value,
         ScoringMode mode_value)
        : resolved(std::move(resolved_value))
        , options(std::move(options_value))
        , mode(mode_value)
        , context(resolved.model->create_context())
    {
        if (mode == ScoringMode::BackendLogits)
        {
            validate_backend_contract(
                resolved.model->tensors(),
                options,
                input_size_value);
        }
        else
        {
            validate_encoder_contract(
                resolved.model->tensors(),
                options,
                input_size_value);
        }

        std::size_t bytes = checked_multiply(
            options.max_pairs,
            sizeof(float),
            "pair logits");
        bytes = checked_add(
            bytes,
            checked_multiply(
                options.max_pairs,
                sizeof(std::int64_t) * 2U,
                "pair indices"),
            "relation fixed outputs");
        bytes = checked_add(
            bytes,
            checked_multiply(
                options.max_pairs,
                sizeof(std::uint8_t),
                "valid mask"),
            "relation fixed outputs");

        std::size_t query_values = 0U;
        if (mode == ScoringMode::HostQueries)
        {
            query_values = checked_multiply(
                options.max_pairs, options.query_dim, "relation query");
            bytes = checked_add(
                bytes,
                checked_multiply(
                    query_values,
                    sizeof(float) * 2U,
                    "relation queries"),
                "relation fixed outputs");
        }

        fixed_output_bytes = bytes;
        if (fixed_output_bytes > options.max_output_bytes)
        {
            throw_resource(
                "relation fixed outputs exceed configured byte limit");
        }

        pair_logits.resize(options.max_pairs);
        subject_indices.resize(options.max_pairs);
        object_indices.resize(options.max_pairs);
        valid_mask.resize(options.max_pairs);
        if (mode == ScoringMode::HostQueries)
        {
            semantic_query.resize(query_values);
            spatial_query.resize(query_values);
        }
    }

    runtime::ResolvedModel resolved;
    OpenVocabularyRelationOptions options;
    ScoringMode mode = ScoringMode::HostQueries;
    std::unique_ptr<runtime::ExecutionContext> context;
    PredicateVocabulary vocabulary;
    std::uint64_t vocabulary_version = 0U;
    std::int32_t input_size_value = 0;
    std::size_t fixed_output_bytes = 0U;

    std::vector<float> semantic_query;
    std::vector<float> spatial_query;
    std::vector<float> pair_logits;
    std::vector<std::int64_t> subject_indices;
    std::vector<std::int64_t> object_indices;
    std::vector<std::uint8_t> valid_mask;
    std::vector<float> pred_logits;

    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

OpenVocabularyRelation::OpenVocabularyRelation(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

OpenVocabularyRelation::~OpenVocabularyRelation() = default;

std::unique_ptr<OpenVocabularyRelation>
OpenVocabularyRelation::load(
    runtime::Runtime& runtime,
    const runtime::ModelPackage& package,
    const runtime::ExecutionPolicy& policy,
    const OpenVocabularyRelationOptions& options)
{
    validate_options(options);

    ScoringMode mode = ScoringMode::HostQueries;
    if (package.model_type() == kDynamicOpenVocabularyRelationModelType)
    {
        mode = ScoringMode::BackendLogits;
    }
    else if (package.model_type() != kOpenVocabularyRelationModelType)
    {
        throw_contract(
            "ModelPackage model_type must be "
            "'relation.open-vocabulary-encoder' or "
            "'relation.open-vocabulary'");
    }

    try
    {
        auto resolved = runtime.load_model(package, policy);
        return std::unique_ptr<OpenVocabularyRelation>(
            new OpenVocabularyRelation(
                std::make_unique<Impl>(
                    std::move(resolved), options, mode)));
    }
    catch (const RelationError&)
    {
        throw;
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("model allocation failed");
    }
}

void OpenVocabularyRelation::set_vocabulary(
    PredicateVocabulary vocabulary)
{
    if (!impl_)
    {
        throw_invalid("model state is unavailable");
    }
    UseGuard guard(impl_->in_use);

    const bool complete_alpha =
        !vocabulary.predicates.empty() &&
        vocabulary.spatial_weights.size() ==
            vocabulary.predicates.size();
    if (impl_->mode == ScoringMode::BackendLogits &&
        !complete_alpha)
    {
        throw_invalid(
            "backend-scoring vocabulary requires one alpha value per predicate");
    }

    try
    {
        PredicateVocabulary candidate =
            normalize_predicate_vocabulary(
                std::move(vocabulary),
                impl_->options.query_dim,
                impl_->options.max_vocabulary_bytes);

        if (impl_->mode == ScoringMode::BackendLogits)
        {
            std::size_t input_bytes = checked_multiply(
                candidate.embeddings.size(),
                sizeof(float),
                "predicate embedding input");
            input_bytes = checked_add(
                input_bytes,
                checked_multiply(
                    candidate.spatial_weights.size(),
                    sizeof(float),
                    "predicate alpha input"),
                "dynamic vocabulary inputs");
            if (input_bytes > impl_->options.max_tensor_bytes)
            {
                throw_resource(
                    "dynamic vocabulary inputs exceed configured tensor byte limit");
            }
        }

        const std::size_t pred_values = checked_multiply(
            impl_->options.max_pairs,
            candidate.predicates.size(),
            "predicate logits");
        const std::size_t pred_bytes = checked_multiply(
            pred_values,
            sizeof(float),
            "predicate logits");
        const std::size_t total_output_bytes = checked_add(
            impl_->fixed_output_bytes,
            pred_bytes,
            "relation outputs");
        if (total_output_bytes > impl_->options.max_output_bytes)
        {
            throw_resource(
                "dynamic predicate logits exceed configured output byte limit");
        }
        if (impl_->vocabulary_version ==
            (std::numeric_limits<std::uint64_t>::max)())
        {
            throw_resource(
                "predicate vocabulary version exhausted");
        }

        std::vector<float> candidate_logits(
            pred_values,
            0.0F);

        impl_->vocabulary = std::move(candidate);
        impl_->pred_logits.swap(candidate_logits);
        ++impl_->vocabulary_version;
    }
    catch (const RelationError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource(
            "dynamic predicate vocabulary allocation failed");
    }
}

TimedRelationFrame OpenVocabularyRelation::infer_timed(
    const image::ImageView& image,
    const std::vector<Region>& regions)
{
    if (!impl_)
    {
        throw_invalid("model state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    const auto total_start = std::chrono::steady_clock::now();
    if (impl_->vocabulary.predicates.empty())
    {
        throw_invalid("set_vocabulary must be called before infer");
    }

    try
    {
        const image::BgrImage source =
            image::CpuImageProcessor::copy_bgr(
                image, impl_->options.max_source_bytes);
        validate_regions(
            regions, source.width, source.height,
            impl_->options.max_boxes);

        const image::BgrImage resized =
            image::CpuImageProcessor::resize_bgr(
                source,
                impl_->input_size_value,
                impl_->input_size_value,
                impl_->options.max_source_bytes);

        image::PreprocessOptions preprocess;
        preprocess.output_format = image::PixelFormat::Rgb8;
        preprocess.mean = {0.0F, 0.0F, 0.0F};
        preprocess.stddev = {1.0F, 1.0F, 1.0F};
        const std::vector<float> image_tensor =
            image::CpuImageProcessor::to_nchw(
                resized, preprocess,
                impl_->options.max_tensor_bytes);

        const std::size_t box_values = checked_multiply(
            impl_->options.max_boxes, 4U, "boxes tensor");
        std::vector<float> boxes(box_values, 0.0F);
        const float image_width = static_cast<float>(source.width);
        const float image_height = static_cast<float>(source.height);
        for (std::size_t index = 0U;
             index < regions.size(); ++index)
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
            1, 3,
            impl_->input_size_value,
            impl_->input_size_value
        };
        const runtime::TensorShape boxes_shape {
            1,
            static_cast<std::int64_t>(impl_->options.max_boxes),
            4
        };
        const runtime::TensorShape count_shape {1};
        const runtime::TensorShape pair_shape {
            1,
            static_cast<std::int64_t>(impl_->options.max_pairs)
        };

        std::vector<runtime::TensorView> inputs {
            {"image", runtime::DataType::Float32, image_shape,
             image_tensor.data(),
             image_tensor.size() * sizeof(float),
             runtime::MemoryKind::Host, {}},
            {"boxes", runtime::DataType::Float32, boxes_shape,
             boxes.data(), boxes.size() * sizeof(float),
             runtime::MemoryKind::Host, {}},
            {"box_counts", runtime::DataType::Int64, count_shape,
             &box_count, sizeof(box_count),
             runtime::MemoryKind::Host, {}},
        };
        std::vector<runtime::MutableTensorView> outputs;
        const auto preprocess_end = std::chrono::steady_clock::now();
        auto backend_end = preprocess_end;
        auto predicate_score_end = preprocess_end;

        if (impl_->mode == ScoringMode::BackendLogits)
        {
            const std::int64_t predicate_count =
                static_cast<std::int64_t>(
                    impl_->vocabulary.predicates.size());
            const runtime::TensorShape bank_shape {
                predicate_count,
                static_cast<std::int64_t>(
                    impl_->options.query_dim)
            };
            const runtime::TensorShape alpha_shape {
                predicate_count
            };
            const runtime::TensorShape pred_shape {
                1,
                static_cast<std::int64_t>(
                    impl_->options.max_pairs),
                predicate_count
            };

            inputs.push_back(
                {"W", runtime::DataType::Float32, bank_shape,
                 impl_->vocabulary.embeddings.data(),
                 impl_->vocabulary.embeddings.size() * sizeof(float),
                 runtime::MemoryKind::Host, {}});
            inputs.push_back(
                {"alpha", runtime::DataType::Float32, alpha_shape,
                 impl_->vocabulary.spatial_weights.data(),
                 impl_->vocabulary.spatial_weights.size() * sizeof(float),
                 runtime::MemoryKind::Host, {}});

            outputs = {
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
            backend_end = std::chrono::steady_clock::now();
            predicate_score_end = backend_end;
        }
        else
        {
            const runtime::TensorShape query_shape {
                1,
                static_cast<std::int64_t>(
                    impl_->options.max_pairs),
                static_cast<std::int64_t>(
                    impl_->options.query_dim)
            };
            outputs = {
                {"semantic_query", runtime::DataType::Float32, query_shape,
                 impl_->semantic_query.data(),
                 impl_->semantic_query.size() * sizeof(float),
                 runtime::MemoryKind::Host, {}},
                {"spatial_query", runtime::DataType::Float32, query_shape,
                 impl_->spatial_query.data(),
                 impl_->spatial_query.size() * sizeof(float),
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
            backend_end = std::chrono::steady_clock::now();

            const detail::RawOpenVocabularyQueries queries {
                impl_->semantic_query.data(),
                impl_->spatial_query.data(),
                impl_->valid_mask.data(),
                impl_->options.max_pairs,
                impl_->options.query_dim,
            };
            const detail::NormalizedVocabularyView vocabulary {
                impl_->vocabulary.embeddings.data(),
                impl_->vocabulary.spatial_weights.data(),
                impl_->vocabulary.predicates.size(),
                impl_->vocabulary.embedding_dim,
            };
            detail::score_open_vocabulary_queries(
                queries, vocabulary,
                impl_->options.logit_scale,
                impl_->options.logit_bias,
                impl_->pred_logits.data());
            predicate_score_end = std::chrono::steady_clock::now();
        }

        const auto runtime_end = predicate_score_end;

        const detail::RawRelationOutputs raw {
            impl_->pred_logits.data(),
            impl_->pair_logits.data(),
            impl_->subject_indices.data(),
            impl_->object_indices.data(),
            impl_->valid_mask.data(),
            impl_->options.max_pairs,
            impl_->vocabulary.predicates.size(),
        };

        RelationFrame result;
        result.image_width = source.width;
        result.image_height = source.height;
        result.vocabulary_version = impl_->vocabulary_version;
        result.edges = detail::decode_relation_outputs(
            raw, regions,
            decode_options(impl_->options, impl_->vocabulary));
        const auto decode_end = std::chrono::steady_clock::now();

        const auto milliseconds = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(
                       end - begin)
                .count();
        };

        TimedRelationFrame timed;
        timed.timing.preprocess_ms =
            milliseconds(total_start, preprocess_end);
        timed.timing.backend_ms =
            milliseconds(preprocess_end, backend_end);
        timed.timing.predicate_score_ms =
            milliseconds(backend_end, predicate_score_end);
        timed.timing.runtime_ms =
            milliseconds(preprocess_end, runtime_end);
        timed.timing.decode_ms =
            milliseconds(runtime_end, decode_end);
        timed.timing.total_ms =
            milliseconds(total_start, decode_end);
        timed.timing.region_count = regions.size();
        timed.timing.predicate_count =
            impl_->vocabulary.predicates.size();
        timed.timing.valid_pair_count =
            static_cast<std::size_t>(
                std::count_if(
                    impl_->valid_mask.begin(),
                    impl_->valid_mask.end(),
                    [](std::uint8_t value) { return value != 0U; }));
        timed.timing.edge_count = result.edges.size();
        timed.frame = std::move(result);
        return timed;
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

RelationFrame OpenVocabularyRelation::infer(
    const image::ImageView& image,
    const std::vector<Region>& regions)
{
    return infer_timed(image, regions).frame;
}

std::int32_t OpenVocabularyRelation::input_size() const noexcept
{
    return impl_ ? impl_->input_size_value : 0;
}

std::size_t OpenVocabularyRelation::max_boxes() const noexcept
{
    return impl_ ? impl_->options.max_boxes : 0U;
}

std::size_t OpenVocabularyRelation::max_pairs() const noexcept
{
    return impl_ ? impl_->options.max_pairs : 0U;
}

std::size_t OpenVocabularyRelation::query_dim() const noexcept
{
    return impl_ ? impl_->options.query_dim : 0U;
}

std::size_t OpenVocabularyRelation::predicate_count() const noexcept
{
    return impl_ ? impl_->vocabulary.predicates.size() : 0U;
}

std::uint64_t OpenVocabularyRelation::vocabulary_version() const noexcept
{
    return impl_ ? impl_->vocabulary_version : 0U;
}

bool OpenVocabularyRelation::backend_scoring() const noexcept
{
    return impl_ &&
        impl_->mode == ScoringMode::BackendLogits;
}

const std::vector<std::string>&
OpenVocabularyRelation::predicates() const noexcept
{
    static const std::vector<std::string> empty;
    return impl_ ? impl_->vocabulary.predicates : empty;
}

const runtime::ExecutionRoute&
OpenVocabularyRelation::execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty {};
    return impl_ ? impl_->resolved.route : empty;
}

} // namespace kfcore::relation
