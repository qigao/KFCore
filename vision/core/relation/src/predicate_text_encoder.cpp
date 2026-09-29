#include "kfcore/relation/predicate_text_encoder.hpp"

#include "kfcore/relation/error.hpp"
#include "kfcore/runtime/error.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kfcore::relation
{
namespace
{

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw RelationError(
        RelationErrorCode::InvalidArgument,
        "PredicateTextEncoder: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw RelationError(
        RelationErrorCode::ModelContractMismatch,
        "PredicateTextEncoder model contract: " + detail);
}

[[noreturn]] void throw_runtime(const std::string& detail)
{
    throw RelationError(
        RelationErrorCode::RuntimeFailure,
        "PredicateTextEncoder runtime: " + detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw RelationError(
        RelationErrorCode::ResourceLimitExceeded,
        "PredicateTextEncoder resource limit: " + detail);
}

std::size_t checked_multiply(
    std::size_t left,
    std::size_t right,
    const char* subject)
{
    if (left != 0U &&
        right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_resource(
            std::string(subject) + " byte count overflow");
    }
    return left * right;
}

std::size_t checked_add(
    std::size_t left,
    std::size_t right,
    const char* subject)
{
    if (right >
        (std::numeric_limits<std::size_t>::max)() - left)
    {
        throw_resource(
            std::string(subject) + " byte count overflow");
    }
    return left + right;
}

class UseGuard final
{
public:
    explicit UseGuard(std::atomic_flag& flag)
        : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw_invalid(
                "calls on one text encoder instance must not overlap");
        }
    }

    ~UseGuard()
    {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag& flag_;
};

void validate_options(
    const PredicateTextEncoderOptions& options)
{
    if (options.max_length == 0U ||
        options.embedding_dim == 0U ||
        options.tokenizer_vocab_size == 0U ||
        options.inference_batch_size == 0U ||
        options.max_predicates == 0U ||
        options.max_text_bytes == 0U ||
        options.max_tensor_bytes == 0U)
    {
        throw_invalid(
            "dimensions and resource limits must be positive");
    }
    if (options.templates.empty())
    {
        throw_invalid(
            "at least one predicate template is required");
    }
    for (const auto& value : options.templates)
    {
        if (value.empty() ||
            value.find("{p}") == std::string::npos)
        {
            throw_invalid(
                "every predicate template must contain {p}");
        }
    }

    const std::size_t output_bytes = checked_multiply(
        checked_multiply(
            options.max_predicates,
            options.embedding_dim,
            "predicate embedding output"),
        sizeof(float),
        "predicate embedding output");
    if (output_bytes > options.max_tensor_bytes)
    {
        throw_resource(
            "configured max predicate output exceeds max_tensor_bytes");
    }
}

const runtime::TensorDescriptor& require_tensor(
    const std::vector<runtime::TensorDescriptor>& tensors,
    const char* name,
    bool is_input)
{
    const auto match = std::find_if(
        tensors.begin(),
        tensors.end(),
        [&](const runtime::TensorDescriptor& tensor)
        {
            return tensor.name == name &&
                   tensor.is_input == is_input;
        });
    if (match == tensors.end())
    {
        throw_contract(
            std::string("missing ") +
            (is_input ? "input '" : "output '") +
            name + "'");
    }
    return *match;
}

void require_dynamic_batch(
    const runtime::TensorDescriptor& tensor,
    const char* name)
{
    if (tensor.shape.empty() ||
        tensor.shape[0] != -1)
    {
        throw_contract(
            std::string(name) +
            " must expose a dynamic predicate batch axis");
    }
}

void require_extent(
    std::int64_t declared,
    std::size_t expected,
    const char* subject)
{
    if (declared != static_cast<std::int64_t>(expected))
    {
        throw_contract(
            std::string(subject) +
            " has incompatible extent");
    }
}

void validate_model_contract(
    const std::vector<runtime::TensorDescriptor>& tensors,
    const PredicateTextEncoderOptions& options)
{
    std::size_t input_count = 0U;
    std::size_t output_count = 0U;
    for (const auto& tensor : tensors)
    {
        tensor.is_input ? ++input_count : ++output_count;
    }
    if (input_count != 2U || output_count != 1U)
    {
        throw_contract(
            "text encoder requires exactly 2 inputs and 1 output");
    }

    const auto& ids =
        require_tensor(tensors, "input_ids", true);
    const auto& mask =
        require_tensor(tensors, "padding_mask", true);
    const auto& embedding =
        require_tensor(tensors, "predicate_embedding", false);

    if (ids.data_type != runtime::DataType::Int64 ||
        ids.shape.size() != 2U)
    {
        throw_contract(
            "input_ids must be INT64 [V,L]");
    }
    if (mask.data_type != runtime::DataType::Bool ||
        mask.shape.size() != 2U)
    {
        throw_contract(
            "padding_mask must be BOOL [V,L]");
    }
    if (embedding.data_type != runtime::DataType::Float32 ||
        embedding.shape.size() != 2U)
    {
        throw_contract(
            "predicate_embedding must be FP32 [V,D]");
    }

    require_dynamic_batch(ids, "input_ids");
    require_dynamic_batch(mask, "padding_mask");
    require_dynamic_batch(
        embedding,
        "predicate_embedding");

    require_extent(
        ids.shape[1],
        options.max_length,
        "input_ids length");
    require_extent(
        mask.shape[1],
        options.max_length,
        "padding_mask length");
    require_extent(
        embedding.shape[1],
        options.embedding_dim,
        "predicate_embedding width");
}

std::string apply_template(
    const std::string& pattern,
    const std::string& predicate)
{
    std::string result;
    std::size_t cursor = 0U;
    for (;;)
    {
        const std::size_t marker =
            pattern.find("{p}", cursor);
        if (marker == std::string::npos)
        {
            result.append(pattern, cursor, std::string::npos);
            break;
        }
        result.append(pattern, cursor, marker - cursor);
        result.append(predicate);
        cursor = marker + 3U;
    }
    return result;
}

std::string template_provenance(
    const std::vector<std::string>& templates)
{
    std::string result;
    for (std::size_t index = 0U;
         index < templates.size();
         ++index)
    {
        if (index != 0U)
        {
            result.push_back('\n');
        }
        result += templates[index];
    }
    return result;
}

void normalize_row(float* values, std::size_t count)
{
    double norm2 = 0.0;
    for (std::size_t index = 0U; index < count; ++index)
    {
        const float value = values[index];
        if (!std::isfinite(value))
        {
            throw_runtime(
                "model returned a non-finite embedding value");
        }
        norm2 += static_cast<double>(value) *
                 static_cast<double>(value);
    }
    if (!(norm2 > 0.0) || !std::isfinite(norm2))
    {
        throw_runtime(
            "model returned a zero/non-finite embedding row");
    }
    const float inv_norm =
        static_cast<float>(1.0 / std::sqrt(norm2));
    for (std::size_t index = 0U; index < count; ++index)
    {
        values[index] *= inv_norm;
    }
}

struct CacheEntry
{
    std::vector<float> embedding;
    std::list<std::string>::iterator lru;
};

} // namespace

struct PredicateTextEncoder::Impl final
{
    Impl(runtime::ResolvedModel resolved_value,
         PredicateTextEncoderOptions options_value,
         std::shared_ptr<const PredicateTokenizer> tokenizer_value,
         std::string text_provenance_value)
        : resolved(std::move(resolved_value))
        , options(std::move(options_value))
        , tokenizer(std::move(tokenizer_value))
        , text_provenance(std::move(text_provenance_value))
        , tokenizer_provenance(tokenizer->provenance())
        , prompt_provenance(
              template_provenance(options.templates))
        , context(resolved.model->create_context())
    {
        validate_model_contract(
            resolved.model->tensors(),
            options);
        if (tokenizer_provenance.empty())
        {
            throw_invalid(
                "tokenizer provenance must not be empty");
        }
    }

    runtime::ResolvedModel resolved;
    PredicateTextEncoderOptions options;
    std::shared_ptr<const PredicateTokenizer> tokenizer;

    std::string text_provenance;
    std::string tokenizer_provenance;
    std::string prompt_provenance;

    std::unique_ptr<runtime::ExecutionContext> context;

    std::unordered_map<std::string, CacheEntry> cache;
    std::list<std::string> lru;
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;

    void touch(
        std::unordered_map<std::string, CacheEntry>::iterator item)
    {
        lru.splice(
            lru.begin(),
            lru,
            item->second.lru);
        item->second.lru = lru.begin();
    }

    void insert_cache(
        const std::string& predicate,
        std::vector<float> embedding)
    {
        if (options.cache_capacity == 0U)
        {
            return;
        }

        auto existing = cache.find(predicate);
        if (existing != cache.end())
        {
            existing->second.embedding =
                std::move(embedding);
            touch(existing);
            return;
        }

        while (cache.size() >= options.cache_capacity &&
               !lru.empty())
        {
            cache.erase(lru.back());
            lru.pop_back();
        }

        lru.push_front(predicate);
        cache.emplace(
            predicate,
            CacheEntry {
                std::move(embedding),
                lru.begin(),
            });
    }

    [[nodiscard]] std::vector<float>
    run_text_batch(
        const std::vector<std::string>& texts)
    {
        if (texts.empty())
        {
            return {};
        }
        const PredicateTokenBatch tokens =
            tokenizer->tokenize(
                texts,
                options.max_length);
        if (tokens.rows != texts.size() ||
            tokens.length != options.max_length)
        {
            throw_invalid(
                "tokenizer returned the wrong batch shape");
        }

        const std::size_t token_values =
            checked_multiply(
                tokens.rows,
                tokens.length,
                "token input");
        if (tokens.input_ids.size() != token_values ||
            tokens.padding_mask.size() != token_values)
        {
            throw_invalid(
                "tokenizer buffers do not match declared shape");
        }

        for (const auto id : tokens.input_ids)
        {
            if (id < 0 ||
                static_cast<std::uint64_t>(id) >=
                    options.tokenizer_vocab_size)
            {
                throw_invalid(
                    "tokenizer emitted an out-of-range token id");
            }
        }
        for (const auto value : tokens.padding_mask)
        {
            if (value > 1U)
            {
                throw_invalid(
                    "tokenizer padding mask must contain only 0/1");
            }
        }

        const std::size_t input_bytes =
            checked_multiply(
                token_values,
                sizeof(std::int64_t) +
                    sizeof(std::uint8_t),
                "token input");
        if (input_bytes > options.max_tensor_bytes)
        {
            throw_resource(
                "token input exceeds max_tensor_bytes");
        }

        const std::size_t output_values =
            checked_multiply(
                tokens.rows,
                options.embedding_dim,
                "predicate embedding");
        const std::size_t output_bytes =
            checked_multiply(
                output_values,
                sizeof(float),
                "predicate embedding");
        if (output_bytes > options.max_tensor_bytes)
        {
            throw_resource(
                "predicate embedding output exceeds max_tensor_bytes");
        }

        std::vector<float> output(
            output_values,
            0.0F);

        const runtime::TensorShape token_shape {
            static_cast<std::int64_t>(tokens.rows),
            static_cast<std::int64_t>(tokens.length),
        };
        const runtime::TensorShape output_shape {
            static_cast<std::int64_t>(tokens.rows),
            static_cast<std::int64_t>(
                options.embedding_dim),
        };

        std::vector<runtime::TensorView> inputs {
            {
                "input_ids",
                runtime::DataType::Int64,
                token_shape,
                tokens.input_ids.data(),
                tokens.input_ids.size() *
                    sizeof(std::int64_t),
                runtime::MemoryKind::Host,
                {},
            },
            {
                "padding_mask",
                runtime::DataType::Bool,
                token_shape,
                tokens.padding_mask.data(),
                tokens.padding_mask.size() *
                    sizeof(std::uint8_t),
                runtime::MemoryKind::Host,
                {},
            },
        };
        std::vector<runtime::MutableTensorView> outputs {
            {
                "predicate_embedding",
                runtime::DataType::Float32,
                output_shape,
                output.data(),
                output.size() * sizeof(float),
                runtime::MemoryKind::Host,
                {},
            },
        };

        context->run(inputs, outputs);

        for (std::size_t row = 0U;
             row < tokens.rows;
             ++row)
        {
            normalize_row(
                output.data() +
                    row * options.embedding_dim,
                options.embedding_dim);
        }
        return output;
    }

    [[nodiscard]] std::vector<float>
    encode_uncached(
        const std::vector<std::string>& predicates)
    {
        const std::size_t template_count =
            options.templates.size();
        const std::size_t predicates_per_batch =
            std::max<std::size_t>(
                1U,
                options.inference_batch_size /
                    template_count);

        std::vector<float> result(
            checked_multiply(
                predicates.size(),
                options.embedding_dim,
                "uncached predicate embeddings"),
            0.0F);

        for (std::size_t begin = 0U;
             begin < predicates.size();
             begin += predicates_per_batch)
        {
            const std::size_t end = std::min(
                predicates.size(),
                begin + predicates_per_batch);
            std::vector<std::string> expanded;
            expanded.reserve(
                (end - begin) * template_count);

            for (std::size_t row = begin;
                 row < end;
                 ++row)
            {
                for (const auto& pattern : options.templates)
                {
                    expanded.push_back(
                        apply_template(
                            pattern,
                            predicates[row]));
                }
            }

            const std::vector<float> encoded =
                run_text_batch(expanded);

            for (std::size_t row = begin;
                 row < end;
                 ++row)
            {
                float* target =
                    result.data() +
                    row * options.embedding_dim;
                const std::size_t local_row =
                    row - begin;
                for (std::size_t template_index = 0U;
                     template_index < template_count;
                     ++template_index)
                {
                    const float* source =
                        encoded.data() +
                        (local_row * template_count +
                         template_index) *
                            options.embedding_dim;
                    for (std::size_t column = 0U;
                         column < options.embedding_dim;
                         ++column)
                    {
                        target[column] += source[column];
                    }
                }
                normalize_row(
                    target,
                    options.embedding_dim);
            }
        }
        return result;
    }
};

PredicateTextEncoder::PredicateTextEncoder(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

PredicateTextEncoder::~PredicateTextEncoder() = default;

std::unique_ptr<PredicateTextEncoder>
PredicateTextEncoder::load(
    runtime::Runtime& runtime,
    const runtime::ModelPackage& package,
    const runtime::ExecutionPolicy& policy,
    std::shared_ptr<const PredicateTokenizer> tokenizer,
    const PredicateTextEncoderOptions& options)
{
    validate_options(options);
    if (!tokenizer)
    {
        throw_invalid(
            "tokenizer must not be null");
    }
    if (package.model_type() !=
        kPredicateTextEncoderModelType)
    {
        throw_contract(
            "ModelPackage model_type must be "
            "'relation.predicate-text-encoder'");
    }

    try
    {
        auto resolved =
            runtime.load_model(package, policy);
        std::string provenance =
            package.id() + ":" + package.version();
        if (!resolved.route.artifact.sha256.empty())
        {
            provenance += ":" +
                resolved.route.artifact.sha256;
        }

        return std::unique_ptr<PredicateTextEncoder>(
            new PredicateTextEncoder(
                std::make_unique<Impl>(
                    std::move(resolved),
                    options,
                    std::move(tokenizer),
                    std::move(provenance))));
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
        throw_resource(
            "text-encoder allocation failed");
    }
}

PredicateVocabulary PredicateTextEncoder::encode(
    const std::vector<std::string>& predicates)
{
    if (!impl_)
    {
        throw_invalid(
            "model state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    if (predicates.empty())
    {
        throw_invalid(
            "predicate list must not be empty");
    }
    if (predicates.size() >
        impl_->options.max_predicates)
    {
        throw_resource(
            "predicate count exceeds configured maximum");
    }

    std::size_t text_bytes = 0U;
    std::unordered_set<std::string> names;
    names.reserve(predicates.size());

    std::vector<float> embeddings(
        checked_multiply(
            predicates.size(),
            impl_->options.embedding_dim,
            "predicate vocabulary output"),
        0.0F);

    std::vector<std::string> misses;
    std::vector<std::size_t> miss_positions;

    for (std::size_t index = 0U;
         index < predicates.size();
         ++index)
    {
        const auto& predicate = predicates[index];
        if (predicate.empty())
        {
            throw_invalid(
                "predicate strings must not be empty");
        }
        text_bytes = checked_add(
            text_bytes,
            predicate.size(),
            "predicate text");
        if (text_bytes > impl_->options.max_text_bytes)
        {
            throw_resource(
                "predicate text exceeds configured byte limit");
        }
        if (!names.insert(predicate).second)
        {
            throw_invalid(
                "predicate strings must be unique");
        }

        auto cached =
            impl_->cache.find(predicate);
        if (cached != impl_->cache.end())
        {
            std::copy(
                cached->second.embedding.begin(),
                cached->second.embedding.end(),
                embeddings.begin() +
                    static_cast<std::ptrdiff_t>(
                        index *
                        impl_->options.embedding_dim));
            impl_->touch(cached);
        }
        else
        {
            misses.push_back(predicate);
            miss_positions.push_back(index);
        }
    }

    if (!misses.empty())
    {
        const std::vector<float> encoded =
            impl_->encode_uncached(misses);
        for (std::size_t miss = 0U;
             miss < misses.size();
             ++miss)
        {
            const float* source =
                encoded.data() +
                miss * impl_->options.embedding_dim;
            const std::size_t position =
                miss_positions[miss];
            std::copy(
                source,
                source + impl_->options.embedding_dim,
                embeddings.begin() +
                    static_cast<std::ptrdiff_t>(
                        position *
                        impl_->options.embedding_dim));

            impl_->insert_cache(
                misses[miss],
                std::vector<float>(
                    source,
                    source +
                        impl_->options.embedding_dim));
        }
    }

    PredicateVocabulary vocabulary;
    vocabulary.predicates = predicates;
    vocabulary.embeddings = std::move(embeddings);
    vocabulary.embedding_dim =
        impl_->options.embedding_dim;
    vocabulary.text_encoder_provenance =
        impl_->text_provenance;
    vocabulary.tokenizer_provenance =
        impl_->tokenizer_provenance;
    vocabulary.template_provenance =
        impl_->prompt_provenance;
    return vocabulary;
}

void PredicateTextEncoder::clear_cache()
{
    if (!impl_)
    {
        return;
    }
    impl_->cache.clear();
    impl_->lru.clear();
}

std::size_t
PredicateTextEncoder::embedding_dim() const noexcept
{
    return impl_ ?
        impl_->options.embedding_dim :
        0U;
}

std::size_t
PredicateTextEncoder::max_length() const noexcept
{
    return impl_ ?
        impl_->options.max_length :
        0U;
}

std::size_t
PredicateTextEncoder::cache_size() const noexcept
{
    return impl_ ?
        impl_->cache.size() :
        0U;
}

const runtime::ExecutionRoute&
PredicateTextEncoder::execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty {};
    return impl_ ? impl_->resolved.route : empty;
}

} // namespace kfcore::relation
