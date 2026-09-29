#pragma once

#include "kfcore/relation/open_vocabulary_relation.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/resolver.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace kfcore::relation
{

inline constexpr std::string_view kPredicateTextEncoderModelType =
    "relation.predicate-text-encoder";

struct PredicateTokenBatch
{
    std::vector<std::int64_t> input_ids;
    std::vector<std::uint8_t> padding_mask;
    std::size_t rows = 0U;
    std::size_t length = 0U;
};

class PredicateTokenizer
{
public:
    virtual ~PredicateTokenizer() = default;

    PredicateTokenizer(const PredicateTokenizer&) = delete;
    PredicateTokenizer& operator=(const PredicateTokenizer&) = delete;

    [[nodiscard]] virtual PredicateTokenBatch
    tokenize(const std::vector<std::string>& texts,
             std::size_t max_length) const = 0;

    [[nodiscard]] virtual std::string provenance() const = 0;

protected:
    PredicateTokenizer() = default;
};

struct PredicateTextEncoderOptions
{
    std::size_t max_length = 32U;
    std::size_t embedding_dim = 512U;
    std::size_t tokenizer_vocab_size = 49408U;
    std::size_t inference_batch_size = 256U;
    std::size_t max_predicates = 19103U;
    std::size_t max_text_bytes = 4U * 1024U * 1024U;
    std::size_t max_tensor_bytes = 64U * 1024U * 1024U;
    std::size_t cache_capacity = 4096U;

    // Apache RelateAnything reference training templates.
    std::vector<std::string> templates {
        "{p}",
        "one object is {p} another object",
        "a photo of something {p} something",
    };
};

class PredicateTextEncoder final
{
public:
    ~PredicateTextEncoder();

    PredicateTextEncoder(const PredicateTextEncoder&) = delete;
    PredicateTextEncoder& operator=(const PredicateTextEncoder&) = delete;

    [[nodiscard]] static std::unique_ptr<PredicateTextEncoder>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& package,
         const runtime::ExecutionPolicy& policy,
         std::shared_ptr<const PredicateTokenizer> tokenizer,
         const PredicateTextEncoderOptions& options = {});

    [[nodiscard]] PredicateVocabulary
    encode(const std::vector<std::string>& predicates);

    void clear_cache();

    [[nodiscard]] std::size_t embedding_dim() const noexcept;
    [[nodiscard]] std::size_t max_length() const noexcept;
    [[nodiscard]] std::size_t cache_size() const noexcept;
    [[nodiscard]] const runtime::ExecutionRoute&
    execution_route() const noexcept;

private:
    struct Impl;
    explicit PredicateTextEncoder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::relation
