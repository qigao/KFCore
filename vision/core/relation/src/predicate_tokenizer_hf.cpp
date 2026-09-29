#include "kfcore/relation/predicate_tokenizer_hf.hpp"

#include "kfcore/relation/error.hpp"
#include "kfcore/runtime/model_package.hpp"

#include <tokenizers_c.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
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
        "ClipTokenizerJson: " + detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw RelationError(
        RelationErrorCode::ResourceLimitExceeded,
        "ClipTokenizerJson resource limit: " + detail);
}

std::string read_asset(
    const std::filesystem::path& path,
    std::size_t max_bytes)
{
    std::error_code error;
    const auto size = std::filesystem::file_size(
        path,
        error);
    if (error)
    {
        throw_invalid(
            "cannot stat tokenizer.json: " +
            error.message());
    }
    if (size >
        static_cast<std::uintmax_t>(max_bytes))
    {
        throw_resource(
            "tokenizer.json exceeds configured byte limit");
    }
    if (size >
        static_cast<std::uintmax_t>(
            (std::numeric_limits<std::size_t>::max)()))
    {
        throw_resource(
            "tokenizer.json size exceeds addressable memory");
    }

    std::ifstream stream(
        path,
        std::ios::binary);
    if (!stream)
    {
        throw_invalid(
            "cannot open tokenizer.json");
    }
    std::string blob(
        static_cast<std::size_t>(size),
        '\0');
    if (!blob.empty())
    {
        stream.read(
            blob.data(),
            static_cast<std::streamsize>(
                blob.size()));
        if (!stream)
        {
            throw_invalid(
                "failed to read tokenizer.json");
        }
    }
    return blob;
}

class EncodeResults final
{
public:
    explicit EncodeResults(std::size_t count)
        : values_(count)
    {
    }

    ~EncodeResults()
    {
        if (!values_.empty())
        {
            tokenizers_free_encode_results(
                values_.data(),
                values_.size());
        }
    }

    EncodeResults(const EncodeResults&) = delete;
    EncodeResults& operator=(const EncodeResults&) = delete;

    TokenizerEncodeResult* data() noexcept
    {
        return values_.data();
    }

    const TokenizerEncodeResult& operator[](
        std::size_t index) const
    {
        return values_[index];
    }

private:
    std::vector<TokenizerEncodeResult> values_;
};

class ClipTokenizerJson final : public PredicateTokenizer
{
public:
    ClipTokenizerJson(
        TokenizerHandle handle,
        ClipTokenizerJsonOptions options,
        std::string provenance)
        : handle_(handle)
        , options_(std::move(options))
        , provenance_(std::move(provenance))
    {
        if (handle_ == nullptr)
        {
            throw_invalid(
                "HuggingFace tokenizer.json failed to load");
        }
    }

    ~ClipTokenizerJson() override
    {
        if (handle_ != nullptr)
        {
            tokenizers_free(handle_);
        }
    }

    PredicateTokenBatch tokenize(
        const std::vector<std::string>& texts,
        std::size_t max_length) const override
    {
        if (max_length < 2U)
        {
            throw_invalid(
                "max_length must reserve BOS and EOS");
        }

        PredicateTokenBatch batch;
        batch.rows = texts.size();
        batch.length = max_length;
        if (texts.empty())
        {
            return batch;
        }

        if (max_length >
            (std::numeric_limits<std::size_t>::max)() /
                texts.size())
        {
            throw_resource(
                "token batch shape overflow");
        }
        const std::size_t total =
            texts.size() * max_length;
        batch.input_ids.assign(
            total,
            options_.pad_token_id);
        batch.padding_mask.assign(
            total,
            std::uint8_t{1});

        std::vector<const char*> pointers;
        std::vector<std::size_t> lengths;
        pointers.reserve(texts.size());
        lengths.reserve(texts.size());
        for (const auto& text : texts)
        {
            pointers.push_back(text.data());
            lengths.push_back(text.size());
        }

        EncodeResults encoded(texts.size());
        std::lock_guard<std::mutex> lock(mutex_);
        tokenizers_encode_batch(
            handle_,
            pointers.data(),
            lengths.data(),
            texts.size(),
            1,
            encoded.data());

        for (std::size_t row = 0U;
             row < texts.size();
             ++row)
        {
            const auto& value = encoded[row];
            if (value.token_ids == nullptr ||
                value.len < 2U)
            {
                throw_invalid(
                    "CLIP tokenizer did not emit BOS/EOS");
            }
            if (value.token_ids[0] !=
                    options_.bos_token_id ||
                value.token_ids[value.len - 1U] !=
                    options_.eos_token_id)
            {
                throw_invalid(
                    "CLIP tokenizer special-token contract drifted");
            }

            const std::size_t copied =
                std::min(
                    value.len,
                    max_length);
            const std::size_t base =
                row * max_length;

            if (value.len <= max_length)
            {
                for (std::size_t index = 0U;
                     index < value.len;
                     ++index)
                {
                    batch.input_ids[base + index] =
                        static_cast<std::int64_t>(
                            value.token_ids[index]);
                    batch.padding_mask[base + index] =
                        std::uint8_t{0};
                }
            }
            else
            {
                // HuggingFace CLIP truncation budgets two special tokens:
                // BOS + first max_length-2 content ids + EOS.
                batch.input_ids[base] =
                    options_.bos_token_id;
                batch.padding_mask[base] =
                    std::uint8_t{0};

                for (std::size_t index = 1U;
                     index + 1U < max_length;
                     ++index)
                {
                    batch.input_ids[base + index] =
                        static_cast<std::int64_t>(
                            value.token_ids[index]);
                    batch.padding_mask[base + index] =
                        std::uint8_t{0};
                }
                batch.input_ids[
                    base + max_length - 1U] =
                    options_.eos_token_id;
                batch.padding_mask[
                    base + max_length - 1U] =
                    std::uint8_t{0};
            }

            for (std::size_t index = 0U;
                 index < copied;
                 ++index)
            {
                const auto token =
                    batch.input_ids[base + index];
                if (token < 0 ||
                    static_cast<std::uint64_t>(token) >=
                        options_.expected_vocab_size)
                {
                    throw_invalid(
                        "CLIP tokenizer emitted an out-of-range id");
                }
            }
        }
        return batch;
    }

    std::string provenance() const override
    {
        return provenance_;
    }

private:
    TokenizerHandle handle_ = nullptr;
    ClipTokenizerJsonOptions options_;
    std::string provenance_;
    mutable std::mutex mutex_;
};

} // namespace

std::shared_ptr<const PredicateTokenizer>
load_clip_tokenizer_json(
    const std::filesystem::path& tokenizer_json,
    const ClipTokenizerJsonOptions& options)
{
    if (options.max_asset_bytes == 0U ||
        options.expected_vocab_size == 0U)
    {
        throw_invalid(
            "asset/vocabulary limits must be positive");
    }
    if (options.bos_token_id < 0 ||
        options.eos_token_id < 0 ||
        options.pad_token_id < 0)
    {
        throw_invalid(
            "special token ids must be non-negative");
    }

    try
    {
        const std::string blob =
            read_asset(
                tokenizer_json,
                options.max_asset_bytes);
        TokenizerHandle handle =
            tokenizers_new_from_str(
                blob.data(),
                blob.size());
        if (handle == nullptr)
        {
            throw_invalid(
                "HuggingFace tokenizer.json failed to load");
        }

        struct HandleGuard
        {
            TokenizerHandle value = nullptr;
            ~HandleGuard()
            {
                if (value != nullptr)
                {
                    tokenizers_free(value);
                }
            }
        } guard {handle};

        std::size_t vocab_size = 0U;
        tokenizers_get_vocab_size(
            handle,
            &vocab_size);
        if (vocab_size !=
            options.expected_vocab_size)
        {
            throw_invalid(
                "CLIP vocabulary size is not 49,408");
        }

        std::int32_t bos = -1;
        std::int32_t eos = -1;
        constexpr char kBos[] =
            "<|startoftext|>";
        constexpr char kEos[] =
            "<|endoftext|>";
        tokenizers_token_to_id(
            handle,
            kBos,
            sizeof(kBos) - 1U,
            &bos);
        tokenizers_token_to_id(
            handle,
            kEos,
            sizeof(kEos) - 1U,
            &eos);
        if (bos != options.bos_token_id ||
            eos != options.eos_token_id)
        {
            throw_invalid(
                "CLIP BOS/EOS ids do not match the reference contract");
        }

        std::string provenance =
            "hf-clip-tokenizer-json:" +
            runtime::compute_model_artifact_sha256(
                tokenizer_json);

        guard.value = nullptr;
        return std::make_shared<ClipTokenizerJson>(
            handle,
            options,
            std::move(provenance));
    }
    catch (const RelationError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        throw_resource(
            "tokenizer allocation failed");
    }
}

} // namespace kfcore::relation
