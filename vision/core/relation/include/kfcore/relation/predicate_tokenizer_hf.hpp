#pragma once

#include "kfcore/relation/predicate_text_encoder.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>

namespace kfcore::relation
{

struct ClipTokenizerJsonOptions
{
    std::size_t max_asset_bytes = 16U * 1024U * 1024U;
    std::size_t expected_vocab_size = 49408U;
    std::int32_t bos_token_id = 49406;
    std::int32_t eos_token_id = 49407;
    std::int64_t pad_token_id = 0;
};

[[nodiscard]] std::shared_ptr<const PredicateTokenizer>
load_clip_tokenizer_json(
    const std::filesystem::path& tokenizer_json,
    const ClipTokenizerJsonOptions& options = {});

} // namespace kfcore::relation
