#include "kfcore/relation/predicate_text_encoder.hpp"

#include "tinytest.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace kfcore::relation;

namespace
{

class StubTokenizer final : public PredicateTokenizer
{
public:
    PredicateTokenBatch tokenize(
        const std::vector<std::string>& texts,
        std::size_t max_length) const override
    {
        PredicateTokenBatch batch;
        batch.rows = texts.size();
        batch.length = max_length;
        batch.input_ids.assign(
            batch.rows * batch.length,
            std::int64_t{0});
        batch.padding_mask.assign(
            batch.rows * batch.length,
            std::uint8_t{1});
        for (std::size_t row = 0U;
             row < batch.rows; ++row)
        {
            if (batch.length != 0U)
            {
                batch.input_ids[
                    row * batch.length] =
                    static_cast<std::int64_t>(row + 1U);
                batch.padding_mask[
                    row * batch.length] = 0U;
            }
        }
        return batch;
    }

    std::string provenance() const override
    {
        return "stub-tokenizer-v1";
    }
};

} // namespace

spec("predicate text encoder public contract")
{
    it("uses the Apache reference dimensions and templates")
    {
        PredicateTextEncoderOptions options;
        check(options.max_length == std::size_t{32U});
        check(options.embedding_dim == std::size_t{512U});
        check(
            options.tokenizer_vocab_size ==
            std::size_t{49408U});
        check(options.templates.size() == std::size_t{3U});
        check(options.templates[0] == "{p}");
        check(
            options.templates[1] ==
            "one object is {p} another object");
        check(
            options.templates[2] ==
            "a photo of something {p} something");
    }

    it("keeps tokenization behind a deterministic provider")
    {
        std::shared_ptr<const PredicateTokenizer> tokenizer =
            std::make_shared<StubTokenizer>();
        const auto batch = tokenizer->tokenize(
            {"holding", "riding"},
            4U);

        check(batch.rows == std::size_t{2U});
        check(batch.length == std::size_t{4U});
        check(batch.input_ids.size() == std::size_t{8U});
        check(batch.padding_mask.size() == std::size_t{8U});
        check(batch.input_ids[0] == std::int64_t{1});
        check(batch.input_ids[4] == std::int64_t{2});
        check(batch.padding_mask[0] == std::uint8_t{0});
        check(batch.padding_mask[1] == std::uint8_t{1});
        check(
            tokenizer->provenance() ==
            "stub-tokenizer-v1");
    }

    it("names the reference text model type explicitly")
    {
        check(
            kPredicateTextEncoderModelType ==
            "relation.predicate-text-encoder");
    }
}
