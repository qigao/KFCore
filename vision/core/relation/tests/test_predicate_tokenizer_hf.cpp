#include "kfcore/relation/predicate_tokenizer_hf.hpp"

#include "kfcore/relation/error.hpp"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct GoldenCase
{
    std::string text;
    std::vector<std::int64_t> ids;
};

unsigned char hex_value(char value)
{
    if (value >= '0' && value <= '9')
    {
        return static_cast<unsigned char>(value - '0');
    }
    if (value >= 'a' && value <= 'f')
    {
        return static_cast<unsigned char>(value - 'a' + 10);
    }
    if (value >= 'A' && value <= 'F')
    {
        return static_cast<unsigned char>(value - 'A' + 10);
    }
    throw std::runtime_error("invalid hex digit");
}

std::string decode_hex(const std::string& value)
{
    if (value.size() % 2U != 0U)
    {
        throw std::runtime_error("hex text length must be even");
    }
    std::string result;
    result.resize(value.size() / 2U);
    for (std::size_t index = 0U; index < result.size(); ++index)
    {
        result[index] = static_cast<char>(
            (hex_value(value[index * 2U]) << 4U) |
            hex_value(value[index * 2U + 1U]));
    }
    return result;
}

std::vector<std::int64_t> parse_ids(const std::string& value)
{
    std::vector<std::int64_t> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        if (item.empty())
        {
            throw std::runtime_error("empty token id");
        }
        result.push_back(std::stoll(item));
    }
    return result;
}

std::vector<GoldenCase> load_golden(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error("cannot open tokenizer golden file");
    }

    std::vector<GoldenCase> result;
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            continue;
        }
        const auto tab = line.find('\t');
        if (tab == std::string::npos)
        {
            throw std::runtime_error("invalid tokenizer golden line");
        }
        GoldenCase value;
        value.text = decode_hex(line.substr(0U, tab));
        value.ids = parse_ids(line.substr(tab + 1U));
        result.push_back(std::move(value));
    }
    if (result.empty())
    {
        throw std::runtime_error("tokenizer golden file is empty");
    }
    return result;
}

void require(bool condition, const char* detail)
{
    if (!condition)
    {
        throw std::runtime_error(detail);
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 3)
        {
            throw std::runtime_error(
                "usage: test_predicate_tokenizer_hf <tokenizer.json> <golden.txt>");
        }

        const auto tokenizer =
            kfcore::relation::load_clip_tokenizer_json(argv[1]);
        require(
            tokenizer->provenance().rfind(
                "hf-clip-tokenizer-json:",
                0U) == 0U,
            "tokenizer provenance prefix drifted");

        const auto golden = load_golden(argv[2]);
        std::vector<std::string> texts;
        texts.reserve(golden.size());
        for (const auto& value : golden)
        {
            texts.push_back(value.text);
        }

        const auto batch = tokenizer->tokenize(texts, 32U);
        require(batch.rows == golden.size(), "tokenizer row count drifted");
        require(batch.length == 32U, "tokenizer length drifted");
        require(
            batch.input_ids.size() == golden.size() * 32U,
            "tokenizer id buffer size drifted");
        require(
            batch.padding_mask.size() == golden.size() * 32U,
            "tokenizer mask buffer size drifted");

        bool saw_full_length = false;
        for (std::size_t row = 0U; row < golden.size(); ++row)
        {
            const auto& expected = golden[row].ids;
            require(!expected.empty(), "golden token row is empty");
            require(expected.size() <= 32U, "golden token row is too long");
            saw_full_length = saw_full_length || expected.size() == 32U;

            const std::size_t base = row * 32U;
            for (std::size_t column = 0U; column < 32U; ++column)
            {
                if (column < expected.size())
                {
                    require(
                        batch.input_ids[base + column] == expected[column],
                        "C++ CLIP token id differs from Python reference");
                    require(
                        batch.padding_mask[base + column] == 0U,
                        "reference token marked as padding");
                }
                else
                {
                    require(
                        batch.input_ids[base + column] == 0,
                        "student padding id must be zero");
                    require(
                        batch.padding_mask[base + column] == 1U,
                        "student padding mask must be true");
                }
            }
            require(
                batch.input_ids[base] == 49406,
                "CLIP BOS id drifted");
            require(
                batch.input_ids[base + expected.size() - 1U] == 49407,
                "CLIP EOS id drifted");
        }
        require(
            saw_full_length,
            "golden cases must exercise max-length truncation");

        bool rejected = false;
        try
        {
            (void)tokenizer->tokenize({"holding"}, 1U);
        }
        catch (const kfcore::relation::RelationError&)
        {
            rejected = true;
        }
        require(rejected, "max_length=1 must be rejected");

        std::cout << "CLIP tokenizer golden parity passed for "
                  << golden.size() << " cases\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
