#include "face_swap_cli.hpp"
#include "tinytest.hpp"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using kfcore::face_applications::cli::Arguments;
using kfcore::face_applications::cli::parse_arguments;

namespace
{

struct TempFile final
{
    TempFile()
        : path(tt_make_temp_file("face-swap-cli", ".bin"))
    {
        check_not_null(path);
        if (path != nullptr)
        {
            constexpr char kByte = 'x';
            check(tt_write_file(path, &kByte, sizeof(kByte)) == 0);
        }
    }

    ~TempFile()
    {
        if (path != nullptr)
        {
            (void)tt_remove_file(path);
            std::free(path);
        }
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    char* path = nullptr;
};

void expect_failure(const std::vector<std::string>& values, const std::string& message)
{
    bool threw = false;
    try
    {
        (void)parse_arguments(values);
    }
    catch (const std::invalid_argument& error)
    {
        threw = true;
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

std::vector<std::string> required_arguments(const char* input_path)
{
    return { "face_swap_image", "--source", input_path, "--target", input_path,
             "--output", "result.png" };
}

} // namespace

spec("face swap CLI arguments")
{
    it("parses only image input and output paths")
    {
        TempFile file;
        const Arguments arguments = parse_arguments(required_arguments(file.path));

        check(arguments.source == file.path);
        check(arguments.target == file.path);
        check(arguments.output == "result.png");
    }

    it("rejects missing, duplicate, unknown, and valueless options")
    {
        TempFile file;
        std::vector<std::string> missing = required_arguments(file.path);
        missing.erase(missing.begin() + 1, missing.begin() + 3);
        expect_failure(missing, "--source");

        std::vector<std::string> duplicate = required_arguments(file.path);
        duplicate.insert(duplicate.end(), { "--output", "other.png" });
        expect_failure(duplicate, "duplicate");

        std::vector<std::string> unknown = required_arguments(file.path);
        unknown.insert(unknown.end(), { "--fallback-face", file.path });
        expect_failure(unknown, "unknown");

        std::vector<std::string> valueless = required_arguments(file.path);
        valueless.emplace_back("--output");
        expect_failure(valueless, "missing value");
    }

    it("rejects nonexistent input files")
    {
        TempFile file;
        std::vector<std::string> values = required_arguments(file.path);
        values[2] = "this-face-swap-file-must-not-exist.jpg";
        expect_failure(values, "--source");
    }

    it("rejects an output alias that resolves to an input file")
    {
        TempFile file;
        std::vector<std::string> values = required_arguments(file.path);
        const std::filesystem::path input(file.path);
        values[6] = (input.parent_path() / "." / input.filename()).string();
        expect_failure(values, "--output");
    }
}
