#include "face_swap_cli.hpp"
#include "tinytest.hpp"

#include <cstdlib>
#include <functional>
#include <filesystem>
#include <fstream>
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

class ModelRootFixture final
{
public:
    ModelRootFixture()
        : root_(std::filesystem::temp_directory_path() / "kfcore_face_cli_models")
    {
        std::filesystem::remove_all(root_);
        write("tensorrt/test-profile/yolov12n-face.engine");
        write("tensorrt/test-profile/2dfan4.engine");
        write("tensorrt/test-profile/arcface_w600k_r50.engine");
        write("tensorrt/test-profile/inswapper_128.engine");
        write("model_matrix.bin");
    }

    ~ModelRootFixture() { std::filesystem::remove_all(root_); }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return root_; }

private:
    void write(const std::filesystem::path& relative)
    {
        const auto path = root_ / relative;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary) << "model";
    }

    std::filesystem::path root_;
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
             "--output", "result.png", "--detector", input_path, "--face68", input_path,
             "--arcface", input_path, "--inswapper", input_path, "--matrix", input_path };
}

} // namespace

spec("face swap CLI arguments")
{
    it("parses every required path and explicit optional models")
    {
        TempFile file;
        std::vector<std::string> values = required_arguments(file.path);
        values.insert(values.end(), { "--gfpgan", file.path, "--age-gender", file.path });
        const Arguments arguments = parse_arguments(values);

        check(arguments.source == file.path);
        check(arguments.target == file.path);
        check(arguments.output == "result.png");
        check(arguments.detector == file.path);
        check(arguments.face68 == file.path);
        check(arguments.arcface == file.path);
        check(arguments.inswapper == file.path);
        check(arguments.matrix == file.path);
        check_true(arguments.gfpgan.has_value());
        check_true(arguments.age_gender.has_value());
    }

    it("derives required inference paths from the configured model root")
    {
        TempFile        input;
        ModelRootFixture models;
        const Arguments arguments = parse_arguments(
            { "face_swap_image", "--source", input.path, "--target", input.path,
              "--output", "result.png" },
            models.path(), "test-profile");

        const auto engine_root = models.path() / "tensorrt" / "test-profile";
        check(arguments.detector == (engine_root / "yolov12n-face.engine").string());
        check(arguments.face68 == (engine_root / "2dfan4.engine").string());
        check(arguments.arcface == (engine_root / "arcface_w600k_r50.engine").string());
        check(arguments.inswapper == (engine_root / "inswapper_128.engine").string());
        check(arguments.matrix == (models.path() / "model_matrix.bin").string());
        check_false(arguments.gfpgan.has_value());
        check_false(arguments.age_gender.has_value());

        std::vector<std::string> explicit_values = required_arguments(input.path);
        const Arguments explicit_models =
            parse_arguments(explicit_values, "Z:/unused-model-root", "unused-profile");
        check(explicit_models.detector == input.path);
    }

    it("rejects missing, duplicate, unknown, and valueless options")
    {
        TempFile file;
        std::vector<std::string> missing = required_arguments(file.path);
        missing.erase(missing.begin() + 1, missing.begin() + 3);
        expect_failure(missing, "--source");

        std::vector<std::string> duplicate = required_arguments(file.path);
        duplicate.insert(duplicate.end(), { "--matrix", file.path });
        expect_failure(duplicate, "duplicate");

        std::vector<std::string> unknown = required_arguments(file.path);
        unknown.insert(unknown.end(), { "--fallback-face", file.path });
        expect_failure(unknown, "unknown");

        std::vector<std::string> valueless = required_arguments(file.path);
        valueless.emplace_back("--gfpgan");
        expect_failure(valueless, "missing value");
    }

    it("rejects nonexistent input, engine, and matrix files")
    {
        TempFile file;
        std::vector<std::string> values = required_arguments(file.path);
        values[2] = "this-face-swap-file-must-not-exist.jpg";
        expect_failure(values, "--source");

        values = required_arguments(file.path);
        values[8] = "this-face-swap-engine-must-not-exist.engine";
        expect_failure(values, "--detector");
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
