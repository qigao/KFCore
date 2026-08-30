#include "face_swap_cli.hpp"

#include "turbo_fs.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::face_applications::cli
{
namespace
{

struct OptionBinding
{
    const char* name;
    std::string Arguments::*required_member;
    std::optional<std::string> Arguments::*optional_member;
    bool input_file;
};

constexpr std::array<OptionBinding, 10> kBindings = {
    OptionBinding { "--source", &Arguments::source, nullptr, true },
    OptionBinding { "--target", &Arguments::target, nullptr, true },
    OptionBinding { "--output", &Arguments::output, nullptr, false },
    OptionBinding { "--detector", &Arguments::detector, nullptr, true },
    OptionBinding { "--face68", &Arguments::face68, nullptr, true },
    OptionBinding { "--arcface", &Arguments::arcface, nullptr, true },
    OptionBinding { "--inswapper", &Arguments::inswapper, nullptr, true },
    OptionBinding { "--matrix", &Arguments::matrix, nullptr, true },
    OptionBinding { "--gfpgan", nullptr, &Arguments::gfpgan, true },
    OptionBinding { "--age-gender", nullptr, &Arguments::age_gender, true },
};

[[noreturn]] void fail(const std::string& message)
{
    throw std::invalid_argument("face_swap_image arguments: " + message);
}

const OptionBinding& find_binding(const std::string& option)
{
    for (const OptionBinding& binding : kBindings)
    {
        if (option == binding.name)
        {
            return binding;
        }
    }
    fail("unknown option: " + option);
}

bool is_set(const Arguments& arguments, const OptionBinding& binding)
{
    if (binding.required_member != nullptr)
    {
        return !(arguments.*binding.required_member).empty();
    }
    return (arguments.*binding.optional_member).has_value();
}

void assign(Arguments& arguments, const OptionBinding& binding, std::string value)
{
    if (is_set(arguments, binding))
    {
        fail(std::string("duplicate option: ") + binding.name);
    }
    if (binding.required_member != nullptr)
    {
        arguments.*binding.required_member = std::move(value);
    }
    else
    {
        arguments.*binding.optional_member = std::move(value);
    }
}

const std::string& value_of(const Arguments& arguments, const OptionBinding& binding)
{
    if (binding.required_member != nullptr)
    {
        return arguments.*binding.required_member;
    }
    return *(arguments.*binding.optional_member);
}

void require_readable_file(const std::string& path, const char* option)
{
    turbo_fs_stat_t status {};
    if (turbo_fs_stat(path.c_str(), &status) != 0 || !status.is_file ||
        turbo_fs_access(path.c_str(), TURBO_FS_ACCESS_READ) != 0)
    {
        fail(std::string(option) + " is not a readable file: " + path);
    }
}

bool resolves_to_same_file(const std::string& left, const std::string& right)
{
    namespace fs = std::filesystem;
    std::error_code error;
    if (fs::equivalent(fs::path(left), fs::path(right), error))
    {
        return true;
    }
    error.clear();
    const fs::path normalized_left = fs::weakly_canonical(fs::path(left), error);
    if (error)
    {
        fail("cannot resolve path: " + left);
    }
    const fs::path normalized_right = fs::weakly_canonical(fs::path(right), error);
    if (error)
    {
        fail("cannot resolve path: " + right);
    }
    return normalized_left == normalized_right;
}

void derive_required_model_paths(Arguments& arguments,
                                 const std::filesystem::path& model_root,
                                 const std::string& tensorrt_profile)
{
    if (model_root.empty())
    {
        return;
    }
    if (arguments.matrix.empty())
    {
        arguments.matrix = (model_root / "model_matrix.bin").string();
    }
    if (tensorrt_profile.empty())
    {
        return;
    }
    const std::filesystem::path root = model_root / "tensorrt" / tensorrt_profile;
    if (arguments.detector.empty())
        arguments.detector = (root / "yolov12n-face.engine").string();
    if (arguments.face68.empty())
        arguments.face68 = (root / "2dfan4.engine").string();
    if (arguments.arcface.empty())
        arguments.arcface = (root / "arcface_w600k_r50.engine").string();
    if (arguments.inswapper.empty())
        arguments.inswapper = (root / "inswapper_128.engine").string();
}

std::string environment_value(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr ? value : "";
}

} // namespace

Arguments parse_arguments(const std::vector<std::string>& values,
                          const std::filesystem::path& model_root,
                          const std::string& tensorrt_profile)
{
    if (values.empty())
    {
        fail("program name is missing");
    }

    Arguments arguments;
    for (std::size_t index = 1; index < values.size(); index += 2U)
    {
        const std::string& option = values[index];
        if (index + 1U >= values.size() || values[index + 1U].empty() ||
            values[index + 1U].rfind("--", 0U) == 0U)
        {
            fail("missing value for " + option);
        }
        const OptionBinding& binding = find_binding(option);
        assign(arguments, binding, values[index + 1U]);
    }

    derive_required_model_paths(arguments, model_root, tensorrt_profile);

    for (const OptionBinding& binding : kBindings)
    {
        if (binding.optional_member == nullptr && !is_set(arguments, binding))
        {
            fail(std::string("missing required option: ") + binding.name);
        }
        if (binding.input_file && is_set(arguments, binding))
        {
            require_readable_file(value_of(arguments, binding), binding.name);
        }
    }
    if (resolves_to_same_file(arguments.output, arguments.source) ||
        resolves_to_same_file(arguments.output, arguments.target))
    {
        fail("--output must differ from --source and --target");
    }
    return arguments;
}

Arguments parse_arguments_from_environment(const std::vector<std::string>& values)
{
    return parse_arguments(values, environment_value("KFCORE_MODEL_ROOT"),
                           environment_value("KFCORE_TENSORRT_ENGINE_PROFILE"));
}

} // namespace kfcore::face_applications::cli
