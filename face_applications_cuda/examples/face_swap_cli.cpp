#include "face_swap_cli.hpp"

#include "turbo_fs.h"

#include <array>
#include <cstddef>
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
    std::string Arguments::*member;
    bool input_file;
};

constexpr std::array<OptionBinding, 3> kBindings = {
    OptionBinding { "--source", &Arguments::source, true },
    OptionBinding { "--target", &Arguments::target, true },
    OptionBinding { "--output", &Arguments::output, false },
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
    return !(arguments.*binding.member).empty();
}

void assign(Arguments& arguments, const OptionBinding& binding, std::string value)
{
    if (is_set(arguments, binding))
    {
        fail(std::string("duplicate option: ") + binding.name);
    }
    arguments.*binding.member = std::move(value);
}

const std::string& value_of(const Arguments& arguments, const OptionBinding& binding)
{
    return arguments.*binding.member;
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

} // namespace

Arguments parse_arguments(const std::vector<std::string>& values)
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

    for (const OptionBinding& binding : kBindings)
    {
        if (!is_set(arguments, binding))
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

} // namespace kfcore::face_applications::cli
