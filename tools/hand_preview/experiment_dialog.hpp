#pragma once
#include <filesystem>
#include <optional>
namespace preview {
std::optional<std::filesystem::path> choose_experiment_file(bool save);
bool confirm_discard_experiment();
}
