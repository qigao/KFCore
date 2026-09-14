#pragma once

#include <filesystem>
#include <memory>

namespace kfcore::runtime::detail
{

class DynamicLibrary final
{
public:
    static std::shared_ptr<DynamicLibrary> open(const std::filesystem::path& path);

    ~DynamicLibrary();
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;

    [[nodiscard]] void* symbol(const char* name) const;
    [[nodiscard]] const std::filesystem::path& path() const noexcept;

private:
    DynamicLibrary(std::filesystem::path path, void* handle) noexcept;

    std::filesystem::path path_;
    void* handle_ = nullptr;
};

} // namespace kfcore::runtime::detail
