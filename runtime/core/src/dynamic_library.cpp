#include "dynamic_library.hpp"

#include "kfcore/runtime/error.hpp"

#include <system_error>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace kfcore::runtime::detail
{
namespace
{

std::filesystem::path canonical_library_path(const std::filesystem::path& path)
{
    if (path.empty())
    {
        throw RuntimeError(RuntimeErrorCode::InvalidArgument,
                           "runtime module path must not be empty");
    }

    std::error_code error;
    const std::filesystem::path canonical = std::filesystem::canonical(path, error);
    if (error || !std::filesystem::is_regular_file(canonical, error) || error)
    {
        throw RuntimeError(RuntimeErrorCode::FileIo,
                           "runtime module path is not a canonical regular file: " +
                               path.string());
    }
    return canonical;
}

} // namespace

DynamicLibrary::DynamicLibrary(std::filesystem::path path, void* handle) noexcept
    : path_(std::move(path))
    , handle_(handle)
{
}

std::shared_ptr<DynamicLibrary> DynamicLibrary::open(const std::filesystem::path& path)
{
    const std::filesystem::path canonical = canonical_library_path(path);

#if defined(_WIN32)
    const HMODULE module = LoadLibraryExW(
        canonical.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (module == nullptr)
    {
        throw RuntimeError(RuntimeErrorCode::ModuleLoad,
                           "LoadLibraryExW failed for runtime module: " + canonical.string());
    }
    return std::shared_ptr<DynamicLibrary>(
        new DynamicLibrary(canonical, reinterpret_cast<void*>(module)));
#else
    void* module = dlopen(canonical.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (module == nullptr)
    {
        const char* error = dlerror();
        throw RuntimeError(RuntimeErrorCode::ModuleLoad,
                           std::string("dlopen failed for runtime module: ") +
                               canonical.string() + (error == nullptr ? "" : ": ") +
                               (error == nullptr ? "" : error));
    }
    return std::shared_ptr<DynamicLibrary>(new DynamicLibrary(canonical, module));
#endif
}

DynamicLibrary::~DynamicLibrary()
{
    if (handle_ == nullptr)
    {
        return;
    }
#if defined(_WIN32)
    (void)FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
    (void)dlclose(handle_);
#endif
}

void* DynamicLibrary::symbol(const char* name) const
{
    if (name == nullptr || *name == '\0')
    {
        throw RuntimeError(RuntimeErrorCode::InvalidArgument,
                           "runtime module symbol name must not be empty");
    }
    if (handle_ == nullptr)
    {
        throw RuntimeError(RuntimeErrorCode::ModuleLoad,
                           "runtime module is not loaded");
    }

#if defined(_WIN32)
    const FARPROC value = GetProcAddress(reinterpret_cast<HMODULE>(handle_), name);
    if (value == nullptr)
    {
        throw RuntimeError(RuntimeErrorCode::SymbolLookup,
                           std::string("GetProcAddress failed for symbol: ") + name);
    }
    return reinterpret_cast<void*>(value);
#else
    dlerror();
    void* value = dlsym(handle_, name);
    const char* error = dlerror();
    if (error != nullptr)
    {
        throw RuntimeError(RuntimeErrorCode::SymbolLookup,
                           std::string("dlsym failed for symbol: ") + name + ": " + error);
    }
    return value;
#endif
}

const std::filesystem::path& DynamicLibrary::path() const noexcept
{
    return path_;
}

} // namespace kfcore::runtime::detail
