#include <kfcore/gesture_interaction/experiment.hpp>
#include <windows.h>
#include <stdexcept>

namespace kfcore::gesture_interaction {
namespace {
[[noreturn]] void file_error(const char* operation) {
    throw std::runtime_error(std::string("experiment ")+operation+" failed; Windows error="+std::to_string(GetLastError()));
}
class File {
public:
    HANDLE handle;
    explicit File(HANDLE value) : handle(value) { if (handle == INVALID_HANDLE_VALUE) file_error("open"); }
    ~File() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    void close() {
        const auto old = handle; handle = INVALID_HANDLE_VALUE;
        if (!CloseHandle(old)) file_error("close");
    }
};
void check_path(const std::filesystem::path& path) {
    if (path.empty() || path.extension() != L".kfesn") throw std::invalid_argument("choose a .kfesn experiment file");
}
}
void save_experiment(const std::filesystem::path& path, const ExperimentArchive& archive) {
    check_path(path);
    const auto bytes = encode_experiment(archive);
    if (std::filesystem::exists(path)) throw std::runtime_error("experiment already exists; choose a NEW filename");
    auto temporary = path; temporary += L".partial";
    // Salts FS only exposes replacing create/rename. Keep exclusive publication
    // in this Windows adapter; never use a check-then-truncate save path.
    File file(CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr));
    try {
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            DWORD written = 0;
            if (!WriteFile(file.handle, bytes.data()+offset, DWORD(bytes.size()-offset), &written, nullptr) || !written)
                file_error("write");
            offset += written;
        }
        if (!FlushFileBuffers(file.handle)) file_error("flush");
        file.close();
        if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_WRITE_THROUGH)) file_error("publish without overwrite");
    } catch (...) {
        if (file.handle != INVALID_HANDLE_VALUE) file.close();
        if (!DeleteFileW(temporary.c_str()))
            throw std::runtime_error("save failed; owned .partial file could not be removed; inspect it before retrying");
        throw;
    }
}
ExperimentArchive load_experiment(const std::filesystem::path& path, const std::string& expected_pipeline) {
    check_path(path);
    File file(CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
    LARGE_INTEGER size{};
    if (GetFileType(file.handle) != FILE_TYPE_DISK || !GetFileSizeEx(file.handle, &size)) file_error("file size");
    if (size.QuadPart < 0 || std::uint64_t(size.QuadPart) > kMaximumExperimentBytes)
        throw std::invalid_argument("experiment file size outside bounds");
    std::vector<std::uint8_t> bytes(std::size_t(size.QuadPart));
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        DWORD read = 0;
        if (!ReadFile(file.handle, bytes.data()+offset, DWORD(bytes.size()-offset), &read, nullptr) || !read)
            file_error("read");
        offset += read;
    }
    file.close();
    return decode_experiment(bytes, expected_pipeline);
}
} // namespace kfcore::gesture_interaction
