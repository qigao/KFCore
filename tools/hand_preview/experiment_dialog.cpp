#include "experiment_dialog.hpp"
#include <windows.h>
#include <commdlg.h>
#include <array>
#include <stdexcept>
#include <string>

namespace preview {
std::optional<std::filesystem::path> choose_experiment_file(bool save) {
    constexpr std::size_t kPathCapacity = 32768;
    std::array<wchar_t, kPathCapacity> path{};
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = GetActiveWindow();
    dialog.lpstrFilter = L"KFCore ESN experiment (*.kfesn)\0*.kfesn\0\0";
    dialog.lpstrFile = path.data(); dialog.nMaxFile = DWORD(path.size());
    dialog.lpstrDefExt = L"kfesn";
    dialog.lpstrTitle = save ? L"Save experiment as NEW file (no overwrite)" : L"Load complete experiment";
    dialog.Flags = OFN_EXPLORER | OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST |
        (save ? 0 : OFN_FILEMUSTEXIST);
    const bool selected = save ? GetSaveFileNameW(&dialog) != FALSE : GetOpenFileNameW(&dialog) != FALSE;
    if (selected) return std::filesystem::path(path.data());
    const auto error = CommDlgExtendedError();
    if (error) throw std::runtime_error("experiment file dialog failed: "+std::to_string(error));
    return std::nullopt;
}
bool confirm_discard_experiment() {
    return MessageBoxW(GetActiveWindow(), L"Replace the current unsaved samples/model with the loaded experiment?\nChoose No to save first.",
        L"Unsaved experiment", MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2) == IDYES;
}
}
