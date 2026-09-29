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
    dialog.lpstrFilter = L"KFCore ESN 实验 (*.kfesn)\0*.kfesn\0\0";
    dialog.lpstrFile = path.data(); dialog.nMaxFile = DWORD(path.size());
    dialog.lpstrDefExt = L"kfesn";
    dialog.lpstrTitle = save ? L"另存实验为新文件（不可覆盖）" : L"加载完整实验";
    dialog.Flags = OFN_EXPLORER | OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST |
        (save ? 0 : OFN_FILEMUSTEXIST);
    const bool selected = save ? GetSaveFileNameW(&dialog) != FALSE : GetOpenFileNameW(&dialog) != FALSE;
    if (selected) return std::filesystem::path(path.data());
    const auto error = CommDlgExtendedError();
    if (error) throw std::runtime_error("实验文件对话框失败："+std::to_string(error));
    return std::nullopt;
}
bool confirm_discard_experiment() {
    return MessageBoxW(GetActiveWindow(), L"用加载的实验替换当前未保存的样本和模型吗？\n选“否”可先保存当前实验。",
        L"实验尚未保存", MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2) == IDYES;
}
}
