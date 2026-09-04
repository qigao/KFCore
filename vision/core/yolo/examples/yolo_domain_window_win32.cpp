#include "yolo_domain_window.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kfcore::yolo::demo
{
namespace
{

constexpr wchar_t kWindowClassName[] = L"KFCoreYoloDomainWindow";
constexpr WORD kArrowCursorResource = 32512U;

std::wstring widen_title(const std::string& title)
{
    if (title.empty())
    {
        throw std::invalid_argument("window title must not be empty");
    }
    const int length = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, title.data(), static_cast<int>(title.size()),
        nullptr, 0);
    if (length <= 0)
    {
        throw std::invalid_argument("window title must be valid UTF-8");
    }
    std::wstring result(static_cast<std::size_t>(length), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, title.data(),
                            static_cast<int>(title.size()), result.data(), length) != length)
    {
        throw std::runtime_error("failed to convert the window title");
    }
    return result;
}

ATOM ensure_window_class(WNDPROC procedure)
{
    static const ATOM registered = [procedure]
    {
        WNDCLASSEXW window_class {};
        window_class.cbSize = sizeof(window_class);
        window_class.lpfnWndProc = procedure;
        window_class.hInstance = GetModuleHandleW(nullptr);
        window_class.hCursor = LoadCursorW(
            nullptr, MAKEINTRESOURCEW(kArrowCursorResource));
        window_class.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
        window_class.lpszClassName = kWindowClassName;
        const ATOM atom = RegisterClassExW(&window_class);
        if (atom == 0 && GetLastError() == ERROR_CLASS_ALREADY_EXISTS)
        {
            return static_cast<ATOM>(1);
        }
        return atom;
    }();
    return registered;
}

} // namespace

struct NativeWindow::Impl final
{
    static LRESULT CALLBACK window_proc(HWND window, UINT message,
                                        WPARAM word, LPARAM value)
    {
        Impl* self = reinterpret_cast<Impl*>(GetWindowLongPtrW(window, GWLP_USERDATA));
        if (message == WM_NCCREATE)
        {
            const auto* create = reinterpret_cast<const CREATESTRUCTW*>(value);
            self = static_cast<Impl*>(create->lpCreateParams);
            SetWindowLongPtrW(window, GWLP_USERDATA,
                              reinterpret_cast<LONG_PTR>(self));
        }
        if (self != nullptr)
        {
            if (message == WM_CLOSE)
            {
                self->closed = true;
                ShowWindow(window, SW_HIDE);
                return 0;
            }
            if (message == WM_KEYDOWN)
            {
                self->pending_key = static_cast<int>(word);
                return 0;
            }
        }
        return DefWindowProcW(window, message, word, value);
    }

    Impl(const std::string& title, int width, int height)
    {
        if (width <= 0 || height <= 0)
        {
            throw std::invalid_argument("window dimensions must be positive");
        }
        if (ensure_window_class(&Impl::window_proc) == 0)
        {
            throw std::runtime_error("failed to register the Win32 window class");
        }
        RECT rectangle { 0, 0, width, height };
        constexpr DWORD kWindowStyle = WS_OVERLAPPEDWINDOW;
        if (AdjustWindowRect(&rectangle, kWindowStyle, FALSE) == 0)
        {
            throw std::runtime_error("failed to calculate the Win32 window size");
        }
        const std::wstring wide_title = widen_title(title);
        window = CreateWindowExW(
            0, kWindowClassName, wide_title.c_str(), kWindowStyle,
            CW_USEDEFAULT, CW_USEDEFAULT, rectangle.right - rectangle.left,
            rectangle.bottom - rectangle.top, nullptr, nullptr,
            GetModuleHandleW(nullptr), this);
        if (window == nullptr)
        {
            throw std::runtime_error("failed to create the Win32 display window");
        }
        ShowWindow(window, SW_SHOW);
        UpdateWindow(window);
    }

    ~Impl()
    {
        if (window != nullptr)
        {
            DestroyWindow(window);
        }
    }

    void pump() noexcept
    {
        MSG message {};
        while (PeekMessageW(&message, window, 0, 0, PM_REMOVE) != 0)
        {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }

    HWND window = nullptr;
    bool closed = false;
    int pending_key = -1;
    std::vector<std::uint8_t> padded_rows;
};

NativeWindow::NativeWindow(std::string title, int width, int height)
    : impl_(std::make_unique<Impl>(title, width, height))
{
}

NativeWindow::~NativeWindow() = default;
NativeWindow::NativeWindow(NativeWindow&&) noexcept = default;
NativeWindow& NativeWindow::operator=(NativeWindow&&) noexcept = default;

void NativeWindow::present(const kfcore::image::BgrImage& image)
{
    if (!impl_ || image.width <= 0 || image.height <= 0)
    {
        throw std::invalid_argument("display window and image must be valid");
    }
    const std::size_t row_bytes = static_cast<std::size_t>(image.width) * 3U;
    if (static_cast<std::size_t>(image.height) >
            (std::numeric_limits<std::size_t>::max)() / row_bytes ||
        image.pixels.size() != row_bytes * static_cast<std::size_t>(image.height))
    {
        throw std::invalid_argument("display image storage is malformed");
    }
    impl_->pump();
    if (impl_->closed)
    {
        return;
    }

    const std::size_t dib_stride = (row_bytes + 3U) & ~std::size_t { 3U };
    const std::uint8_t* pixels = image.pixels.data();
    if (dib_stride != row_bytes)
    {
        impl_->padded_rows.resize(dib_stride * static_cast<std::size_t>(image.height));
        for (int row = 0; row < image.height; ++row)
        {
            std::copy_n(image.pixels.data() + static_cast<std::size_t>(row) * row_bytes,
                        row_bytes,
                        impl_->padded_rows.data() + static_cast<std::size_t>(row) * dib_stride);
        }
        pixels = impl_->padded_rows.data();
    }

    BITMAPINFO bitmap {};
    bitmap.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmap.bmiHeader.biWidth = image.width;
    bitmap.bmiHeader.biHeight = -image.height;
    bitmap.bmiHeader.biPlanes = 1;
    bitmap.bmiHeader.biBitCount = 24;
    bitmap.bmiHeader.biCompression = BI_RGB;
    RECT client {};
    GetClientRect(impl_->window, &client);
    HDC device = GetDC(impl_->window);
    if (device == nullptr)
    {
        throw std::runtime_error("failed to acquire the Win32 display context");
    }
    SetStretchBltMode(device, COLORONCOLOR);
    const int result = StretchDIBits(
        device, 0, 0, client.right - client.left, client.bottom - client.top,
        0, 0, image.width, image.height, pixels, &bitmap, DIB_RGB_COLORS, SRCCOPY);
    ReleaseDC(impl_->window, device);
    if (result == GDI_ERROR)
    {
        throw std::runtime_error("Win32 failed to present the BGR frame");
    }
}

int NativeWindow::poll_key() noexcept
{
    if (!impl_)
    {
        return 27;
    }
    impl_->pump();
    if (impl_->closed)
    {
        return 27;
    }
    const int result = impl_->pending_key;
    impl_->pending_key = -1;
    return result;
}

} // namespace kfcore::yolo::demo
