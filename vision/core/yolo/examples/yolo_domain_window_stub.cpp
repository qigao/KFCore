#include "yolo_domain_window.hpp"

#include <stdexcept>
#include <utility>

namespace kfcore::yolo::demo
{

struct NativeWindow::Impl final {};

NativeWindow::NativeWindow(std::string, int, int)
{
    throw std::runtime_error(
        "interactive YOLO display is not implemented on this platform; use --headless");
}

NativeWindow::~NativeWindow() = default;
NativeWindow::NativeWindow(NativeWindow&&) noexcept = default;
NativeWindow& NativeWindow::operator=(NativeWindow&&) noexcept = default;

void NativeWindow::present(const kfcore::image::BgrImage&)
{
    throw std::logic_error("interactive YOLO display is unavailable");
}

int NativeWindow::poll_key() noexcept
{
    return 27;
}

} // namespace kfcore::yolo::demo
