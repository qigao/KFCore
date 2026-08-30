#pragma once

#include "kfcore/image_processor/types.hpp"

#include <memory>
#include <string>

namespace kfcore::yolo::demo
{

class NativeWindow final
{
public:
    NativeWindow(std::string title, int width, int height);
    ~NativeWindow();
    NativeWindow(NativeWindow&&) noexcept;
    NativeWindow& operator=(NativeWindow&&) noexcept;
    NativeWindow(const NativeWindow&)            = delete;
    NativeWindow& operator=(const NativeWindow&) = delete;

    void present(const kfcore::image::BgrImage& image);
    [[nodiscard]] int poll_key() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::yolo::demo
