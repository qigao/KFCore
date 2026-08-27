#pragma once

#include "kfcore/face_models/types.hpp"

#include <filesystem>
#include <memory>

namespace kfcore::face_models
{

class InSwapperEmbeddingProjector final
{
public:
    ~InSwapperEmbeddingProjector();

    InSwapperEmbeddingProjector(InSwapperEmbeddingProjector&&) noexcept;
    InSwapperEmbeddingProjector& operator=(InSwapperEmbeddingProjector&&) noexcept;
    InSwapperEmbeddingProjector(const InSwapperEmbeddingProjector&)            = delete;
    InSwapperEmbeddingProjector& operator=(const InSwapperEmbeddingProjector&) = delete;

    [[nodiscard]] static InSwapperEmbeddingProjector
    load(const std::filesystem::path& matrix_path);

    // Applies the model-specific row-major 512x512 projection and L2 normalization.
    [[nodiscard]] ArcFaceResult project(const ArcFaceResult& embedding) const;

private:
    struct Impl;
    explicit InSwapperEmbeddingProjector(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::face_models
