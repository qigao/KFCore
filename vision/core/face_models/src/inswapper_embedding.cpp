#include "kfcore/face_models/inswapper_embedding.hpp"

#include "kfcore/face_models/error.hpp"
#include "salts_fs.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::face_models
{
namespace
{

constexpr std::size_t kMatrixElementCount =
    kInSwapperEmbeddingLength * kInSwapperEmbeddingLength;
constexpr std::size_t kMatrixByteCount = kMatrixElementCount * sizeof(float);

[[noreturn]] void throw_asset(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::InvalidModelAsset,
                         "InSwapper embedding matrix validation stage: " + detail);
}

struct FileBuffer final
{
    ~FileBuffer()
    {
        salts_fs_buf_free(&value);
    }

    salts_fs_buf_t value {};
};

double finite_norm(const ArcFaceResult& values, const char* subject)
{
    double squared_norm = 0.0;
    for (float value : values)
    {
        if (!std::isfinite(value))
        {
            throw_asset(std::string(subject) + " values must be finite");
        }
        const double wide_value = static_cast<double>(value);
        squared_norm += wide_value * wide_value;
    }
    const double norm = std::sqrt(squared_norm);
    if (!std::isfinite(norm) || norm <= (std::numeric_limits<double>::min)())
    {
        throw_asset(std::string(subject) + " norm must be finite and positive");
    }
    return norm;
}

} // namespace

struct InSwapperEmbeddingProjector::Impl final
{
    explicit Impl(std::vector<float> matrix_in)
        : matrix(std::move(matrix_in))
    {
    }

    std::vector<float> matrix;
};

InSwapperEmbeddingProjector::InSwapperEmbeddingProjector(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

InSwapperEmbeddingProjector::~InSwapperEmbeddingProjector() = default;
InSwapperEmbeddingProjector::InSwapperEmbeddingProjector(
    InSwapperEmbeddingProjector&&) noexcept = default;
InSwapperEmbeddingProjector& InSwapperEmbeddingProjector::operator=(
    InSwapperEmbeddingProjector&&) noexcept = default;

InSwapperEmbeddingProjector
InSwapperEmbeddingProjector::load(const std::filesystem::path& matrix_path)
{
    try
    {
        if (matrix_path.empty())
        {
            throw_asset("path must not be empty");
        }
        const std::string encoded_path = matrix_path.u8string();
        FileBuffer file;
        if (salts_fs_read_file(encoded_path.c_str(), &file.value) != 0)
        {
            throw_asset("file cannot be read: " + encoded_path);
        }
        if (file.value.len != kMatrixByteCount)
        {
            throw_asset("file must contain exactly " + std::to_string(kMatrixByteCount) +
                        " bytes");
        }

        std::vector<float> matrix(kMatrixElementCount);
        std::memcpy(matrix.data(), file.value.base, kMatrixByteCount);
        for (float value : matrix)
        {
            if (!std::isfinite(value))
            {
                throw_asset("all matrix elements must be finite");
            }
        }
        return InSwapperEmbeddingProjector(
            std::make_unique<Impl>(std::move(matrix)));
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const std::bad_alloc&)
    {
        throw_asset("allocation failed");
    }
}

ArcFaceResult InSwapperEmbeddingProjector::project(const ArcFaceResult& embedding) const
{
    const double source_norm = finite_norm(embedding, "source embedding");
    ArcFaceResult projected {};
    for (std::size_t column = 0; column < kInSwapperEmbeddingLength; ++column)
    {
        double value = 0.0;
        for (std::size_t row = 0; row < kInSwapperEmbeddingLength; ++row)
        {
            value += (static_cast<double>(embedding[row]) / source_norm) *
                     static_cast<double>(impl_->matrix[row * kInSwapperEmbeddingLength + column]);
        }
        if (!std::isfinite(value))
        {
            throw_asset("projected values must remain finite");
        }
        projected[column] = static_cast<float>(value);
    }

    const double projected_norm = finite_norm(projected, "projected embedding");
    for (float& value : projected)
    {
        value = static_cast<float>(static_cast<double>(value) / projected_norm);
    }
    return projected;
}

} // namespace kfcore::face_models
