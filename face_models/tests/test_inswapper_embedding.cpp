#include "kfcore/face_models/error.hpp"
#include "kfcore/face_models/inswapper_embedding.hpp"
#include "tinytest.hpp"

#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <vector>

using namespace kfcore::face_models;

namespace
{

constexpr std::size_t kMatrixElementCount =
    kInSwapperEmbeddingLength * kInSwapperEmbeddingLength;

struct TempFile final
{
    explicit TempFile(const std::vector<float>& values)
        : path(tt_make_temp_file("inswapper-matrix", ".bin"))
    {
        check_not_null(path);
        if (path != nullptr)
        {
            check(tt_write_file(path, values.data(), values.size() * sizeof(float)) == 0);
        }
    }

    ~TempFile()
    {
        if (path != nullptr)
        {
            (void)tt_remove_file(path);
            std::free(path);
        }
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    char* path = nullptr;
};

void expect_error(const std::function<void()>& operation, const std::string& message)
{
    bool threw = false;
    try
    {
        operation();
    }
    catch (const FaceModelError& error)
    {
        threw = true;
        check(error.code() == FaceModelErrorCode::InvalidModelAsset);
        check(std::string(error.what()).find(message) != std::string::npos);
    }
    check_true(threw);
}

std::vector<float> identity_matrix()
{
    std::vector<float> matrix(kMatrixElementCount, 0.0F);
    for (std::size_t index = 0; index < kInSwapperEmbeddingLength; ++index)
    {
        matrix[index * kInSwapperEmbeddingLength + index] = 1.0F;
    }
    return matrix;
}

} // namespace

spec("InSwapper embedding projector")
{
    it("rejects short and trailing matrix data")
    {
        TempFile short_file(std::vector<float>(kMatrixElementCount - 1U, 0.0F));
        expect_error([&] { (void)InSwapperEmbeddingProjector::load(short_file.path); },
                     "exactly");

        TempFile long_file(std::vector<float>(kMatrixElementCount + 1U, 0.0F));
        expect_error([&] { (void)InSwapperEmbeddingProjector::load(long_file.path); },
                     "exactly");
    }

    it("rejects non-finite matrix elements")
    {
        std::vector<float> matrix = identity_matrix();
        matrix[17] = (std::numeric_limits<float>::quiet_NaN)();
        TempFile file(matrix);
        expect_error([&] { (void)InSwapperEmbeddingProjector::load(file.path); }, "finite");
    }

    it("multiplies row-major and L2 normalizes the projected embedding")
    {
        std::vector<float> matrix(kMatrixElementCount, 0.0F);
        matrix[0U * kInSwapperEmbeddingLength + 2U] = 2.0F;
        matrix[1U * kInSwapperEmbeddingLength + 3U] = 4.0F;
        TempFile file(matrix);
        const InSwapperEmbeddingProjector projector =
            InSwapperEmbeddingProjector::load(file.path);

        ArcFaceResult source {};
        source[0] = 3.0F;
        source[1] = 2.0F;
        const ArcFaceResult projected = projector.project(source);

        check(std::fabs(projected[2] - 0.6F) <= 1.0e-6F);
        check(std::fabs(projected[3] - 0.8F) <= 1.0e-6F);
        check(std::fabs(projected[0]) <= 1.0e-6F);
    }

    it("rejects non-finite input and zero-norm projection")
    {
        TempFile identity(identity_matrix());
        const InSwapperEmbeddingProjector projector =
            InSwapperEmbeddingProjector::load(identity.path);

        ArcFaceResult invalid {};
        invalid[4] = (std::numeric_limits<float>::infinity)();
        expect_error([&] { (void)projector.project(invalid); }, "finite");

        TempFile zero(std::vector<float>(kMatrixElementCount, 0.0F));
        const InSwapperEmbeddingProjector zero_projector =
            InSwapperEmbeddingProjector::load(zero.path);
        ArcFaceResult source {};
        source[0] = 1.0F;
        expect_error([&] { (void)zero_projector.project(source); }, "norm");
    }
}
