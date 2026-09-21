#include "preprocess.hpp"

#include "kfcore/face_applications/application.hpp"
#include "kfcore/image_processor/cpu.hpp"

#include <array>
#include <string>

namespace kfcore::face_applications::detail
{
namespace
{

void require_extent(const kfcore::image::BgrImage& image, int extent, const char* model)
{
    if (image.width != extent || image.height != extent)
    {
        throw FaceApplicationError(FaceApplicationErrorCode::InvalidArgument,
                                      std::string(model) +
                                          " input must have the required square extent");
    }
}

kfcore::image::PreprocessOptions bgr_unit()
{
    kfcore::image::PreprocessOptions options;
    options.output_format = kfcore::image::PixelFormat::Bgr8;
    return options;
}

kfcore::image::PreprocessOptions rgb_unit()
{
    kfcore::image::PreprocessOptions options;
    options.output_format = kfcore::image::PixelFormat::Rgb8;
    return options;
}

kfcore::image::PreprocessOptions rgb_signed()
{
    auto options = rgb_unit();
    options.mean = { 0.5F, 0.5F, 0.5F };
    options.stddev = { 0.5F, 0.5F, 0.5F };
    return options;
}

} // namespace

std::vector<float> preprocess_face68(const kfcore::image::BgrImage& image,
                                     std::size_t max_tensor_bytes)
{
    require_extent(image, 256, "Face68");
    return kfcore::image::CpuImageProcessor::to_nchw(image, bgr_unit(), max_tensor_bytes);
}

std::vector<float> preprocess_arcface(const kfcore::image::BgrImage& image,
                                      std::size_t max_tensor_bytes)
{
    require_extent(image, 112, "ArcFace");
    return kfcore::image::CpuImageProcessor::to_nchw(image, rgb_signed(), max_tensor_bytes);
}

std::vector<float> preprocess_inswapper(const kfcore::image::BgrImage& image,
                                        std::size_t max_tensor_bytes)
{
    require_extent(image, 128, "InSwapper");
    return kfcore::image::CpuImageProcessor::to_nchw(image, rgb_unit(), max_tensor_bytes);
}

std::vector<float> preprocess_gfpgan(const kfcore::image::BgrImage& image,
                                     std::size_t max_tensor_bytes)
{
    require_extent(image, 512, "GFPGAN");
    return kfcore::image::CpuImageProcessor::to_nchw(image, rgb_signed(), max_tensor_bytes);
}

std::vector<float> preprocess_age_gender(const kfcore::image::BgrImage& image,
                                         std::size_t max_image_bytes,
                                         std::size_t max_tensor_bytes)
{
    const kfcore::image::BgrImage resized =
        kfcore::image::CpuImageProcessor::resize_bgr(image, 224, 224, max_image_bytes);
    auto options = rgb_unit();
    options.mean = { 0.485F, 0.456F, 0.406F };
    options.stddev = { 0.229F, 0.224F, 0.225F };
    return kfcore::image::CpuImageProcessor::to_nchw(resized, options, max_tensor_bytes);
}

} // namespace kfcore::face_applications::detail

