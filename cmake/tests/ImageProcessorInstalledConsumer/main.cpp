#include <kfcore/image_processor/image_processor.hpp>

int main()
{
    const auto transform = kfcore::image::ImageProcessor::letterbox_transform(2, 1, 2, 3);
    const kfcore::image::AffineTransform affine;
    const kfcore::image::CudaImageProcessorOptions options;
    return transform.scale == 1.0f && transform.pad_x == 0.0f && transform.pad_y == 1.0f &&
                   affine.destination_to_source[0] == 1.0f && options.device_id == 0
               ? 0
               : 1;
}
