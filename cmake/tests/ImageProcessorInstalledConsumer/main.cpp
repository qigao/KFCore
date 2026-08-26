#include <kfcore/image_processor/image_processor.hpp>

int main()
{
    const auto transform = kfcore::image::ImageProcessor::letterbox_transform(2, 1, 2, 3);
    return transform.scale == 1.0f && transform.pad_x == 0.0f && transform.pad_y == 1.0f ? 0 : 1;
}
