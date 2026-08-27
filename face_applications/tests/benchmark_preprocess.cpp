#include "kfcore/face_applications/geometry.hpp"
#include "kfcore/face_applications/preprocess.hpp"
#include "kfcore/image_processor/image_processor.hpp"
#include "tinytest.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

using namespace kfcore::face_applications;

namespace
{

constexpr std::size_t kWarmupIterations = 10;
constexpr std::size_t kMeasuredSamples  = 100;

cv::Mat patterned_bgr()
{
    cv::Mat image(1080, 1920, CV_8UC3);
    for (int row = 0; row < image.rows; ++row)
    {
        cv::Vec3b* pixels = image.ptr<cv::Vec3b>(row);
        for (int column = 0; column < image.cols; ++column)
        {
            pixels[column] = {
                static_cast<unsigned char>((row + column) % 256),
                static_cast<unsigned char>((row * 3 + column) % 256),
                static_cast<unsigned char>((row + column * 5) % 256),
            };
        }
    }
    return image;
}

FiveLandmarks source_landmarks()
{
    FiveLandmarks result = gfpgan_template();
    for (cv::Point2f& point : result)
    {
        point.x = point.x * 1.5F + 500.0F;
        point.y = point.y * 1.5F + 120.0F;
    }
    return result;
}

template <typename Operation> void run_benchmark(const char* name, Operation operation)
{
    float checksum = 0.0F;
    for (std::size_t iteration = 0; iteration < kWarmupIterations; ++iteration)
    {
        checksum += operation().front();
    }
    benchmark_batch(name, kMeasuredSamples)
    {
        checksum += operation().front();
    }
    check_true(checksum == checksum);
}

kfcore::image::ImageView image_view(const cv::Mat& image)
{
    return { image.data,
             image.total() * image.elemSize(),
             image.cols,
             image.rows,
             image.step[0],
             kfcore::image::PixelFormat::Bgr8,
             kfcore::image::MemoryKind::Host };
}

kfcore::image::AffineTransform affine(const cv::Matx23f& inverse)
{
    return { { inverse(0, 0), inverse(0, 1), inverse(0, 2), inverse(1, 0), inverse(1, 1),
               inverse(1, 2) } };
}

kfcore::image::PreprocessOptions preprocess_options(bool rgb, bool is_signed)
{
    kfcore::image::PreprocessOptions result;
    result.output_format =
        rgb ? kfcore::image::PixelFormat::Rgb8 : kfcore::image::PixelFormat::Bgr8;
    result.border_value = 0.0F;
    if (is_signed)
    {
        result.mean   = { 0.5F, 0.5F, 0.5F };
        result.stddev = { 0.5F, 0.5F, 0.5F };
    }
    return result;
}

std::vector<float> download(const kfcore::image::TensorView& tensor)
{
    std::vector<float> result(tensor.byte_size / sizeof(float));
    check(cudaMemcpy(result.data(), tensor.data, tensor.byte_size, cudaMemcpyDeviceToHost) ==
          cudaSuccess);
    return result;
}

void check_reference(const std::vector<float>& actual, const std::vector<float>& expected,
                     float tolerance)
{
    check(actual.size() == expected.size());
    float maximum_error = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index)
    {
        maximum_error = (std::max)(maximum_error, std::fabs(actual[index] - expected[index]));
    }
    check(maximum_error <= tolerance);
}

} // namespace

suite("face preprocessing benchmarks")
{
    bench("OpenCV affine plus CPU normalization baseline")
    {
        const cv::Mat       image = patterned_bgr();
        const FaceBox       box { 650.0F, 250.0F, 1250.0F, 850.0F };
        const FiveLandmarks landmarks = source_landmarks();

        run_benchmark("Face68 256x256",
                      [&] { return preprocess_face68(crop_face68(image, box).image); });
        run_benchmark("ArcFace 112x112",
                      [&]
                      {
                          return preprocess_arcface(
                              align_face(image, landmarks, arcface_template(), 112).image);
                      });
        run_benchmark("InSwapper 128x128",
                      [&]
                      {
                          return preprocess_inswapper(
                              align_face(image, landmarks, inswapper_template(), 128).image);
                      });
        run_benchmark("GFPGAN 512x512",
                      [&]
                      {
                          return preprocess_gfpgan(
                              align_face(image, landmarks, gfpgan_template(), 512).image);
                      });

        kfcore::image::CudaImageProcessorOptions processor_options;
        processor_options.max_source_bytes = 16U * 1024U * 1024U;
        processor_options.max_tensor_bytes = 16U * 1024U * 1024U;
        auto processor = kfcore::image::CudaImageProcessor::create(processor_options);
        kfcore::image::ImageView staged    = processor->stage(image_view(image));
        const FaceTransform      face68    = face68_transform(box);
        const FaceTransform      arcface   = alignment_transform(landmarks, arcface_template());
        const FaceTransform      inswapper = alignment_transform(landmarks, inswapper_template());
        const FaceTransform      gfpgan    = alignment_transform(landmarks, gfpgan_template());

        check_reference(
            download(processor->process_affine(staged, 256, 256, affine(face68.aligned_to_source),
                                               preprocess_options(false, false))),
            preprocess_face68(crop_face68(image, box).image), 0.035F);
        check_reference(
            download(processor->process_affine(staged, 112, 112, affine(arcface.aligned_to_source),
                                               preprocess_options(true, true))),
            preprocess_arcface(align_face(image, landmarks, arcface_template(), 112).image), 0.07F);
        check_reference(
            download(processor->process_affine(staged, 128, 128,
                                               affine(inswapper.aligned_to_source),
                                               preprocess_options(true, false))),
            preprocess_inswapper(align_face(image, landmarks, inswapper_template(), 128).image),
            0.035F);
        check_reference(
            download(processor->process_affine(staged, 512, 512, affine(gfpgan.aligned_to_source),
                                               preprocess_options(true, true))),
            preprocess_gfpgan(align_face(image, landmarks, gfpgan_template(), 512).image), 0.07F);

        const auto run_cuda = [&](const char* name, int extent, const FaceTransform& transform,
                                  const kfcore::image::PreprocessOptions& options)
        {
            void* output = nullptr;
            for (std::size_t iteration = 0; iteration < kWarmupIterations; ++iteration)
            {
                output = processor
                             ->process_affine(staged, extent, extent,
                                              affine(transform.aligned_to_source), options)
                             .data;
            }
            benchmark_batch(name, kMeasuredSamples)
            {
                output = processor
                             ->process_affine(staged, extent, extent,
                                              affine(transform.aligned_to_source), options)
                             .data;
            }
            check(output != nullptr);
        };
        benchmark_batch("CUDA stage 1080p frame", kMeasuredSamples)
        {
            staged = processor->stage(image_view(image));
        }
        run_cuda("CUDA Face68 256x256", 256, face68, preprocess_options(false, false));
        run_cuda("CUDA ArcFace 112x112", 112, arcface, preprocess_options(true, true));
        run_cuda("CUDA InSwapper 128x128", 128, inswapper, preprocess_options(true, false));
        run_cuda("CUDA GFPGAN 512x512", 512, gfpgan, preprocess_options(true, true));
    }
}
