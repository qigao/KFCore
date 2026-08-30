#include "kfcore/yolo/onnx.hpp"

#include "compact_nms.hpp"
#include "kfcore/image_processor/cpu.hpp"
#include "kfcore/runtime_onnx/runtime.hpp"
#include "kfcore/yolo/error.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kfcore::yolo
{
namespace
{

constexpr std::size_t kChannels = 3U;
constexpr std::size_t kDetectionValues = 6U;

namespace onnx_runtime = kfcore::runtime_onnx;

[[noreturn]] void throw_invalid(const std::string& message)
{
    throw YoloError(YoloErrorCode::InvalidArgument, "ONNX detector: " + message);
}

[[noreturn]] void throw_file(const std::string& message)
{
    throw YoloError(YoloErrorCode::FileIo, "ONNX detector: " + message);
}

[[noreturn]] void throw_contract(const std::string& message)
{
    throw YoloError(YoloErrorCode::EngineContractMismatch,
                    "ONNX detector contract: " + message);
}

[[noreturn]] void throw_resource(const std::string& message)
{
    throw YoloError(YoloErrorCode::ResourceLimitExceeded,
                    "ONNX detector resource limit: " + message);
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char* object)
{
    if (left != 0U && right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_resource(std::string(object) + " byte count overflow");
    }
    return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right,
                        const char* object)
{
    if (right > (std::numeric_limits<std::size_t>::max)() - left)
    {
        throw_resource(std::string(object) + " byte count overflow");
    }
    return left + right;
}

void validate_options(const OnnxDetectorOptions& options)
{
    if (options.intra_op_threads < 0 || options.inter_op_threads < 0)
    {
        throw_invalid("thread counts must not be negative");
    }
    if (options.max_model_bytes == 0U || options.max_source_bytes == 0U ||
        options.max_tensor_bytes == 0U || options.max_output_bytes == 0U)
    {
        throw_resource("model, source, tensor, and output limits must be positive");
    }
    if (options.max_detections == 0U)
    {
        throw_resource("maximum detections must be positive");
    }
    if (!std::isfinite(options.border_value) || options.border_value < 0.0F ||
        options.border_value > 255.0F)
    {
        throw_invalid("border value must be finite within [0,255]");
    }
}

void validate_model_file(const std::filesystem::path& path, std::size_t max_bytes)
{
    if (path.empty())
    {
        throw_file("model path must not be empty");
    }
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
    {
        throw_file("model is not a readable regular file: " + path.u8string());
    }
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (error)
    {
        throw_file("model size cannot be read: " + path.u8string());
    }
    if (size == 0U || size > static_cast<std::uintmax_t>(max_bytes))
    {
        throw_resource("model bytes are zero or exceed the configured limit");
    }
}

onnx_runtime::SessionOptions runtime_options(const OnnxDetectorOptions& options)
{
    onnx_runtime::SessionOptions result;
    result.intra_op_threads = options.intra_op_threads;
    result.inter_op_threads = options.inter_op_threads;
    result.max_model_bytes = options.max_model_bytes;
    result.max_output_bytes = options.max_output_bytes;
    return result;
}

onnx_runtime::ModelContract model_contract()
{
    return { "YOLO ONNX detector",
             { { "images", onnx_runtime::ElementType::Float32, { 1, 3, -1, -1 } } },
             { { "output0", onnx_runtime::ElementType::Float32, { 1, -1, 6 } } } };
}

[[noreturn]] void rethrow_runtime(const onnx_runtime::Error& error)
{
    switch (error.code())
    {
    case onnx_runtime::ErrorCode::InvalidArgument:
    case onnx_runtime::ErrorCode::InvalidTensorView:
        throw_invalid(error.what());
    case onnx_runtime::ErrorCode::InvalidModelAsset:
        throw_file(error.what());
    case onnx_runtime::ErrorCode::ModelContractMismatch:
        throw_contract(error.what());
    case onnx_runtime::ErrorCode::ResourceLimitExceeded:
        throw_resource(error.what());
    case onnx_runtime::ErrorCode::RuntimeFailure:
        throw YoloError(YoloErrorCode::OnnxRuntimeFailure,
                        std::string("ONNX detector runtime: ") + error.what());
    }
    throw YoloError(YoloErrorCode::OnnxRuntimeFailure,
                    "ONNX detector runtime returned an unknown error");
}

std::size_t source_span(const ImageView& image)
{
    if (image.data == nullptr)
    {
        throw_invalid("image data must not be null");
    }
    if (image.width <= 0 || image.height <= 0)
    {
        throw_invalid("image dimensions must be positive");
    }
    if (image.memory_kind != MemoryKind::Host)
    {
        throw_invalid("CPU backend accepts Host images only");
    }
    const std::size_t width = static_cast<std::size_t>(image.width);
    const std::size_t height = static_cast<std::size_t>(image.height);
    if (image.pixel_format == PixelFormat::Nv12 ||
        image.pixel_format == PixelFormat::I420 ||
        image.pixel_format == PixelFormat::Nv21)
    {
        if ((image.width & 1) != 0 || (image.height & 1) != 0)
        {
            throw_invalid("NV12, I420, and NV21 image dimensions must be even");
        }
        if (image.row_stride < width || (image.row_stride & 1U) != 0U)
        {
            throw_invalid("YUV image row stride must be even and at least the width");
        }
        const std::size_t y_storage = checked_multiply(
            image.row_stride, height, "source span");
        const std::size_t chroma_rows = height / 2U;
        if (image.pixel_format == PixelFormat::Nv12 ||
            image.pixel_format == PixelFormat::Nv21)
        {
            return checked_add(
                y_storage,
                checked_add(checked_multiply(chroma_rows - 1U, image.row_stride,
                                              "source span"),
                            width, "source span"),
                "source span");
        }
        const std::size_t chroma_stride = image.row_stride / 2U;
        const std::size_t u_storage = checked_multiply(
            chroma_stride, chroma_rows, "source span");
        const std::size_t v_span = checked_add(
            checked_multiply(chroma_rows - 1U, chroma_stride, "source span"),
            width / 2U, "source span");
        return checked_add(checked_add(y_storage, u_storage, "source span"),
                           v_span, "source span");
    }
    const bool packed_422 = image.pixel_format == PixelFormat::Yuy2 ||
                            image.pixel_format == PixelFormat::Uyvy;
    if (image.pixel_format != PixelFormat::Bgr8 &&
        image.pixel_format != PixelFormat::Rgb8 && !packed_422)
    {
        throw_invalid(
            "image pixel format must be Bgr8, Rgb8, Nv12, I420, Nv21, Yuy2, or Uyvy");
    }
    if (packed_422 && (image.width & 1) != 0)
    {
        throw_invalid("YUY2 and UYVY image width must be even");
    }
    const std::size_t row_bytes = checked_multiply(
        width, packed_422 ? 2U : kChannels, "source row");
    if (image.row_stride < row_bytes)
    {
        throw_invalid("image row stride is smaller than a packed row");
    }
    const std::size_t preceding = checked_multiply(
        static_cast<std::size_t>(image.height - 1), image.row_stride, "source span");
    if (row_bytes > (std::numeric_limits<std::size_t>::max)() - preceding)
    {
        throw_resource("source span byte count overflow");
    }
    return preceding + row_bytes;
}

kfcore::image::ImageView to_image_view(const ImageView& image)
{
    kfcore::image::PixelFormat format = kfcore::image::PixelFormat::Rgb8;
    switch (image.pixel_format)
    {
    case PixelFormat::Bgr8:
        format = kfcore::image::PixelFormat::Bgr8;
        break;
    case PixelFormat::Rgb8:
        format = kfcore::image::PixelFormat::Rgb8;
        break;
    case PixelFormat::Nv12:
        format = kfcore::image::PixelFormat::Nv12;
        break;
    case PixelFormat::I420:
        format = kfcore::image::PixelFormat::I420;
        break;
    case PixelFormat::Nv21:
        format = kfcore::image::PixelFormat::Nv21;
        break;
    case PixelFormat::Yuy2:
        format = kfcore::image::PixelFormat::Yuy2;
        break;
    case PixelFormat::Uyvy:
        format = kfcore::image::PixelFormat::Uyvy;
        break;
    default:
        throw_invalid("image pixel format is unsupported");
    }
    return { image.data,
             source_span(image),
             image.width,
             image.height,
             image.row_stride,
             format,
             kfcore::image::MemoryKind::Host };
}

class UseGuard final
{
public:
    explicit UseGuard(std::atomic_flag& flag)
        : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw_invalid("calls on one detector instance must not overlap");
        }
    }

    ~UseGuard()
    {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag& flag_;
};

} // namespace

struct OnnxDetector::Impl final
{
    Impl(const std::filesystem::path& path, const OnnxDetectorOptions& options_value)
        : options(options_value)
        , environment("KFCoreYoloOnnx")
        , session(environment, path, model_contract(), runtime_options(options))
    {
        validate_contract();
    }

    void validate_contract()
    {
        const std::vector<std::int64_t>& input_shape =
            session.declared_input_dimensions(0U);
        const std::vector<std::int64_t>& output_shape =
            session.declared_output_dimensions(0U);
        if (input_shape.size() != 4U || input_shape[0] != 1 || input_shape[1] != 3 ||
            input_shape[2] <= 0 || input_shape[3] <= 0 ||
            input_shape[2] > (std::numeric_limits<std::int32_t>::max)() ||
            input_shape[3] > (std::numeric_limits<std::int32_t>::max)())
        {
            throw_contract("images shape must be static [1,3,H,W]");
        }
        if (output_shape.size() != 3U || output_shape[0] != 1 ||
            output_shape[1] <= 0 || output_shape[2] != 6)
        {
            throw_contract("output0 shape must be static [1,N,6]");
        }
        if (static_cast<std::uintmax_t>(output_shape[1]) >
            static_cast<std::uintmax_t>(options.max_detections))
        {
            throw_resource("model detections exceed the configured maximum detections");
        }
        input_height_value = static_cast<std::int32_t>(input_shape[2]);
        input_width_value = static_cast<std::int32_t>(input_shape[3]);
        detection_count = static_cast<std::size_t>(output_shape[1]);
        const std::size_t output_elements = checked_multiply(
            detection_count, kDetectionValues, "output tensor");
        const std::size_t output_bytes = checked_multiply(
            output_elements, sizeof(float), "output tensor");
        if (output_bytes > options.max_output_bytes)
        {
            throw_resource("model output exceeds the configured output limit");
        }
    }

    OnnxDetectorOptions           options;
    onnx_runtime::Environment     environment;
    onnx_runtime::Session         session;
    std::int32_t                  input_width_value = 0;
    std::int32_t                  input_height_value = 0;
    std::size_t                   detection_count = 0U;
    std::atomic_flag              in_use = ATOMIC_FLAG_INIT;
};

OnnxDetector::OnnxDetector(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

OnnxDetector::~OnnxDetector() = default;
OnnxDetector::OnnxDetector(OnnxDetector&&) noexcept = default;
OnnxDetector& OnnxDetector::operator=(OnnxDetector&&) noexcept = default;

std::unique_ptr<OnnxDetector> OnnxDetector::load(
    const std::filesystem::path& model_path, const OnnxDetectorOptions& options)
{
    validate_options(options);
    validate_model_file(model_path, options.max_model_bytes);
    try
    {
        return std::unique_ptr<OnnxDetector>(
            new OnnxDetector(std::make_unique<Impl>(model_path, options)));
    }
    catch (const YoloError&)
    {
        throw;
    }
    catch (const onnx_runtime::Error& error)
    {
        rethrow_runtime(error);
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("model session allocation failed");
    }
}

DetectionFrame OnnxDetector::detect(const ImageView& image)
{
    if (!impl_)
    {
        throw_invalid("detector state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    try
    {
        const kfcore::image::ImageView source = to_image_view(image);
        kfcore::image::PreprocessOptions preprocess;
        preprocess.output_format = kfcore::image::PixelFormat::Rgb8;
        preprocess.border_value = impl_->options.border_value;
        preprocess.mirror_horizontal = impl_->options.mirror_horizontal;
        kfcore::image::LetterboxTransform transform;
        std::vector<float> input = kfcore::image::CpuImageProcessor::letterbox_nchw(
            source, impl_->input_width_value, impl_->input_height_value, preprocess,
            impl_->options.max_source_bytes, impl_->options.max_tensor_bytes, &transform);

        std::vector<std::int64_t> input_shape {
            1, 3, impl_->input_height_value, impl_->input_width_value
        };
        std::vector<onnx_runtime::HostTensor> outputs = impl_->session.run(
            { { input.data(), input.size(), std::move(input_shape) } });
        if (outputs.size() != 1U ||
            outputs[0].element_type != onnx_runtime::ElementType::Float32)
        {
            throw_contract("runtime output set is invalid");
        }
        const std::vector<std::int64_t> expected_output_shape {
            1, static_cast<std::int64_t>(impl_->detection_count), 6
        };
        if (outputs[0].dimensions != expected_output_shape)
        {
            throw_contract("runtime output shape does not match [1,N,6]");
        }
        const std::size_t output_elements = checked_multiply(
            impl_->detection_count, kDetectionValues, "runtime output tensor");
        const std::vector<ImageView> images { image };
        const std::vector<detail::LetterboxTransform> transforms { transform };
        const detail::CompactNmsOutputView compact {
            outputs[0].float_values.data(), output_elements,
            impl_->detection_count, TensorDataType::Float32
        };
        return detail::decode_compact_nms(images, transforms, compact).front();
    }
    catch (const YoloError&)
    {
        throw;
    }
    catch (const kfcore::image::ImageProcessorError& error)
    {
        if (error.code() == kfcore::image::ImageProcessorErrorCode::ResourceLimitExceeded)
        {
            throw_resource(error.what());
        }
        throw_invalid(error.what());
    }
    catch (const onnx_runtime::Error& error)
    {
        rethrow_runtime(error);
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("inference allocation failed");
    }
}

std::int32_t OnnxDetector::input_width() const noexcept
{
    return impl_ ? impl_->input_width_value : 0;
}

std::int32_t OnnxDetector::input_height() const noexcept
{
    return impl_ ? impl_->input_height_value : 0;
}

std::size_t OnnxDetector::max_detections() const noexcept
{
    return impl_ ? impl_->detection_count : 0U;
}

} // namespace kfcore::yolo
