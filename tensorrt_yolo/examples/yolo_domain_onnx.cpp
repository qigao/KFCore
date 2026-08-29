#include "yolo_domain_onnx.hpp"

#include "detector_helpers.hpp"
#include "kfcore/image_processor/cpu.hpp"
#include "kfcore/yolo/error.hpp"

#include <onnxruntime_cxx_api.h>

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

namespace kfcore::yolo::demo
{
namespace
{

constexpr std::size_t kChannels = 3U;
constexpr std::size_t kDetectionValues = 6U;

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
    if (image.pixel_format != PixelFormat::Bgr8 &&
        image.pixel_format != PixelFormat::Rgb8)
    {
        throw_invalid("image pixel format must be Bgr8 or Rgb8");
    }
    const std::size_t row_bytes = checked_multiply(
        static_cast<std::size_t>(image.width), kChannels, "source row");
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
    return { image.data,
             source_span(image),
             image.width,
             image.height,
             image.row_stride,
             image.pixel_format == PixelFormat::Bgr8
                 ? kfcore::image::PixelFormat::Bgr8
                 : kfcore::image::PixelFormat::Rgb8,
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

struct OnnxDomainDetector::Impl final
{
    Impl(const std::filesystem::path& path, const OnnxDetectorOptions& options_value)
        : options(options_value)
        , environment(ORT_LOGGING_LEVEL_ERROR, "KFCoreYoloDomainCpu")
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        if (options.intra_op_threads > 0)
        {
            session_options.SetIntraOpNumThreads(options.intra_op_threads);
        }
        if (options.inter_op_threads > 0)
        {
            session_options.SetInterOpNumThreads(options.inter_op_threads);
        }
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
        session = std::make_unique<Ort::Session>(environment, path.c_str(), session_options);
#else
        session = std::make_unique<Ort::Session>(environment, path.string().c_str(),
                                                 session_options);
#endif
        validate_contract();
    }

    void validate_contract()
    {
        if (session->GetInputCount() != 1U || session->GetOutputCount() != 1U)
        {
            throw_contract("expected exactly one input and one output tensor");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        const auto actual_input_name = session->GetInputNameAllocated(0U, allocator);
        const auto actual_output_name = session->GetOutputNameAllocated(0U, allocator);
        if (actual_input_name.get() == nullptr || input_name != actual_input_name.get())
        {
            throw_contract("input tensor must be named images");
        }
        if (actual_output_name.get() == nullptr || output_name != actual_output_name.get())
        {
            throw_contract("output tensor must be named output0");
        }

        const Ort::TypeInfo input_type_info = session->GetInputTypeInfo(0U);
        const Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(0U);
        const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
        const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
        if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        {
            throw_contract("input and output tensors must be FP32");
        }
        const std::vector<std::int64_t> input_shape = input_info.GetShape();
        const std::vector<std::int64_t> output_shape = output_info.GetShape();
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
    Ort::Env                      environment;
    Ort::SessionOptions           session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo               memory_info;
    std::int32_t                  input_width_value = 0;
    std::int32_t                  input_height_value = 0;
    std::size_t                   detection_count = 0U;
    std::string                   input_name = "images";
    std::string                   output_name = "output0";
    std::atomic_flag              in_use = ATOMIC_FLAG_INIT;
};

OnnxDomainDetector::OnnxDomainDetector(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

OnnxDomainDetector::~OnnxDomainDetector() = default;
OnnxDomainDetector::OnnxDomainDetector(OnnxDomainDetector&&) noexcept = default;
OnnxDomainDetector& OnnxDomainDetector::operator=(OnnxDomainDetector&&) noexcept = default;

std::unique_ptr<OnnxDomainDetector> OnnxDomainDetector::load(
    const std::filesystem::path& model_path, const OnnxDetectorOptions& options)
{
    validate_options(options);
    validate_model_file(model_path, options.max_model_bytes);
    try
    {
        return std::unique_ptr<OnnxDomainDetector>(
            new OnnxDomainDetector(std::make_unique<Impl>(model_path, options)));
    }
    catch (const YoloError&)
    {
        throw;
    }
    catch (const Ort::Exception& error)
    {
        throw std::runtime_error(std::string("ONNX Runtime model load failed: ") +
                                 error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("model session allocation failed");
    }
}

DetectionFrame OnnxDomainDetector::detect(const ImageView& image)
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
        kfcore::image::LetterboxTransform transform;
        std::vector<float> input = kfcore::image::CpuImageProcessor::letterbox_nchw(
            source, impl_->input_width_value, impl_->input_height_value, preprocess,
            impl_->options.max_source_bytes, impl_->options.max_tensor_bytes, &transform);

        const std::array<std::int64_t, 4> input_shape {
            1, 3, impl_->input_height_value, impl_->input_width_value
        };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            impl_->memory_info, input.data(), input.size(), input_shape.data(),
            input_shape.size());
        const char* input_names[] = { impl_->input_name.c_str() };
        const char* output_names[] = { impl_->output_name.c_str() };
        Ort::RunOptions run_options;
        std::vector<Ort::Value> outputs = impl_->session->Run(
            run_options, input_names, &input_tensor, 1U, output_names, 1U);
        if (outputs.size() != 1U || !outputs[0].IsTensor())
        {
            throw_contract("runtime output set is invalid");
        }
        const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
        if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        {
            throw_contract("runtime output type is not FP32");
        }
        const std::vector<std::int64_t> expected_output_shape {
            1, static_cast<std::int64_t>(impl_->detection_count), 6
        };
        if (output_info.GetShape() != expected_output_shape)
        {
            throw_contract("runtime output shape does not match [1,N,6]");
        }
        const std::size_t output_elements = checked_multiply(
            impl_->detection_count, kDetectionValues, "runtime output tensor");
        const std::vector<ImageView> images { image };
        const std::vector<detail::LetterboxTransform> transforms { transform };
        const detail::CompactNmsOutputView compact {
            outputs[0].GetTensorData<float>(), output_elements,
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
    catch (const Ort::Exception& error)
    {
        throw std::runtime_error(std::string("ONNX Runtime inference failed: ") +
                                 error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("inference allocation failed");
    }
}

std::int32_t OnnxDomainDetector::input_width() const noexcept
{
    return impl_ ? impl_->input_width_value : 0;
}

std::int32_t OnnxDomainDetector::input_height() const noexcept
{
    return impl_ ? impl_->input_height_value : 0;
}

std::size_t OnnxDomainDetector::max_detections() const noexcept
{
    return impl_ ? impl_->detection_count : 0U;
}

} // namespace kfcore::yolo::demo
