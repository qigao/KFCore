#include "kfcore/yolo/detector.hpp"

#include "compact_nms.hpp"
#include "efficient_nms.hpp"
#include "raw_yolo.hpp"
#include "kfcore/image_processor/cpu.hpp"
#include "kfcore/runtime/error.hpp"
#include "kfcore/yolo/error.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kfcore::yolo
{
namespace
{

constexpr std::size_t kChannels = 3U;
constexpr std::size_t kCompactValues = 6U;
constexpr std::size_t kRawBoxValues = 4U;

[[noreturn]] void throw_invalid(const std::string& message)
{
    throw YoloError(YoloErrorCode::InvalidArgument, "YOLO detector: " + message);
}

[[noreturn]] void throw_contract(const std::string& message)
{
    throw YoloError(YoloErrorCode::EngineContractMismatch,
                    "YOLO detector contract: " + message);
}

[[noreturn]] void throw_resource(const std::string& message)
{
    throw YoloError(YoloErrorCode::ResourceLimitExceeded,
                    "YOLO detector resource limit: " + message);
}

[[noreturn]] void throw_runtime(const std::string& message)
{
    throw YoloError(YoloErrorCode::RuntimeFailure, "YOLO detector runtime: " + message);
}

std::size_t checked_multiply(std::size_t left, std::size_t right, const char* object)
{
    if (left != 0U && right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_resource(std::string(object) + " byte count overflow");
    }
    return left * right;
}

void validate_options(const YoloDetectorOptions& options)
{
    if ((options.input_width < 0) || (options.input_height < 0) ||
        (options.input_width == 0) != (options.input_height == 0))
    {
        throw_invalid("input width and height must both be zero or both be positive");
    }
    if (!std::isfinite(options.border_value) || options.border_value < 0.0F ||
        options.border_value > 255.0F)
    {
        throw_invalid("border value must be finite within [0,255]");
    }
    if (!std::isfinite(options.score_threshold) || options.score_threshold < 0.0F ||
        options.score_threshold > 1.0F || !std::isfinite(options.iou_threshold) ||
        options.iou_threshold < 0.0F || options.iou_threshold > 1.0F)
    {
        throw_invalid("score and IoU thresholds must be finite within [0,1]");
    }
    if (options.max_detections == 0U || options.max_source_bytes == 0U ||
        options.max_tensor_bytes == 0U || options.max_output_bytes == 0U)
    {
        throw_resource("configured byte and detection limits must be positive");
    }
}

std::size_t source_span(const ImageView& image)
{
    if (image.data == nullptr || image.width <= 0 || image.height <= 0)
    {
        throw_invalid("image data and dimensions must be valid");
    }
    if (image.memory_kind != MemoryKind::Host)
    {
        throw_invalid("v1 backend-neutral YOLO preprocessing accepts Host images only");
    }

    const std::size_t width = static_cast<std::size_t>(image.width);
    const std::size_t height = static_cast<std::size_t>(image.height);
    if (image.pixel_format == PixelFormat::Nv12 || image.pixel_format == PixelFormat::I420 ||
        image.pixel_format == PixelFormat::Nv21)
    {
        if ((image.width & 1) != 0 || (image.height & 1) != 0 ||
            image.row_stride < width || (image.row_stride & 1U) != 0U)
        {
            throw_invalid("planar/semi-planar YUV dimensions and stride are invalid");
        }
        const std::size_t y_storage = checked_multiply(image.row_stride, height, "source");
        const std::size_t chroma_rows = height / 2U;
        if (image.pixel_format == PixelFormat::Nv12 || image.pixel_format == PixelFormat::Nv21)
        {
            return y_storage + checked_multiply(chroma_rows, image.row_stride, "source");
        }
        const std::size_t chroma_stride = image.row_stride / 2U;
        return y_storage + checked_multiply(chroma_rows, chroma_stride * 2U, "source");
    }

    const bool packed_422 = image.pixel_format == PixelFormat::Yuy2 ||
                            image.pixel_format == PixelFormat::Uyvy;
    if (packed_422 && (image.width & 1) != 0)
    {
        throw_invalid("packed YUV422 width must be even");
    }
    const std::size_t bytes_per_pixel = packed_422 ? 2U : 3U;
    const std::size_t row_bytes = checked_multiply(width, bytes_per_pixel, "source row");
    if (image.row_stride < row_bytes)
    {
        throw_invalid("image row stride is smaller than a packed row");
    }
    return checked_multiply(static_cast<std::size_t>(image.height - 1), image.row_stride,
                            "source span") + row_bytes;
}

kfcore::image::ImageView to_image_view(const ImageView& image)
{
    kfcore::image::PixelFormat format;
    switch (image.pixel_format)
    {
    case PixelFormat::Bgr8: format = kfcore::image::PixelFormat::Bgr8; break;
    case PixelFormat::Rgb8: format = kfcore::image::PixelFormat::Rgb8; break;
    case PixelFormat::Nv12: format = kfcore::image::PixelFormat::Nv12; break;
    case PixelFormat::I420: format = kfcore::image::PixelFormat::I420; break;
    case PixelFormat::Nv21: format = kfcore::image::PixelFormat::Nv21; break;
    case PixelFormat::Yuy2: format = kfcore::image::PixelFormat::Yuy2; break;
    case PixelFormat::Uyvy: format = kfcore::image::PixelFormat::Uyvy; break;
    default: throw_invalid("unsupported image pixel format");
    }
    return {image.data, source_span(image), image.width, image.height, image.row_stride,
            format, kfcore::image::MemoryKind::Host};
}

std::size_t element_size(runtime::DataType type)
{
    using runtime::DataType;
    switch (type)
    {
    case DataType::Float32: return sizeof(float);
    case DataType::Float16: return sizeof(std::uint16_t);
    case DataType::BFloat16: return sizeof(std::uint16_t);
    case DataType::Int8: return sizeof(std::int8_t);
    case DataType::UInt8: return sizeof(std::uint8_t);
    case DataType::Int32: return sizeof(std::int32_t);
    case DataType::Int64: return sizeof(std::int64_t);
    case DataType::Bool: return sizeof(std::uint8_t);
    }
    throw_contract("unsupported tensor data type");
}

std::size_t element_count(const runtime::TensorShape& shape, const char* name)
{
    std::size_t count = 1U;
    for (const auto dimension : shape)
    {
        if (dimension <= 0 ||
            static_cast<std::uintmax_t>(dimension) >
                static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)()))
        {
            throw_contract(std::string(name) + " contains unresolved tensor dimensions");
        }
        count = checked_multiply(count, static_cast<std::size_t>(dimension), name);
    }
    return count;
}

runtime::TensorShape resolve_input_shape(const runtime::TensorDescriptor& input,
                                         const YoloDetectorOptions& options,
                                         std::int32_t& out_width,
                                         std::int32_t& out_height)
{
    if (input.shape.size() != 4U)
    {
        throw_contract("input tensor rank must be 4 (NCHW)");
    }
    runtime::TensorShape shape = input.shape;
    if (shape[0] == -1)
    {
        shape[0] = 1;
    }
    if (shape[0] != 1)
    {
        throw_contract("v1 typed YOLO requires batch size 1");
    }
    if (shape[1] == -1)
    {
        shape[1] = 3;
    }
    if (shape[1] != 3)
    {
        throw_contract("input tensor channel count must be 3");
    }

    auto resolve_spatial = [](std::int64_t declared, std::int32_t requested,
                              const char* name) -> std::int32_t {
        if (declared > 0)
        {
            if (declared > (std::numeric_limits<std::int32_t>::max)())
            {
                throw_contract(std::string(name) + " exceeds int32 range");
            }
            const auto value = static_cast<std::int32_t>(declared);
            if (requested > 0 && requested != value)
            {
                throw_contract(std::string(name) + " conflicts with static artifact shape");
            }
            return value;
        }
        if (declared == -1 && requested > 0)
        {
            return requested;
        }
        throw_contract(std::string(name) +
                       " is dynamic; YoloDetectorOptions must select an explicit size");
    };

    out_height = resolve_spatial(shape[2], options.input_height, "input height");
    out_width = resolve_spatial(shape[3], options.input_width, "input width");
    shape[2] = out_height;
    shape[3] = out_width;
    return shape;
}

runtime::TensorShape resolve_output_shape(const runtime::TensorDescriptor& descriptor,
                                          std::size_t max_detections,
                                          bool detection_axis_dynamic)
{
    runtime::TensorShape shape = descriptor.shape;
    for (std::size_t index = 0U; index < shape.size(); ++index)
    {
        if (shape[index] > 0)
        {
            continue;
        }
        if (shape[index] != -1)
        {
            throw_contract("output tensor contains invalid dimension");
        }
        if (index == 0U)
        {
            shape[index] = 1;
            continue;
        }
        if (detection_axis_dynamic && index == 1U)
        {
            shape[index] = static_cast<std::int64_t>(max_detections);
            continue;
        }
        throw_contract("output tensor has unsupported dynamic non-batch dimension");
    }
    return shape;
}

TensorDataType decoder_type(runtime::DataType type)
{
    switch (type)
    {
    case runtime::DataType::Float32: return TensorDataType::Float32;
    case runtime::DataType::Float16: return TensorDataType::Float16;
    case runtime::DataType::Int32: return TensorDataType::Int32;
    default: throw_contract("YOLO decoder supports FP32/FP16 floating outputs and Int32 labels only");
    }
}

std::uint16_t float_to_half(float value) noexcept
{
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23U) & 0xffU;
    const std::uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU)
    {
        return static_cast<std::uint16_t>(sign | (mantissa == 0U ? 0x7c00U : 0x7e00U));
    }
    const int adjusted = static_cast<int>(exponent) - 127 + 15;
    if (adjusted >= 31)
    {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    if (adjusted <= 0)
    {
        if (adjusted < -10)
        {
            return static_cast<std::uint16_t>(sign);
        }
        std::uint32_t normalized = mantissa | 0x800000U;
        const int shift = 14 - adjusted;
        const std::uint32_t rounded = (normalized + (1U << (shift - 1))) >> shift;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const std::uint32_t rounded = mantissa + 0x1000U;
    if ((rounded & 0x800000U) != 0U)
    {
        return static_cast<std::uint16_t>(sign | ((adjusted + 1) << 10U));
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(adjusted) << 10U) |
                                      (rounded >> 13U));
}

class UseGuard final
{
public:
    explicit UseGuard(std::atomic_flag& flag) : flag_(flag)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
        {
            throw_invalid("calls on one detector instance must not overlap");
        }
    }
    ~UseGuard() { flag_.clear(std::memory_order_release); }

private:
    std::atomic_flag& flag_;
};

struct HostOutput
{
    runtime::TensorDescriptor descriptor;
    runtime::TensorShape shape;
    std::vector<std::max_align_t> storage;
    std::size_t bytes = 0U;

    void allocate(std::size_t max_output_bytes)
    {
        const std::size_t count = element_count(shape, descriptor.name.c_str());
        bytes = checked_multiply(count, element_size(descriptor.data_type), descriptor.name.c_str());
        if (bytes > max_output_bytes)
        {
            throw_resource("one output tensor exceeds max_output_bytes");
        }
        const std::size_t units = (bytes + sizeof(std::max_align_t) - 1U) / sizeof(std::max_align_t);
        storage.resize(units);
    }

    void* data() noexcept { return storage.empty() ? nullptr : storage.data(); }
    const void* data() const noexcept { return storage.empty() ? nullptr : storage.data(); }
    std::size_t count() const { return element_count(shape, descriptor.name.c_str()); }

    runtime::MutableTensorView view()
    {
        return {descriptor.name, descriptor.data_type, shape, data(), bytes,
                runtime::MemoryKind::Host, {}};
    }

    runtime::DynamicMutableTensorView dynamic_view()
    {
        return {descriptor.name, descriptor.data_type, data(), bytes,
                runtime::MemoryKind::Host, {}, {}, 0U};
    }
};

std::string lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

} // namespace

struct YoloDetector::Impl final
{
    enum class OutputLayout
    {
        CompactNms,
        RawYolo,
        RawYoloX,
        EfficientNms
    };

    Impl(runtime::ResolvedModel resolved_value, YoloDetectorOptions options_value)
        : resolved(std::move(resolved_value))
        , options(options_value)
        , context(resolved.model->create_context())
    {
        validate_contract();
    }

    void validate_contract()
    {
        const auto tensors = resolved.model->tensors();
        std::vector<runtime::TensorDescriptor> inputs;
        for (const auto& tensor : tensors)
        {
            if (tensor.is_input)
            {
                inputs.push_back(tensor);
            }
            else
            {
                output_descriptors.push_back(tensor);
            }
        }
        if (inputs.size() != 1U || output_descriptors.empty())
        {
            throw_contract("YOLO requires exactly one input and at least one output tensor");
        }
        input_descriptor = inputs.front();
        if (input_descriptor.data_type != runtime::DataType::Float32 &&
            input_descriptor.data_type != runtime::DataType::Float16)
        {
            throw_contract("YOLO input tensor must be FP32 or FP16");
        }
        input_shape = resolve_input_shape(input_descriptor, options, input_width_value,
                                          input_height_value);

        const std::string flavor = lower(resolved.route.artifact.flavor);
        if (flavor.empty())
        {
            throw_contract("YOLO artifact must declare flavor explicitly");
        }

        if (flavor == "compact-nms")
        {
            layout = OutputLayout::CompactNms;
            if (output_descriptors.size() != 1U || output_descriptors[0].shape.size() != 3U)
            {
                throw_contract("compact-nms requires one rank-3 output tensor");
            }
            HostOutput output;
            output.descriptor = output_descriptors[0];
            if ((output.descriptor.shape[0] != -1 && output.descriptor.shape[0] != 1) ||
                output.descriptor.shape[2] != static_cast<std::int64_t>(kCompactValues))
            {
                throw_contract("compact-nms declared output shape must be [1,N,6]");
            }
            compact_dynamic_output = output.descriptor.shape[1] == -1;
            output.shape = resolve_output_shape(output.descriptor, options.max_detections, true);
            if (output.shape[0] != 1 || output.shape[2] != static_cast<std::int64_t>(kCompactValues))
            {
                throw_contract("compact-nms output shape must be [1,N,6]");
            }
            if (output.descriptor.data_type != runtime::DataType::Float32 &&
                output.descriptor.data_type != runtime::DataType::Float16)
            {
                throw_contract("compact-nms output must be FP32 or FP16");
            }
            detection_count = compact_dynamic_output ? 0U : static_cast<std::size_t>(output.shape[1]);
            output.allocate(options.max_output_bytes);
            outputs.push_back(std::move(output));
        }
        else if (flavor == "raw-yolo" ||
                 flavor == "raw-yolox")
        {
            layout =
                flavor == "raw-yolox"
                    ? OutputLayout::RawYoloX
                    : OutputLayout::RawYolo;
            if (output_descriptors.size() != 1U ||
                output_descriptors[0].shape.size() != 3U)
            {
                throw_contract(
                    "raw YOLO flavors require one rank-3 output tensor");
            }
            HostOutput output;
            output.descriptor = output_descriptors[0];
            output.shape =
                resolve_output_shape(
                    output.descriptor,
                    options.max_detections,
                    false);
            if (output.descriptor.data_type != runtime::DataType::Float32 &&
                output.descriptor.data_type != runtime::DataType::Float16)
            {
                throw_contract("raw YOLO output must be FP32 or FP16");
            }

            if (layout == OutputLayout::RawYolo)
            {
                if (output.shape[0] != 1 ||
                    output.shape[1] <=
                        static_cast<std::int64_t>(kRawBoxValues) ||
                    output.shape[2] <= 0)
                {
                    throw_contract(
                        "raw-yolo output shape must be [1,4+C,A]");
                }
                class_count =
                    static_cast<std::size_t>(
                        output.shape[1]) -
                    kRawBoxValues;
                candidate_count =
                    static_cast<std::size_t>(
                        output.shape[2]);
            }
            else
            {
                constexpr std::int64_t kYoloXBaseValues = 5;
                if (output.shape[0] != 1 ||
                    output.shape[1] <= 0 ||
                    output.shape[2] <= kYoloXBaseValues)
                {
                    throw_contract(
                        "raw-yolox output shape must be [1,A,5+C]");
                }
                candidate_count =
                    static_cast<std::size_t>(
                        output.shape[1]);
                class_count =
                    static_cast<std::size_t>(
                        output.shape[2] -
                        kYoloXBaseValues);
            }
            output.allocate(options.max_output_bytes);
            outputs.push_back(std::move(output));
        }
        else if (flavor == "efficient-nms")
        {
            layout = OutputLayout::EfficientNms;
            if (output_descriptors.size() != 4U)
            {
                throw_contract("efficient-nms requires four output tensors");
            }
            configure_efficient_outputs();
        }
        else
        {
            throw_contract("unsupported artifact flavor: " + resolved.route.artifact.flavor);
        }
    }

    void configure_efficient_outputs()
    {
        outputs.clear();
        outputs.reserve(output_descriptors.size());
        for (const auto& descriptor : output_descriptors)
        {
            HostOutput output;
            output.descriptor = descriptor;
            const std::string name = lower(descriptor.name);
            const bool count_tensor = name.find("num") != std::string::npos &&
                                      name.find("det") != std::string::npos;
            output.shape = resolve_output_shape(descriptor, options.max_detections, !count_tensor);
            output.allocate(options.max_output_bytes);
            outputs.push_back(std::move(output));
        }

        for (std::size_t i = 0U; i < outputs.size(); ++i)
        {
            const auto& output = outputs[i];
            const std::string name = lower(output.descriptor.name);
            if (name.find("box") != std::string::npos)
            {
                boxes_index = i;
            }
            else if (name.find("score") != std::string::npos)
            {
                scores_index = i;
            }
            else if (name.find("label") != std::string::npos ||
                     name.find("class") != std::string::npos)
            {
                labels_index = i;
            }
            else if (name.find("num") != std::string::npos &&
                     name.find("det") != std::string::npos)
            {
                count_index = i;
            }
        }

        const std::size_t missing = (std::numeric_limits<std::size_t>::max)();
        if (count_index == missing || boxes_index == missing || scores_index == missing ||
            labels_index == missing)
        {
            for (std::size_t i = 0U; i < outputs.size(); ++i)
            {
                const auto& output = outputs[i];
                if (output.descriptor.data_type == runtime::DataType::Int32 && output.count() == 1U)
                {
                    count_index = i;
                }
                else if (output.descriptor.data_type == runtime::DataType::Int32)
                {
                    labels_index = i;
                }
                else if (output.shape.size() == 3U && output.shape.back() == 4)
                {
                    boxes_index = i;
                }
                else if (output.descriptor.data_type == runtime::DataType::Float32 ||
                         output.descriptor.data_type == runtime::DataType::Float16)
                {
                    scores_index = i;
                }
            }
        }
        if (count_index == missing || boxes_index == missing || scores_index == missing ||
            labels_index == missing)
        {
            throw_contract("cannot map efficient-nms tensor roles");
        }

        const auto& boxes = outputs[boxes_index];
        if (boxes.shape.size() != 3U || boxes.shape[0] != 1 || boxes.shape[2] != 4 ||
            boxes.shape[1] <= 0)
        {
            throw_contract("efficient-nms boxes shape must be [1,N,4]");
        }
        detection_count = static_cast<std::size_t>(boxes.shape[1]);
        if (outputs[count_index].descriptor.data_type != runtime::DataType::Int32 ||
            outputs[labels_index].descriptor.data_type != runtime::DataType::Int32 ||
            outputs[scores_index].descriptor.data_type != boxes.descriptor.data_type ||
            (boxes.descriptor.data_type != runtime::DataType::Float32 &&
             boxes.descriptor.data_type != runtime::DataType::Float16))
        {
            throw_contract("efficient-nms tensor types are invalid");
        }
        if (outputs[count_index].count() != 1U ||
            outputs[scores_index].count() != detection_count ||
            outputs[labels_index].count() != detection_count ||
            boxes.count() != checked_multiply(detection_count, 4U, "boxes"))
        {
            throw_contract("efficient-nms tensor element counts are inconsistent");
        }
    }

    runtime::ResolvedModel resolved;
    YoloDetectorOptions options;
    std::unique_ptr<runtime::ExecutionContext> context;
    runtime::TensorDescriptor input_descriptor;
    runtime::TensorShape input_shape;
    std::vector<runtime::TensorDescriptor> output_descriptors;
    std::vector<HostOutput> outputs;
    std::int32_t input_width_value = 0;
    std::int32_t input_height_value = 0;
    OutputLayout layout = OutputLayout::RawYolo;
    bool compact_dynamic_output = false;
    std::size_t detection_count = 0U;
    std::size_t class_count = 0U;
    std::size_t candidate_count = 0U;
    std::size_t count_index = (std::numeric_limits<std::size_t>::max)();
    std::size_t boxes_index = (std::numeric_limits<std::size_t>::max)();
    std::size_t scores_index = (std::numeric_limits<std::size_t>::max)();
    std::size_t labels_index = (std::numeric_limits<std::size_t>::max)();
    std::atomic_flag in_use = ATOMIC_FLAG_INIT;
};

YoloDetector::YoloDetector(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
YoloDetector::~YoloDetector() = default;
YoloDetector::YoloDetector(YoloDetector&&) noexcept = default;
YoloDetector& YoloDetector::operator=(YoloDetector&&) noexcept = default;

std::unique_ptr<YoloDetector> YoloDetector::load(
    const runtime::ModelPackage& package,
    const runtime::BackendRegistry& backends,
    const runtime::ExecutionPolicy& policy,
    const YoloDetectorOptions& options)
{
    validate_options(options);
    if (package.model_type() != kYoloDetectionModelType)
    {
        throw_contract("ModelPackage model_type must be 'yolo-detection'");
    }
    try
    {
        auto resolved = runtime::ModelResolver::load(package, backends, policy);
        return std::unique_ptr<YoloDetector>(
            new YoloDetector(std::make_unique<Impl>(std::move(resolved), options)));
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
}

DetectionFrame YoloDetector::detect(const ImageView& image)
{
    if (!impl_ || !impl_->context)
    {
        throw_invalid("detector state is unavailable");
    }
    UseGuard guard(impl_->in_use);
    try
    {
        const kfcore::image::ImageView source = to_image_view(image);
        kfcore::image::PreprocessOptions preprocess;
        preprocess.output_format =
            impl_->layout == Impl::OutputLayout::RawYoloX
                ? kfcore::image::PixelFormat::Bgr8
                : kfcore::image::PixelFormat::Rgb8;
        preprocess.border_value = impl_->options.border_value;
        preprocess.mirror_horizontal =
            impl_->options.mirror_horizontal;
        if (impl_->layout == Impl::OutputLayout::RawYoloX)
        {
            preprocess.center_letterbox = false;
            preprocess.stddev = {
                1.0F / 255.0F,
                1.0F / 255.0F,
                1.0F / 255.0F,
            };
        }
        kfcore::image::LetterboxTransform transform;
        std::vector<float> input = kfcore::image::CpuImageProcessor::letterbox_nchw(
            source, impl_->input_width_value, impl_->input_height_value, preprocess,
            impl_->options.max_source_bytes, impl_->options.max_tensor_bytes, &transform);

        std::vector<std::uint16_t> half_input;
        const void* input_data = input.data();
        std::size_t input_bytes = checked_multiply(input.size(), sizeof(float), "input tensor");
        if (impl_->input_descriptor.data_type == runtime::DataType::Float16)
        {
            half_input.resize(input.size());
            std::transform(input.begin(), input.end(), half_input.begin(), float_to_half);
            input_data = half_input.data();
            input_bytes = checked_multiply(half_input.size(), sizeof(std::uint16_t), "input tensor");
        }
        if (input_bytes > impl_->options.max_tensor_bytes)
        {
            throw_resource("input tensor exceeds max_tensor_bytes");
        }

        std::vector<runtime::TensorView> inputs{
            {impl_->input_descriptor.name, impl_->input_descriptor.data_type,
             impl_->input_shape, input_data, input_bytes, runtime::MemoryKind::Host, {}}
        };

        std::size_t compact_detection_count = impl_->detection_count;
        if (impl_->layout == Impl::OutputLayout::CompactNms && impl_->compact_dynamic_output)
        {
            auto& output = impl_->outputs.front();
            std::vector<runtime::DynamicMutableTensorView> dynamic_outputs{output.dynamic_view()};
            impl_->context->run_dynamic(inputs, dynamic_outputs);
            const auto& actual = dynamic_outputs.front();
            if (actual.shape.size() != 3U || actual.shape[0] != 1 || actual.shape[1] < 0 ||
                actual.shape[2] != static_cast<std::int64_t>(kCompactValues))
            {
                throw_contract("compact-nms dynamic output must resolve to [1,N,6]");
            }
            if (static_cast<std::uintmax_t>(actual.shape[1]) >
                static_cast<std::uintmax_t>(impl_->options.max_detections))
            {
                throw_resource("compact-nms dynamic output exceeds max_detections");
            }
            compact_detection_count = static_cast<std::size_t>(actual.shape[1]);
            const std::size_t actual_elements = checked_multiply(
                compact_detection_count, kCompactValues, "compact-nms dynamic output");
            const std::size_t expected_bytes = checked_multiply(
                actual_elements, element_size(output.descriptor.data_type),
                "compact-nms dynamic output");
            if (actual.byte_size != expected_bytes)
            {
                throw_contract("compact-nms dynamic output byte count does not match actual shape");
            }
        }
        else
        {
            std::vector<runtime::MutableTensorView> output_views;
            output_views.reserve(impl_->outputs.size());
            std::size_t aggregate_output_bytes = 0U;
            for (auto& output : impl_->outputs)
            {
                if (output.bytes > impl_->options.max_output_bytes - aggregate_output_bytes)
                {
                    throw_resource("aggregate outputs exceed max_output_bytes");
                }
                aggregate_output_bytes += output.bytes;
                output_views.push_back(output.view());
            }
            impl_->context->run(inputs, output_views);
        }

        const std::vector<ImageView> images{image};
        const std::vector<detail::LetterboxTransform> transforms{transform};
        if (impl_->layout == Impl::OutputLayout::CompactNms)
        {
            const auto& output = impl_->outputs.front();
            const std::size_t compact_elements = checked_multiply(
                compact_detection_count, kCompactValues, "compact-nms output");
            const detail::CompactNmsOutputView view{
                output.data(), compact_elements, compact_detection_count,
                decoder_type(output.descriptor.data_type)};
            return detail::decode_compact_nms(images, transforms, view).front();
        }
        if (impl_->layout == Impl::OutputLayout::RawYolo ||
            impl_->layout == Impl::OutputLayout::RawYoloX)
        {
            const auto& output = impl_->outputs.front();
            const detail::RawYoloOutputView view{
                output.data(),
                output.count(),
                impl_->class_count,
                impl_->candidate_count,
                decoder_type(output.descriptor.data_type),
                impl_->options.score_threshold,
                impl_->options.iou_threshold,
                impl_->options.max_detections,
                impl_->layout == Impl::OutputLayout::RawYoloX
                    ? detail::RawYoloOutputLayout::
                          AnchorsFirstObjectnessClassScores
                    : detail::RawYoloOutputLayout::
                          ChannelsFirstClassScores};
            return detail::decode_raw_yolo(
                images, transforms, view).front();
        }

        const auto& counts = impl_->outputs[impl_->count_index];
        const auto& boxes = impl_->outputs[impl_->boxes_index];
        const auto& scores = impl_->outputs[impl_->scores_index];
        const auto& labels = impl_->outputs[impl_->labels_index];
        const detail::EfficientNmsOutputView view{
            static_cast<const std::int32_t*>(counts.data()), counts.count(),
            boxes.data(), boxes.count(), scores.data(), scores.count(),
            static_cast<const std::int32_t*>(labels.data()), labels.count(),
            impl_->detection_count, decoder_type(boxes.descriptor.data_type)};
        return detail::decode_efficient_nms(images, transforms, view).front();
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
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("inference allocation failed");
    }
}

std::int32_t YoloDetector::input_width() const noexcept
{
    return impl_ ? impl_->input_width_value : 0;
}

std::int32_t YoloDetector::input_height() const noexcept
{
    return impl_ ? impl_->input_height_value : 0;
}

const runtime::ExecutionRoute& YoloDetector::execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty{};
    return impl_ ? impl_->resolved.route : empty;
}

} // namespace kfcore::yolo
