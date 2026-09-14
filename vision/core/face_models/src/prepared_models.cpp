#include "kfcore/face_models/runtime.hpp"

#include "decode.hpp"
#include "kfcore/runtime/error.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
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

using Shape = runtime::TensorShape;

[[noreturn]] void throw_invalid(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::InvalidArgument,
                         "prepared face model: " + detail);
}

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::ModelContractMismatch,
                         "prepared face model contract: " + detail);
}

[[noreturn]] void throw_resource(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::ResourceLimitExceeded,
                         "prepared face model resource limit: " + detail);
}

[[noreturn]] void throw_runtime(const std::string& detail)
{
    throw FaceModelError(FaceModelErrorCode::RuntimeFailure,
                         "prepared face model execution: " + detail);
}

void validate_options(const PreparedFaceModelOptions& options)
{
    if (options.max_tensor_bytes == 0U || options.max_output_bytes == 0U)
    {
        throw_invalid("byte limits must be positive");
    }
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char* subject)
{
    if (left != 0U && right > (std::numeric_limits<std::size_t>::max)() / left)
    {
        throw_resource(std::string(subject) + " size overflow");
    }
    return left * right;
}

std::size_t element_count(const Shape& shape, const char* subject)
{
    std::size_t result = 1U;
    for (const auto dimension : shape)
    {
        if (dimension <= 0 ||
            static_cast<std::uintmax_t>(dimension) >
                static_cast<std::uintmax_t>((std::numeric_limits<std::size_t>::max)()))
        {
            throw_contract(std::string(subject) + " contains invalid dimensions");
        }
        result = checked_multiply(result, static_cast<std::size_t>(dimension), subject);
    }
    return result;
}

std::size_t element_size(runtime::DataType type)
{
    switch (type)
    {
    case runtime::DataType::Float32: return sizeof(float);
    case runtime::DataType::Float16: return sizeof(std::uint16_t);
    default: throw_contract("prepared face tensors must use FP32 or FP16");
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
        return static_cast<std::uint16_t>(sign | (mantissa == 0U ? 0x7c00U : 0x7e00U));
    const int adjusted = static_cast<int>(exponent) - 127 + 15;
    if (adjusted >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
    if (adjusted <= 0)
    {
        if (adjusted < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t normalized = mantissa | 0x800000U;
        const int shift = 14 - adjusted;
        return static_cast<std::uint16_t>(
            sign | ((normalized + (UINT32_C(1) << (shift - 1))) >> shift));
    }
    const std::uint32_t rounded = mantissa + 0x1000U;
    if ((rounded & 0x800000U) != 0U)
        return static_cast<std::uint16_t>(
            sign | (static_cast<std::uint32_t>(adjusted + 1) << 10U));
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(adjusted) << 10U) | (rounded >> 13U));
}

float half_to_float(std::uint16_t bits) noexcept
{
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1fU;
    const std::uint32_t mantissa = bits & 0x03ffU;
    std::uint32_t out = 0U;
    if (exponent == 0U)
    {
        if (mantissa == 0U)
        {
            out = sign;
        }
        else
        {
            std::uint32_t normalized = mantissa;
            int shift = 0;
            while ((normalized & 0x0400U) == 0U)
            {
                normalized <<= 1U;
                ++shift;
            }
            normalized &= 0x03ffU;
            const std::uint32_t exp = static_cast<std::uint32_t>(127 - 15 - shift);
            out = sign | (exp << 23U) | (normalized << 13U);
        }
    }
    else if (exponent == 0x1fU)
    {
        out = sign | 0x7f800000U | (mantissa << 13U);
    }
    else
    {
        out = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    float result = 0.0F;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}

bool compatible_shape(const Shape& declared, const Shape& expected)
{
    if (declared.size() != expected.size()) return false;
    for (std::size_t i = 0U; i < declared.size(); ++i)
    {
        if (declared[i] != -1 && declared[i] != expected[i]) return false;
    }
    return true;
}

std::vector<runtime::TensorDescriptor> match_tensors(
    const std::vector<runtime::TensorDescriptor>& tensors,
    bool inputs,
    const std::vector<Shape>& expected,
    const char* subject)
{
    std::vector<runtime::TensorDescriptor> available;
    for (const auto& tensor : tensors)
    {
        if (tensor.is_input == inputs) available.push_back(tensor);
    }
    if (available.size() != expected.size())
        throw_contract(std::string(subject) + " tensor count does not match model contract");

    std::vector<runtime::TensorDescriptor> result;
    result.reserve(expected.size());
    std::vector<bool> used(available.size(), false);
    for (const auto& shape : expected)
    {
        std::size_t match = available.size();
        for (std::size_t i = 0U; i < available.size(); ++i)
        {
            if (!used[i] && compatible_shape(available[i].shape, shape))
            {
                if (match != available.size())
                    throw_contract(std::string(subject) + " has ambiguous tensor shapes");
                match = i;
            }
        }
        if (match == available.size())
            throw_contract(std::string(subject) + " tensor shape is missing");
        if (available[match].data_type != runtime::DataType::Float32 &&
            available[match].data_type != runtime::DataType::Float16)
            throw_contract(std::string(subject) + " tensor must use FP32 or FP16");
        used[match] = true;
        result.push_back(available[match]);
    }
    return result;
}

struct InputStorage
{
    std::vector<std::uint16_t> half;
    const void* data = nullptr;
    std::size_t bytes = 0U;
};

InputStorage prepare_input(const PreparedTensorView& view,
                           const Shape& shape,
                           runtime::DataType type,
                           std::size_t max_bytes)
{
    const std::size_t expected = element_count(shape, "prepared input");
    if (view.data == nullptr || view.element_count != expected)
        throw_invalid("prepared input pointer/count does not match model contract");
    InputStorage result;
    if (type == runtime::DataType::Float32)
    {
        result.data = view.data;
        result.bytes = checked_multiply(view.element_count, sizeof(float), "prepared input");
    }
    else if (type == runtime::DataType::Float16)
    {
        result.half.resize(view.element_count);
        std::transform(view.data, view.data + view.element_count,
                       result.half.begin(), float_to_half);
        result.data = result.half.data();
        result.bytes = checked_multiply(result.half.size(), sizeof(std::uint16_t),
                                        "prepared input");
    }
    else
    {
        throw_contract("prepared input must use FP32 or FP16");
    }
    if (result.bytes > max_bytes) throw_resource("prepared input exceeds max_tensor_bytes");
    return result;
}

struct OutputBuffer
{
    runtime::TensorDescriptor descriptor;
    Shape shape;
    std::vector<std::max_align_t> storage;
    std::size_t bytes = 0U;

    void allocate(std::size_t max_bytes)
    {
        const std::size_t count = element_count(shape, descriptor.name.c_str());
        bytes = checked_multiply(count, element_size(descriptor.data_type),
                                 descriptor.name.c_str());
        if (bytes > max_bytes) throw_resource("prepared output exceeds max_output_bytes");
        storage.resize((bytes + sizeof(std::max_align_t) - 1U) / sizeof(std::max_align_t));
    }

    runtime::MutableTensorView view()
    {
        return {descriptor.name, descriptor.data_type, shape,
                storage.empty() ? nullptr : storage.data(), bytes,
                runtime::MemoryKind::Host, {}};
    }

    std::vector<float> floats() const
    {
        const std::size_t count = element_count(shape, descriptor.name.c_str());
        std::vector<float> result(count);
        if (descriptor.data_type == runtime::DataType::Float32)
        {
            std::memcpy(result.data(), storage.data(), count * sizeof(float));
        }
        else
        {
            const auto* values = reinterpret_cast<const std::uint16_t*>(storage.data());
            std::transform(values, values + count, result.begin(), half_to_float);
        }
        return result;
    }
};

class CommonModel final
{
public:
    CommonModel(runtime::ResolvedModel resolved,
                std::vector<Shape> input_shapes,
                std::vector<Shape> output_shapes,
                PreparedFaceModelOptions options) try
        : resolved_(std::move(resolved)), input_shapes_(std::move(input_shapes)),
          output_shapes_(std::move(output_shapes)), options_(options),
          context_(resolved_.model->create_context())
    {
        const auto tensors = resolved_.model->tensors();
        inputs_ = match_tensors(tensors, true, input_shapes_, "prepared face input");
        const auto output_descriptors =
            match_tensors(tensors, false, output_shapes_, "prepared face output");
        outputs_.resize(output_descriptors.size());
        std::size_t aggregate = 0U;
        for (std::size_t i = 0U; i < outputs_.size(); ++i)
        {
            outputs_[i].descriptor = output_descriptors[i];
            outputs_[i].shape = output_shapes_[i];
            outputs_[i].allocate(options_.max_output_bytes);
            if (outputs_[i].bytes > options_.max_output_bytes - aggregate)
                throw_resource("aggregate outputs exceed max_output_bytes");
            aggregate += outputs_[i].bytes;
        }
    }
    catch (const FaceModelError&)
    {
        throw;
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
    catch (const std::bad_alloc&)
    {
        throw_resource("prepared face model allocation failed");
    }

    std::vector<std::vector<float>> run(const std::vector<PreparedTensorView>& prepared)
    {
        if (in_use_.test_and_set(std::memory_order_acquire))
            throw FaceModelError(FaceModelErrorCode::ConcurrentExecution,
                                 "prepared face model calls must not overlap");
        struct ClearFlag
        {
            std::atomic_flag& flag;
            ~ClearFlag() { flag.clear(std::memory_order_release); }
        } clear{in_use_};

        if (prepared.size() != inputs_.size()) throw_invalid("prepared input count mismatch");
        std::vector<InputStorage> input_storage;
        std::vector<runtime::TensorView> input_views;
        input_storage.reserve(prepared.size());
        input_views.reserve(prepared.size());
        std::size_t aggregate_input = 0U;
        for (std::size_t i = 0U; i < prepared.size(); ++i)
        {
            input_storage.push_back(prepare_input(prepared[i], input_shapes_[i],
                                                  inputs_[i].data_type,
                                                  options_.max_tensor_bytes));
            if (input_storage.back().bytes > options_.max_tensor_bytes - aggregate_input)
                throw_resource("aggregate inputs exceed max_tensor_bytes");
            aggregate_input += input_storage.back().bytes;
            input_views.push_back({inputs_[i].name, inputs_[i].data_type, input_shapes_[i],
                                   input_storage.back().data, input_storage.back().bytes,
                                   runtime::MemoryKind::Host, {}});
        }

        std::vector<runtime::MutableTensorView> output_views;
        output_views.reserve(outputs_.size());
        for (auto& output : outputs_) output_views.push_back(output.view());
        context_->run(input_views, output_views);

        std::vector<std::vector<float>> result;
        result.reserve(outputs_.size());
        for (const auto& output : outputs_) result.push_back(output.floats());
        return result;
    }

    const runtime::ExecutionRoute& route() const noexcept { return resolved_.route; }

private:
    runtime::ResolvedModel resolved_;
    std::vector<Shape> input_shapes_;
    std::vector<Shape> output_shapes_;
    PreparedFaceModelOptions options_;
    std::unique_ptr<runtime::ExecutionContext> context_;
    std::vector<runtime::TensorDescriptor> inputs_;
    std::vector<OutputBuffer> outputs_;
    std::atomic_flag in_use_ = ATOMIC_FLAG_INIT;
};

runtime::ResolvedModel resolve(runtime::Runtime& runtime,
                               const runtime::ModelPackage& package,
                               const runtime::ExecutionPolicy& policy,
                               const char* model_type,
                               const PreparedFaceModelOptions& options)
{
    validate_options(options);
    if (package.model_type() != model_type)
        throw_contract(std::string("ModelPackage model_type must be '") + model_type + "'");
    try
    {
        return runtime.load_model(package, policy);
    }
    catch (const runtime::RuntimeError& error)
    {
        throw_runtime(error.what());
    }
}

} // namespace

struct Face68::Impl final
{
    Impl(runtime::ResolvedModel resolved, PreparedFaceModelOptions options)
        : model(std::move(resolved),
                {{1, 3, kFace68InputExtent, kFace68InputExtent}},
                {{1, static_cast<std::int64_t>(kFace68LandmarkCount), kFace68LandmarkWidth},
                 {1, static_cast<std::int64_t>(kFace68LandmarkCount),
                  kFace68HeatmapExtent, kFace68HeatmapExtent}},
                options) {}
    CommonModel model;
};

Face68::Face68(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Face68::~Face68() = default;

std::unique_ptr<Face68> Face68::load(runtime::Runtime& runtime,
                                     const runtime::ModelPackage& package,
                                     const runtime::ExecutionPolicy& policy,
                                     const PreparedFaceModelOptions& options)
{
    try
    {
        return std::unique_ptr<Face68>(new Face68(std::make_unique<Impl>(
            resolve(runtime, package, policy, "face.face68", options), options)));
    }
    catch (const FaceModelError&) { throw; }
    catch (const std::bad_alloc&) { throw_resource("Face68 allocation failed"); }
}

Face68Result Face68::infer(const PreparedTensorView& prepared_input)
{
    try
    {
        const auto values = impl_->model.run({prepared_input});
        return detail::decode_face68(values[0].data(), values[0].size(), 1U).front();
    }
    catch (const FaceModelError&) { throw; }
    catch (const runtime::RuntimeError& error) { throw_runtime(error.what()); }
    catch (const std::bad_alloc&) { throw_resource("Face68 inference allocation failed"); }
}

const runtime::ExecutionRoute& Face68::execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty{};
    return impl_ ? impl_->model.route() : empty;
}

struct ArcFace::Impl final
{
    Impl(runtime::ResolvedModel resolved, PreparedFaceModelOptions options)
        : model(std::move(resolved),
                {{1, 3, kArcFaceInputExtent, kArcFaceInputExtent}},
                {{1, static_cast<std::int64_t>(kArcFaceEmbeddingLength)}}, options) {}
    CommonModel model;
};

ArcFace::ArcFace(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ArcFace::~ArcFace() = default;

std::unique_ptr<ArcFace> ArcFace::load(runtime::Runtime& runtime,
                                       const runtime::ModelPackage& package,
                                       const runtime::ExecutionPolicy& policy,
                                       const PreparedFaceModelOptions& options)
{
    try
    {
        return std::unique_ptr<ArcFace>(new ArcFace(std::make_unique<Impl>(
            resolve(runtime, package, policy, "face.arcface", options), options)));
    }
    catch (const FaceModelError&) { throw; }
    catch (const std::bad_alloc&) { throw_resource("ArcFace allocation failed"); }
}

ArcFaceResult ArcFace::infer(const PreparedTensorView& prepared_input)
{
    try
    {
        const auto values = impl_->model.run({prepared_input});
        return detail::decode_arcface(values[0].data(), values[0].size(), 1U).front();
    }
    catch (const FaceModelError&) { throw; }
    catch (const runtime::RuntimeError& error) { throw_runtime(error.what()); }
    catch (const std::bad_alloc&) { throw_resource("ArcFace inference allocation failed"); }
}

const runtime::ExecutionRoute& ArcFace::execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty{};
    return impl_ ? impl_->model.route() : empty;
}

struct AgeGender::Impl final
{
    Impl(runtime::ResolvedModel resolved, PreparedFaceModelOptions options)
        : model(std::move(resolved),
                {{1, 3, kAgeGenderInputExtent, kAgeGenderInputExtent}},
                {{1, static_cast<std::int64_t>(kAgeGenderLogitCount)}}, options) {}
    CommonModel model;
};

AgeGender::AgeGender(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
AgeGender::~AgeGender() = default;

std::unique_ptr<AgeGender> AgeGender::load(runtime::Runtime& runtime,
                                           const runtime::ModelPackage& package,
                                           const runtime::ExecutionPolicy& policy,
                                           const PreparedFaceModelOptions& options)
{
    try
    {
        return std::unique_ptr<AgeGender>(new AgeGender(std::make_unique<Impl>(
            resolve(runtime, package, policy, "face.age-gender", options), options)));
    }
    catch (const FaceModelError&) { throw; }
    catch (const std::bad_alloc&) { throw_resource("AgeGender allocation failed"); }
}

AgeGenderResult AgeGender::infer(const PreparedTensorView& prepared_input)
{
    try
    {
        const auto values = impl_->model.run({prepared_input});
        return detail::decode_age_gender(values[0].data(), values[0].size(), 1U).front();
    }
    catch (const FaceModelError&) { throw; }
    catch (const runtime::RuntimeError& error) { throw_runtime(error.what()); }
    catch (const std::bad_alloc&) { throw_resource("AgeGender inference allocation failed"); }
}

const runtime::ExecutionRoute& AgeGender::execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty{};
    return impl_ ? impl_->model.route() : empty;
}

struct InSwapper::Impl final
{
    Impl(runtime::ResolvedModel resolved, PreparedFaceModelOptions options)
        : model(std::move(resolved),
                {{1, 3, kInSwapperInputExtent, kInSwapperInputExtent},
                 {1, static_cast<std::int64_t>(kInSwapperEmbeddingLength)}},
                {{1, 3, kInSwapperInputExtent, kInSwapperInputExtent}}, options) {}
    CommonModel model;
};

InSwapper::InSwapper(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
InSwapper::~InSwapper() = default;

std::unique_ptr<InSwapper> InSwapper::load(runtime::Runtime& runtime,
                                            const runtime::ModelPackage& package,
                                            const runtime::ExecutionPolicy& policy,
                                            const PreparedFaceModelOptions& options)
{
    try
    {
        return std::unique_ptr<InSwapper>(new InSwapper(std::make_unique<Impl>(
            resolve(runtime, package, policy, "face.inswapper", options), options)));
    }
    catch (const FaceModelError&) { throw; }
    catch (const std::bad_alloc&) { throw_resource("InSwapper allocation failed"); }
}

InSwapperResult InSwapper::infer(const PreparedTensorView& prepared_target,
                                  const PreparedTensorView& projected_source)
{
    try
    {
        auto values = impl_->model.run({prepared_target, projected_source});
        InSwapperResult result;
        result.values = std::move(values[0]);
        return result;
    }
    catch (const FaceModelError&) { throw; }
    catch (const runtime::RuntimeError& error) { throw_runtime(error.what()); }
    catch (const std::bad_alloc&) { throw_resource("InSwapper inference allocation failed"); }
}

const runtime::ExecutionRoute& InSwapper::execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty{};
    return impl_ ? impl_->model.route() : empty;
}

struct GfpGan::Impl final
{
    Impl(runtime::ResolvedModel resolved, PreparedFaceModelOptions options)
        : model(std::move(resolved),
                {{1, 3, kGfpGanInputExtent, kGfpGanInputExtent}},
                {{1, 3, kGfpGanInputExtent, kGfpGanInputExtent}}, options) {}
    CommonModel model;
};

GfpGan::GfpGan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GfpGan::~GfpGan() = default;

std::unique_ptr<GfpGan> GfpGan::load(runtime::Runtime& runtime,
                                      const runtime::ModelPackage& package,
                                      const runtime::ExecutionPolicy& policy,
                                      const PreparedFaceModelOptions& options)
{
    try
    {
        return std::unique_ptr<GfpGan>(new GfpGan(std::make_unique<Impl>(
            resolve(runtime, package, policy, "face.gfpgan", options), options)));
    }
    catch (const FaceModelError&) { throw; }
    catch (const std::bad_alloc&) { throw_resource("GFPGAN allocation failed"); }
}

GfpGanResult GfpGan::infer(const PreparedTensorView& prepared_input)
{
    try
    {
        auto values = impl_->model.run({prepared_input});
        GfpGanResult result;
        result.values = std::move(values[0]);
        return result;
    }
    catch (const FaceModelError&) { throw; }
    catch (const runtime::RuntimeError& error) { throw_runtime(error.what()); }
    catch (const std::bad_alloc&) { throw_resource("GFPGAN inference allocation failed"); }
}

const runtime::ExecutionRoute& GfpGan::execution_route() const noexcept
{
    static const runtime::ExecutionRoute empty{};
    return impl_ ? impl_->model.route() : empty;
}

} // namespace kfcore::face_models
