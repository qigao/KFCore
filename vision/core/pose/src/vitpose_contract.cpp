#include "vitpose_contract.hpp"

#include "kfcore/pose/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace kfcore::pose::detail
{
namespace
{

[[noreturn]] void throw_contract(const std::string& detail)
{
    throw PoseError(
        PoseErrorCode::ModelContractMismatch,
        "ViTPose model contract: " + detail);
}

std::int32_t resolve_input_extent(
    std::int64_t declared,
    std::int32_t requested,
    const char* name)
{
    if (declared > 0)
    {
        if (declared >
            (std::numeric_limits<std::int32_t>::max)())
        {
            throw_contract(
                std::string(name) + " exceeds int32 range");
        }
        const auto value =
            static_cast<std::int32_t>(declared);
        if (requested > 0 && requested != value)
        {
            throw_contract(
                std::string(name) +
                " conflicts with static artifact shape");
        }
        if (value <= 1)
        {
            throw_contract(
                std::string(name) +
                " must be greater than one for UDP preprocessing");
        }
        return value;
    }

    if (declared == -1 && requested > 1)
    {
        return requested;
    }

    throw_contract(
        std::string(name) +
        " is dynamic; VitPoseOptions must provide an explicit size");
}

bool supported_float(runtime::DataType type) noexcept
{
    return type == runtime::DataType::Float32 ||
           type == runtime::DataType::Float16;
}

std::size_t positive_extent(
    std::int64_t value,
    const char* name)
{
    if (value <= 1)
    {
        throw_contract(
            std::string(name) +
            " must be a static extent greater than one");
    }
    const auto unsigned_value =
        static_cast<std::uint64_t>(value);
    if (unsigned_value >
        static_cast<std::uint64_t>(
            (std::numeric_limits<std::size_t>::max)()))
    {
        throw_contract(
            std::string(name) +
            " exceeds size_t range");
    }
    return static_cast<std::size_t>(unsigned_value);
}

} // namespace

VitPoseTensorContract
resolve_vitpose_tensor_contract(
    const std::vector<runtime::TensorDescriptor>& tensors,
    const PoseSchema& schema,
    std::int32_t requested_input_width,
    std::int32_t requested_input_height,
    const std::string& requested_input_name,
    const std::string& requested_heatmap_name)
{
    if ((requested_input_width < 0) ||
        (requested_input_height < 0) ||
        ((requested_input_width == 0) !=
         (requested_input_height == 0)))
    {
        throw_contract(
            "requested input width and height must both be zero "
            "or both be positive");
    }
    if (schema.output_map.empty())
    {
        throw_contract(
            "selected PoseSchema has an empty output map");
    }

    std::vector<runtime::TensorDescriptor> inputs;
    std::vector<runtime::TensorDescriptor> outputs;
    for (const auto& tensor : tensors)
    {
        (tensor.is_input ? inputs : outputs).push_back(tensor);
    }

    if (inputs.size() != 1U)
    {
        throw_contract(
            "ViTPose requires exactly one image input tensor");
    }
    if (outputs.size() != 1U)
    {
        throw_contract(
            "ViTPose v1 requires exactly one raw heatmap output tensor");
    }

    VitPoseTensorContract result;
    result.input = inputs.front();
    result.heatmap = outputs.front();

    if (!requested_input_name.empty() &&
        result.input.name != requested_input_name)
    {
        throw_contract(
            "input tensor name does not match VitPoseOptions::input_name");
    }
    if (!requested_heatmap_name.empty() &&
        result.heatmap.name != requested_heatmap_name)
    {
        throw_contract(
            "heatmap tensor name does not match VitPoseOptions::heatmap_name");
    }
    if (!supported_float(result.input.data_type))
    {
        throw_contract(
            "input tensor must use FP32 or FP16");
    }
    if (!supported_float(result.heatmap.data_type))
    {
        throw_contract(
            "heatmap output must use FP32 or FP16");
    }

    if (result.input.shape.size() != 4U)
    {
        throw_contract(
            "input tensor must be NCHW rank 4");
    }
    result.input_shape = result.input.shape;
    if (result.input_shape[0] == -1)
    {
        result.input_shape[0] = 1;
    }
    if (result.input_shape[0] != 1)
    {
        throw_contract(
            "ViTPose v1 supports batch size 1 per execution call");
    }
    if (result.input_shape[1] == -1)
    {
        result.input_shape[1] = 3;
    }
    if (result.input_shape[1] != 3)
    {
        throw_contract(
            "input tensor must have three channels");
    }

    result.input_height = resolve_input_extent(
        result.input_shape[2],
        requested_input_height,
        "input height");
    result.input_width = resolve_input_extent(
        result.input_shape[3],
        requested_input_width,
        "input width");
    result.input_shape[2] = result.input_height;
    result.input_shape[3] = result.input_width;

    result.heatmap_shape = result.heatmap.shape;
    std::size_t keypoint_axis = 0U;
    std::size_t height_axis = 0U;
    std::size_t width_axis = 0U;

    if (result.heatmap_shape.size() == 4U)
    {
        if (result.heatmap_shape[0] == -1)
        {
            result.heatmap_shape[0] = 1;
        }
        if (result.heatmap_shape[0] != 1)
        {
            throw_contract(
                "heatmap output batch must be one");
        }
        keypoint_axis = 1U;
        height_axis = 2U;
        width_axis = 3U;
    }
    else if (result.heatmap_shape.size() == 3U)
    {
        keypoint_axis = 0U;
        height_axis = 1U;
        width_axis = 2U;
    }
    else
    {
        throw_contract(
            "heatmap output must have shape [K,H,W] or [1,K,H,W]");
    }

    const std::size_t keypoint_count =
        schema.output_map.size();
    if (result.heatmap_shape[keypoint_axis] == -1)
    {
        result.heatmap_shape[keypoint_axis] =
            static_cast<std::int64_t>(keypoint_count);
    }
    if (result.heatmap_shape[keypoint_axis] !=
        static_cast<std::int64_t>(keypoint_count))
    {
        throw_contract(
            "heatmap keypoint axis does not match PoseSchema output map");
    }

    result.heatmap_height = positive_extent(
        result.heatmap_shape[height_axis],
        "heatmap height");
    result.heatmap_width = positive_extent(
        result.heatmap_shape[width_axis],
        "heatmap width");

    return result;
}

PoseResult
decode_vitpose_pose(
    const HeatmapView& heatmap,
    const HeatmapSourceGeometry& geometry,
    const RectF& source_box,
    const PoseSchema& schema)
{
    if (heatmap.keypoints != schema.output_map.size())
    {
        throw_contract(
            "heatmap keypoint count does not match PoseSchema output map");
    }

    std::vector<DecodedHeatmapKeypoint>
        decoded(heatmap.keypoints);
    decode_gaussian_heatmap_udp(
        heatmap,
        {11U},
        decoded.data(),
        decoded.size());

    PoseResult result;
    result.source_box = source_box;
    result.schema_id = schema.id;
    result.capabilities = kPoseCapabilityConfidence;
    result.keypoints.reserve(decoded.size());

    for (std::size_t channel = 0U;
         channel < decoded.size();
         ++channel)
    {
        PoseKeypoint point;
        point.id = schema.output_map[channel];
        point.confidence = decoded[channel].confidence;

        if (point.confidence > 0.0F)
        {
            const auto source = heatmap_udp_to_source(
                geometry,
                decoded[channel].x,
                decoded[channel].y,
                heatmap.width,
                heatmap.height);
            point.x = source.first;
            point.y = source.second;
            point.flags |= kPoseKeypointFlagValid;
        }

        result.keypoints.push_back(point);
    }

    return result;
}

} // namespace kfcore::pose::detail
