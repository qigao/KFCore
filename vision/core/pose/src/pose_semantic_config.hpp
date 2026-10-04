#pragma once

#include "simcc_decode.hpp"

#include "kfcore/runtime/model_package.hpp"

namespace kfcore::pose::detail
{

struct PoseSemanticConfig
{
    bool decode_visibility = false;
    SimccVisibilityDecodeDesc visibility;
};

[[nodiscard]] PoseSemanticConfig
load_pose_semantic_config(const runtime::ModelPackage& package);

} // namespace kfcore::pose::detail
