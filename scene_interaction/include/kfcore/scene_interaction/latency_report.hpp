#pragma once

#include "kfcore/scene_interaction/pipeline.hpp"

#include <string>

namespace kfcore::scene_interaction
{

inline constexpr const char* kSceneBehaviorTimingSampleSchema =
    "kfcore.scene-behavior-timing-sample/1";

[[nodiscard]] std::string
scene_behavior_timing_json(const SceneBehaviorTiming& timing);

} // namespace kfcore::scene_interaction
