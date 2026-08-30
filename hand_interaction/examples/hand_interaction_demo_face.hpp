#pragma once

#include "hand_interaction_demo_cli.hpp"

#include "kfcore/face_models/core.hpp"

#include <memory>

namespace kfcore::hand_interaction::demo
{

[[nodiscard]] std::unique_ptr<face_models::FaceMeshPipeline>
make_face_pipeline(const Arguments& arguments);

} // namespace kfcore::hand_interaction::demo
