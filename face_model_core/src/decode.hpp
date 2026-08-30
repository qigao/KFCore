#pragma once

#include "kfcore/face_models/types.hpp"

#include <cstddef>
#include <vector>

namespace kfcore::face_models::detail
{

std::vector<Face68Result> decode_face68(const float* values, std::size_t element_count,
                                        std::size_t batch);
std::vector<ArcFaceResult> decode_arcface(const float* values, std::size_t element_count,
                                          std::size_t batch);
std::vector<AgeGenderResult> decode_age_gender(const float* values, std::size_t element_count,
                                               std::size_t batch);

} // namespace kfcore::face_models::detail
