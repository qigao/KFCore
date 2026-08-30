#pragma once

#include "kfcore/tensorrt/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace kfcore::tensorrt::detail
{

std::size_t scalar_byte_size(DataType data_type);

std::size_t checked_shape_byte_size(const TensorShape& shape, DataType data_type,
                                    const char* stage);

std::size_t checked_data_dependent_shape_byte_size(const TensorShape& shape,
                                                   DataType data_type, const char* stage);

void validate_profile_bounds(const TensorProfile& profile, const std::string& tensor_name);

void validate_tensor_metadata(const std::vector<TensorDescriptor>& tensors,
                              const EngineOptions&                 options);

void validate_input_views(const std::vector<TensorDescriptor>& expected,
                          const std::vector<TensorView>& views, std::size_t max_aggregate_bytes);

void validate_output_views(const std::vector<TensorDescriptor>&  expected,
                           const std::vector<MutableTensorView>& views,
                           std::size_t                           max_aggregate_bytes);

void validate_dynamic_output_requests(
    const std::vector<TensorDescriptor>& expected,
    const std::vector<DynamicOutputRequest>& requests, std::size_t max_aggregate_bytes);

} // namespace kfcore::tensorrt::detail
