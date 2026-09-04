#pragma once

#include "decode.hpp"
#include "kfcore/face_models/tensorrt.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

namespace kfcore::face_models::detail
{

class AdapterCallGuard final
{
public:
    AdapterCallGuard(std::atomic_flag& in_use, const char* model_name);
    ~AdapterCallGuard();

    AdapterCallGuard(const AdapterCallGuard&)            = delete;
    AdapterCallGuard& operator=(const AdapterCallGuard&) = delete;

private:
    std::atomic_flag& in_use_;
};

class BorrowedInputGuard final
{
public:
    BorrowedInputGuard(kfcore::tensorrt::TensorView&       target,
                       const kfcore::tensorrt::TensorView& source) noexcept;
    ~BorrowedInputGuard() noexcept;

    BorrowedInputGuard(const BorrowedInputGuard&)            = delete;
    BorrowedInputGuard& operator=(const BorrowedInputGuard&) = delete;

private:
    kfcore::tensorrt::TensorView& target_;
};

class BorrowedOutputGuard final
{
public:
    BorrowedOutputGuard(kfcore::tensorrt::MutableTensorView&       target,
                        const kfcore::tensorrt::MutableTensorView& source) noexcept;
    ~BorrowedOutputGuard() noexcept;

    BorrowedOutputGuard(const BorrowedOutputGuard&)            = delete;
    BorrowedOutputGuard& operator=(const BorrowedOutputGuard&) = delete;

private:
    kfcore::tensorrt::MutableTensorView& target_;
    void*                                previous_data_ = nullptr;
    std::size_t                          previous_byte_size_ = 0;
    kfcore::tensorrt::MemoryKind         previous_memory_kind_ =
        kfcore::tensorrt::MemoryKind::Host;
};

struct BatchBounds
{
    std::size_t minimum = 0;
    std::size_t maximum = 0;
};

struct SingleOutputContract
{
    std::string input_name;
    std::string output_name;
    BatchBounds batch;
    std::size_t output_float_capacity = 0;
};

struct Face68Contract
{
    std::string input_name;
    std::string landmark_output_name;
    std::string heatmap_output_name;
    BatchBounds batch;
    bool        has_heatmaps            = false;
    std::size_t landmark_float_capacity = 0;
    std::size_t heatmap_float_capacity  = 0;
};

struct InSwapperContract
{
    std::string target_input_name;
    std::string source_input_name;
    std::string output_name;
    BatchBounds batch;
    std::size_t output_float_capacity = 0;
};

void validate_face68_options(const Face68Options& options);
void validate_arcface_options(const ArcFaceOptions& options);
void validate_age_gender_options(const AgeGenderOptions& options);
void validate_inswapper_options(const InSwapperOptions& options);
void validate_gfpgan_options(const GfpGanOptions& options);

Face68Contract validate_face68_contract(
    const std::vector<kfcore::tensorrt::TensorDescriptor>& tensors,
    const Face68Options& options);

SingleOutputContract validate_arcface_contract(
    const std::vector<kfcore::tensorrt::TensorDescriptor>& tensors,
    const ArcFaceOptions& options);

SingleOutputContract validate_age_gender_contract(
    const std::vector<kfcore::tensorrt::TensorDescriptor>& tensors,
    const AgeGenderOptions& options);

InSwapperContract validate_inswapper_contract(
    const std::vector<kfcore::tensorrt::TensorDescriptor>& tensors,
    const InSwapperOptions& options);

SingleOutputContract validate_gfpgan_contract(
    const std::vector<kfcore::tensorrt::TensorDescriptor>& tensors,
    const GfpGanOptions& options);

void validate_prepared_input(const kfcore::tensorrt::TensorView& input,
                             const std::string& expected_name, const BatchBounds& batch,
                             const std::array<std::int64_t, 3>& fixed_dimensions,
                             const char* model_name);

void validate_prepared_vector_input(const kfcore::tensorrt::TensorView& input,
                                    const std::string& expected_name,
                                    const BatchBounds& batch,
                                    const std::vector<std::int64_t>& fixed_dimensions,
                                    const char* model_name);

void validate_prepared_output(const kfcore::tensorrt::MutableTensorView& output,
                              const std::string& expected_name, std::int64_t expected_batch,
                              const std::array<std::int64_t, 3>& fixed_dimensions,
                              const char* model_name);

[[noreturn]] void rethrow_tensorrt(const kfcore::tensorrt::TensorRtError& error,
                                   const char* model_name, const char* stage);
[[noreturn]] void throw_allocation_failure(const char* model_name, const char* stage);
[[noreturn]] void throw_capacity_failure(const char* model_name, const char* stage);

} // namespace kfcore::face_models::detail
