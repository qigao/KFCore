#pragma once

#include "kfcore/image_processor/types.hpp"
#include "kfcore/relation/types.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/resolver.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace kfcore::relation
{

inline constexpr std::string_view kRelateAnythingModelType =
    "relation.relate-anything";

struct RelateAnythingOptions
{
    std::int32_t input_size = 0;
    std::size_t max_boxes = kLegacyRelationMaxBoxes;
    std::size_t max_pairs = 128U;
    std::vector<std::string> predicates;
    float threshold = 0.40F;
    float pair_weight = 1.0F;
    float calibration_a = 1.0F;
    float calibration_b = 0.0F;
    std::size_t top_k = 20U;
    bool weight_ranking_by_detector_score = true;
    std::size_t max_source_bytes = 64U * 1024U * 1024U;
    std::size_t max_tensor_bytes = 64U * 1024U * 1024U;
    std::size_t max_output_bytes = 64U * 1024U * 1024U;
};

[[nodiscard]] inline RelateAnythingOptions
apache_released_relate_anything_options()
{
    RelateAnythingOptions options;
    options.max_boxes = kApacheReleasedMaxBoxes;
    return options;
}

class RelateAnything final
{
public:
    ~RelateAnything();

    RelateAnything(const RelateAnything&) = delete;
    RelateAnything& operator=(const RelateAnything&) = delete;

    [[nodiscard]] static std::unique_ptr<RelateAnything>
    load(runtime::Runtime& runtime,
         const runtime::ModelPackage& package,
         const runtime::ExecutionPolicy& policy,
         const RelateAnythingOptions& options);

    [[nodiscard]] RelationFrame infer(const image::ImageView& image,
                                      const std::vector<Region>& regions);

    [[nodiscard]] std::int32_t input_size() const noexcept;
    [[nodiscard]] std::size_t max_boxes() const noexcept;
    [[nodiscard]] std::size_t max_pairs() const noexcept;
    [[nodiscard]] std::uint64_t vocabulary_version() const noexcept;
    [[nodiscard]] const std::vector<std::string>& predicates() const noexcept;
    [[nodiscard]] const runtime::ExecutionRoute& execution_route() const noexcept;

private:
    struct Impl;
    explicit RelateAnything(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace kfcore::relation
