#include "face_swap_demo_metrics.hpp"

#include <algorithm>
#include <utility>

namespace kfcore::face_applications::demo
{
namespace
{

std::optional<FaceSwapDuration> nearest_rank(std::vector<FaceSwapDuration> values,
                                             std::size_t percentile)
{
    if (values.empty())
    {
        return std::nullopt;
    }
    std::sort(values.begin(), values.end());
    const std::size_t rank = (percentile * values.size() + 99U) / 100U;
    return values[rank - 1U];
}

#define KFCORE_REQUIRED_ACCESSOR(name, expression)                                      \
    std::optional<FaceSwapDuration> name(const FaceSwapTimingReport& report)            \
    {                                                                                    \
        return expression;                                                               \
    }

KFCORE_REQUIRED_ACCESSOR(total, report.total)
KFCORE_REQUIRED_ACCESSOR(source_total, report.source_analysis.total)
KFCORE_REQUIRED_ACCESSOR(source_stage, report.source_analysis.initial_staging)
KFCORE_REQUIRED_ACCESSOR(source_detect, report.source_analysis.detection)
KFCORE_REQUIRED_ACCESSOR(source_face68_preprocess, report.source_analysis.face68_preprocess)
KFCORE_REQUIRED_ACCESSOR(source_face68_inference,
                         report.source_analysis.face68_inference_and_postprocess)
KFCORE_REQUIRED_ACCESSOR(source_arcface_preprocess,
                         report.source_analysis.arcface_preprocess)
KFCORE_REQUIRED_ACCESSOR(source_arcface_inference, report.source_analysis.arcface_inference)
KFCORE_REQUIRED_ACCESSOR(target_total, report.target_analysis.total)
KFCORE_REQUIRED_ACCESSOR(target_stage, report.target_analysis.initial_staging)
KFCORE_REQUIRED_ACCESSOR(target_detect, report.target_analysis.detection)
KFCORE_REQUIRED_ACCESSOR(target_face68_preprocess, report.target_analysis.face68_preprocess)
KFCORE_REQUIRED_ACCESSOR(target_face68_inference,
                         report.target_analysis.face68_inference_and_postprocess)
KFCORE_REQUIRED_ACCESSOR(target_arcface_preprocess,
                         report.target_analysis.arcface_preprocess)
KFCORE_REQUIRED_ACCESSOR(target_arcface_inference, report.target_analysis.arcface_inference)
KFCORE_REQUIRED_ACCESSOR(project, report.embedding_projection)
KFCORE_REQUIRED_ACCESSOR(inswapper_preprocess, report.inswapper_preprocess)
KFCORE_REQUIRED_ACCESSOR(inswapper_inference, report.inswapper_inference_and_decode)
KFCORE_REQUIRED_ACCESSOR(inswapper_composition, report.inswapper_composition)

#undef KFCORE_REQUIRED_ACCESSOR

std::optional<FaceSwapDuration> source_age_gender(const FaceSwapTimingReport& report)
{
    return report.source_analysis.age_gender;
}

std::optional<FaceSwapDuration> target_age_gender(const FaceSwapTimingReport& report)
{
    return report.target_analysis.age_gender;
}

std::optional<FaceSwapDuration> gfpgan_preprocess(const FaceSwapTimingReport& report)
{
    return report.gfpgan_preprocess;
}

std::optional<FaceSwapDuration> gfpgan_inference(const FaceSwapTimingReport& report)
{
    return report.gfpgan_inference_and_decode;
}

std::optional<FaceSwapDuration> gfpgan_composition(const FaceSwapTimingReport& report)
{
    return report.gfpgan_composition;
}

} // namespace

void TimingHistory::add(const FaceSwapTimingReport& report) noexcept
{
    reports_[next_] = report;
    next_ = (next_ + 1U) % kTimingHistoryCapacity;
    count_ = std::min(count_ + 1U, kTimingHistoryCapacity);
}

std::size_t TimingHistory::sample_count() const noexcept
{
    return count_;
}

TimingRow TimingHistory::make_row(std::string_view label, Accessor accessor) const
{
    if (count_ == 0U)
    {
        return { label, std::nullopt, std::nullopt, std::nullopt };
    }

    const std::size_t latest = (next_ + kTimingHistoryCapacity - 1U) % kTimingHistoryCapacity;
    std::vector<FaceSwapDuration> values;
    values.reserve(count_);
    for (std::size_t index = 0U; index < count_; ++index)
    {
        const auto value = accessor(reports_[index]);
        if (value)
        {
            values.push_back(*value);
        }
    }
    return { label, accessor(reports_[latest]), nearest_rank(values, 50U),
             nearest_rank(std::move(values), 95U) };
}

std::vector<TimingRow> TimingHistory::rows() const
{
    return {
        make_row("Total", total),
        make_row("Source total", source_total),
        make_row("Source staging", source_stage),
        make_row("Source detect", source_detect),
        make_row("Source Face68 prep", source_face68_preprocess),
        make_row("Source Face68 run", source_face68_inference),
        make_row("Source ArcFace prep", source_arcface_preprocess),
        make_row("Source ArcFace run", source_arcface_inference),
        make_row("Source age/gender", source_age_gender),
        make_row("Target total", target_total),
        make_row("Target staging", target_stage),
        make_row("Target detect", target_detect),
        make_row("Target Face68 prep", target_face68_preprocess),
        make_row("Target Face68 run", target_face68_inference),
        make_row("Target ArcFace prep", target_arcface_preprocess),
        make_row("Target ArcFace run", target_arcface_inference),
        make_row("Target age/gender", target_age_gender),
        make_row("Embedding project", project),
        make_row("InSwapper prep", inswapper_preprocess),
        make_row("InSwapper run", inswapper_inference),
        make_row("InSwapper compose", inswapper_composition),
        make_row("GFPGAN preprocess", gfpgan_preprocess),
        make_row("GFPGAN run", gfpgan_inference),
        make_row("GFPGAN compose", gfpgan_composition),
    };
}

} // namespace kfcore::face_applications::demo
