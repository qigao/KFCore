#pragma once

#include "kfcore/face_applications/tensorrt.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

namespace kfcore::face_applications::demo
{

inline constexpr std::size_t kTimingHistoryCapacity = 120U;

struct TimingRow
{
    std::string_view label;
    std::optional<FaceSwapDuration> current;
    std::optional<FaceSwapDuration> p50;
    std::optional<FaceSwapDuration> p95;
};

class TimingHistory final
{
public:
    void add(const FaceSwapTimingReport& report) noexcept;

    [[nodiscard]] std::size_t sample_count() const noexcept;
    [[nodiscard]] std::vector<TimingRow> rows() const;

private:
    using Accessor = std::optional<FaceSwapDuration> (*)(const FaceSwapTimingReport&);

    [[nodiscard]] TimingRow make_row(std::string_view label, Accessor accessor) const;

    std::array<FaceSwapTimingReport, kTimingHistoryCapacity> reports_ {};
    std::size_t next_ = 0U;
    std::size_t count_ = 0U;
};

} // namespace kfcore::face_applications::demo
