#pragma once
#include <kfcore/gesture_interaction/action_gate.hpp>
#include <filesystem>
#include <string>
#include <memory>

namespace kfcore::gesture_interaction {
struct RecordedClip {
    CompositionClip samples;
    Composition label = Composition::None;
    std::uint64_t batch = 0;
    std::uint64_t recorded_at_ms = 0;
};
struct SessionArchive {
    CompositionOptions options;
    ActionGateOptions gate;
    CompositionWeights weights;
    bool trained = false;
    std::vector<RecordedClip> training, heldout;
};
struct ExperimentArchive {
    std::string pipeline_identity;
    std::vector<SessionArchive> sessions;
};
inline constexpr std::size_t kMaximumExperimentBytes = 32 * 1024 * 1024;
// Validates all raw clips and restores the exact model; never fits a new readout.
std::unique_ptr<CompositionEsn> restore_model(const SessionArchive& session);
// Explicit little-endian wire format; SHA-256 detects damage, not authenticity.
std::vector<std::uint8_t> encode_experiment(const ExperimentArchive& experiment);
ExperimentArchive decode_experiment(const std::vector<std::uint8_t>& bytes, const std::string& expected_pipeline);
// Windows-only adapter: exclusive temporary creation, flush, then no-replace publication.
#ifdef _WIN32
void save_experiment(const std::filesystem::path& path, const ExperimentArchive& experiment);
ExperimentArchive load_experiment(const std::filesystem::path& path, const std::string& expected_pipeline);
#endif
} // namespace kfcore::gesture_interaction
