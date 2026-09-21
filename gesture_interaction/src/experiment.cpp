#include <kfcore/gesture_interaction/experiment.hpp>
#include <salts/crypto.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace kfcore::gesture_interaction {
namespace {
constexpr std::uint64_t kMagic = 0x315058454E53464BULL; // KFSNEXP1
constexpr std::uint64_t kFormatVersion = 1, kFeatureVersion = 1;
constexpr std::size_t kMaximumIdentity = 2048, kDigestBytes = SALTS_CRYPTO_SHA256_DIGEST_SIZE;
static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559);
struct Writer {
    std::vector<std::uint8_t> bytes;
    void integer(std::uint64_t value) {
        if (bytes.size() > kMaximumExperimentBytes-kDigestBytes-sizeof(value))
            throw std::runtime_error("experiment exceeds 32 MiB limit");
        for (int shift = 0; shift < 64; shift += 8) bytes.push_back(std::uint8_t(value >> shift));
    }
    void number(double value) {
        if (!std::isfinite(value)) throw std::invalid_argument("non-finite experiment value");
        std::uint64_t bits; std::memcpy(&bits, &value, sizeof(bits)); integer(bits);
    }
};
struct Reader {
    const std::vector<std::uint8_t>& bytes;
    std::size_t position = 0, end;
    std::uint64_t integer() {
        if (end-position < sizeof(std::uint64_t)) throw std::invalid_argument("truncated experiment");
        std::uint64_t value = 0;
        for (int shift = 0; shift < 64; shift += 8) value |= std::uint64_t(bytes[position++]) << shift;
        return value;
    }
    std::uint64_t bounded(std::uint64_t maximum) {
        const auto value = integer();
        if (value > maximum) throw std::invalid_argument("experiment count or enum outside bounds");
        return value;
    }
    double number() {
        const auto bits = integer(); double value;
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value)) throw std::invalid_argument("non-finite experiment value");
        return value;
    }
    float scalar() {
        const auto value = number();
        if (std::abs(value) > std::numeric_limits<float>::max()) throw std::invalid_argument("experiment float overflow");
        return float(value);
    }
};
std::array<std::uint8_t, kDigestBytes> digest(const void* data, std::size_t size) {
    std::array<std::uint8_t, kDigestBytes> result{};
    if (salts_crypto_sha256(data, size, result.data()) != SALTS_CRYPTO_OK)
        throw std::runtime_error("experiment SHA-256 failed");
    return result;
}
void write_options(Writer& w, const CompositionOptions& o, const ActionGateOptions& g) {
    w.integer(std::uint64_t(o.task));
    w.number(o.displacement_scale); w.number(o.velocity_scale);
    w.number(o.live_stride_seconds); w.number(o.record_countdown_seconds);
    w.number(o.duration_seconds); w.number(o.maximum_gap_seconds);
    w.integer(o.maximum_samples); w.integer(o.maximum_clips); w.integer(o.minimum_clips_per_class);
    w.number(o.ridge); w.number(o.leak); w.number(o.spectral_radius); w.number(o.input_scale); w.integer(o.seed);
    w.number(g.minimum_score); w.number(g.minimum_margin); w.number(g.confirmation_seconds); w.number(g.rearm_seconds);
}
void read_options(Reader& r, CompositionOptions& o, ActionGateOptions& g) {
    o.task = CompositionTask(r.bounded(std::uint64_t(CompositionTask::Motion)));
    o.displacement_scale = r.scalar(); o.velocity_scale = r.scalar();
    o.live_stride_seconds = r.number(); o.record_countdown_seconds = r.number();
    o.duration_seconds = r.number(); o.maximum_gap_seconds = r.number();
    constexpr std::size_t kMaximumCount = 4096;
    o.maximum_samples = std::size_t(r.bounded(kMaximumCount));
    o.maximum_clips = std::size_t(r.bounded(kMaximumCount));
    o.minimum_clips_per_class = int(r.bounded(kMaximumCount));
    o.ridge = r.scalar(); o.leak = r.scalar(); o.spectral_radius = r.scalar(); o.input_scale = r.scalar(); o.seed = r.integer();
    g.minimum_score = r.scalar(); g.minimum_margin = r.scalar(); g.confirmation_seconds = r.number(); g.rearm_seconds = r.number();
}
void write_clips(Writer& w, const std::vector<RecordedClip>& clips) {
    w.integer(clips.size());
    for (const auto& clip : clips) {
        w.integer(std::uint64_t(clip.label)); w.integer(clip.batch); w.integer(clip.recorded_at_ms);
        w.integer(clip.samples.size());
        for (const auto& sample : clip.samples) {
            w.number(sample.seconds);
            for (float value : sample.values) w.number(value);
            w.integer(sample.wrist.has_value());
            if (sample.wrist) for (float value : *sample.wrist) w.number(value);
        }
    }
}
std::vector<RecordedClip> read_clips(Reader& r, const CompositionOptions& options) {
    const auto count = r.bounded(options.maximum_clips);
    std::vector<RecordedClip> clips;
    for (std::uint64_t i = 0; i < count; ++i) {
        RecordedClip clip;
        clip.label = Composition(r.bounded(std::uint64_t(Composition::Wave)));
        (void)composition_index(options.task, clip.label);
        clip.batch = r.integer(); clip.recorded_at_ms = r.integer();
        const auto samples = r.bounded(options.maximum_samples);
        constexpr std::size_t kMinimumSampleBytes = (kCompositionInputs+2)*sizeof(std::uint64_t);
        if (samples > (r.end-r.position)/kMinimumSampleBytes) throw std::invalid_argument("truncated sample data");
        clip.samples.reserve(std::size_t(samples));
        for (std::uint64_t j = 0; j < samples; ++j) {
            CompositionSample sample{r.number()};
            for (float& value : sample.values) value = r.scalar();
            if (r.bounded(1)) sample.wrist = std::array<float,2>{r.scalar(), r.scalar()};
            clip.samples.push_back(sample);
        }
        clips.push_back(std::move(clip));
    }
    return clips;
}
void check_layout(const ExperimentArchive& archive) {
    if (archive.pipeline_identity.empty() || archive.pipeline_identity.size() > kMaximumIdentity)
        throw std::invalid_argument("invalid experiment pipeline identity");
    const bool legacy = archive.sessions.size() == 1 && archive.sessions[0].options.task == CompositionTask::Legacy;
    const bool actions = archive.sessions.size() == 2 && archive.sessions[0].options.task == CompositionTask::Interaction &&
        archive.sessions[1].options.task == CompositionTask::Motion;
    if (!legacy && !actions) throw std::invalid_argument("experiment branch layout mismatch");
}
}
std::vector<std::uint8_t> encode_experiment(const ExperimentArchive& archive) {
    check_layout(archive);
    Writer w;
    w.integer(kMagic); w.integer(kFormatVersion); w.integer(kFeatureVersion);
    w.integer(kCompositionInputs); w.integer(kMotionInputs); w.integer(kCompositionFeatures);
    w.integer(archive.pipeline_identity.size());
    for (unsigned char c : archive.pipeline_identity) w.integer(c);
    w.integer(archive.sessions.size());
    for (const auto& session : archive.sessions) {
        (void)restore_model(session); // Validate before any filesystem side effect.
        write_options(w, session.options, session.gate);
        const auto& labels = composition_labels(session.options.task);
        w.integer(labels.size());
        for (auto label : labels) w.integer(std::uint64_t(label));
        w.integer(session.trained);
        for (float value : session.weights.input) w.number(value);
        for (float value : session.weights.recurrent) w.number(value);
        for (float value : session.weights.bias) w.number(value);
        for (float value : session.weights.readout) w.number(value);
        write_clips(w, session.training); write_clips(w, session.heldout);
    }
    const auto hash = digest(w.bytes.data(), w.bytes.size());
    w.bytes.insert(w.bytes.end(), hash.begin(), hash.end());
    return std::move(w.bytes);
}
std::unique_ptr<CompositionEsn> restore_model(const SessionArchive& saved) {
    auto model = std::make_unique<CompositionEsn>(saved.options);
    (void)ActionGate(saved.gate);
    if (saved.training.size() > saved.options.maximum_clips || saved.heldout.size() > saved.options.maximum_clips)
        throw std::invalid_argument("saved clip capacity exceeded");
    model->restore_weights(saved.weights);
    for (const auto& clip : saved.training) model->add_training(clip.samples, clip.label);
    for (const auto& clip : saved.heldout) {
        (void)composition_index(saved.options.task, clip.label);
        (void)model->encode(clip.samples);
    }
    if (saved.trained) model->restore_trained_readout(saved.weights);
    return model;
}
ExperimentArchive decode_experiment(const std::vector<std::uint8_t>& bytes, const std::string& expected_pipeline) {
    if (bytes.size() < kDigestBytes || bytes.size() > kMaximumExperimentBytes)
        throw std::invalid_argument("experiment file size outside bounds");
    const auto end = bytes.size()-kDigestBytes;
    const auto hash = digest(bytes.data(), end);
    if (!std::equal(hash.begin(), hash.end(), bytes.begin()+end)) throw std::invalid_argument("experiment checksum mismatch");
    Reader r{bytes, 0, end};
    if (r.integer() != kMagic || r.integer() != kFormatVersion || r.integer() != kFeatureVersion)
        throw std::invalid_argument("unsupported experiment format or feature version");
    if (r.integer() != kCompositionInputs || r.integer() != kMotionInputs || r.integer() != kCompositionFeatures)
        throw std::invalid_argument("experiment feature dimensions mismatch");
    ExperimentArchive archive;
    const auto identity_size = r.bounded(kMaximumIdentity);
    for (std::uint64_t i = 0; i < identity_size; ++i) archive.pipeline_identity += char(r.bounded(255));
    if (archive.pipeline_identity != expected_pipeline) throw std::invalid_argument("MediaPipe pipeline identity mismatch");
    const auto branches = r.bounded(2);
    for (std::uint64_t i = 0; i < branches; ++i) {
        SessionArchive session;
        read_options(r, session.options, session.gate);
        const auto& labels = composition_labels(session.options.task);
        if (r.integer() != labels.size()) throw std::invalid_argument("experiment class count mismatch");
        for (auto label : labels)
            if (r.integer() != std::uint64_t(label)) throw std::invalid_argument("experiment label order mismatch");
        session.trained = r.bounded(1) != 0;
        for (float& value : session.weights.input) value = r.scalar();
        for (float& value : session.weights.recurrent) value = r.scalar();
        for (float& value : session.weights.bias) value = r.scalar();
        for (float& value : session.weights.readout) value = r.scalar();
        session.training = read_clips(r, session.options); session.heldout = read_clips(r, session.options);
        (void)restore_model(session);
        archive.sessions.push_back(std::move(session));
    }
    if (r.position != end) throw std::invalid_argument("unexpected experiment trailing data");
    check_layout(archive);
    return archive;
}
} // namespace kfcore::gesture_interaction
