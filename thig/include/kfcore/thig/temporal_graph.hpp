#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace kfcore::thig {

enum class TruthValue { False, True, Unknown };
enum class EvidenceType { State, Transition, Trend };

struct EntityRef {
  std::string kind;
  std::int64_t id = 0;

  friend bool operator==(const EntityRef& left, const EntityRef& right) {
    return left.kind == right.kind && left.id == right.id;
  }
  friend bool operator!=(const EntityRef& left, const EntityRef& right) {
    return !(left == right);
  }
};

struct Observation {
  std::uint64_t serial = 0;
  EntityRef source;
  std::optional<EntityRef> target;
  std::string relation;
  TruthValue truth = TruthValue::True;
  EvidenceType evidenceType = EvidenceType::State;
  float confidence = 0.0f;
  std::string producer;
  std::string diagnostic;
};

struct RelationSpec {
  std::string relation;
  std::string exclusiveGroup;
};

// Stabilizes raw mutually-exclusive observations before they enter the
// relation graph. Samples are owned by the single TemporalGraphEngine owner
// and are bounded by both time and count.
struct ObservationWindowSpec {
  std::string exclusiveGroup;
  int windowMs = 0;
  int minimumSupportingObservations = 1;
  float minimumSupportRatio = 1.0f;
  float switchMargin = 0.0f;
  std::size_t maxSamplesPerSource = 32;
  std::vector<std::string> ignoredRelations;
  // Some exclusive groups represent real transitions rather than noisy
  // categorical states. When enabled, an observed competing relation ends
  // the previous stable relation immediately, while the competing relation
  // must still satisfy this window before it becomes stable.
  bool rejectStableOnConflict = false;
};

struct RelationEvent {
  std::uint64_t eventId = 0;
  std::string relation;
  EntityRef source;
  std::optional<EntityRef> target;
  std::uint64_t startMs = 0;
  std::uint64_t endMs = 0;
  std::uint64_t lastObservedMs = 0;
  EvidenceType evidenceType = EvidenceType::State;
  float confidence = 0.0f;
  float support = 0.0f;
  std::uint64_t firstSourceSerial = 0;
  std::uint64_t lastSourceSerial = 0;
  std::uint64_t rawFrames = 0;
  std::uint64_t observedFrames = 0;
  std::uint64_t supportFrames = 0;
  std::uint64_t unknownFrames = 0;
  bool active = false;
  std::string producer;
  std::string diagnostic;
  std::string specVersion;
};

enum class PatternOperator {
  Atom,
  Seq,
  Both,
  During,
  Within,
  OnsetWithin,
  Repeat,
};

// Declares how a binary pattern joins entity-bound evidence. Same preserves
// the historic single-entity behavior; Distinct uses two different sources and
// exposes the second participant as ActionEvent::target. A concurrent Both
// requires current evidence from both distinct sources.
enum class PatternSourceJoin { Same, Distinct };

struct PatternNode {
  std::string id;
  PatternOperator operation = PatternOperator::Atom;
  std::vector<std::size_t> inputs;
  std::string relation;
  int minDurationMs = 0;
  int windowMs = 0;
  int repeatCount = 0;
  PatternSourceJoin sourceJoin = PatternSourceJoin::Same;
  // A positive value requires Both inputs to share this much real interval
  // overlap. Zero preserves boundary-touch compatibility for existing specs.
  int minOverlapMs = 0;
};

struct PatternGraph {
  std::vector<PatternNode> nodes;
  std::size_t root = 0;
  // Stable relations in this list invalidate a match when their intervals
  // overlap the matched interval for either bound participant.
  std::vector<std::string> forbiddenRelations;

  static PatternGraph Atom(std::string relation, int minDurationMs = 0);
};

struct ActionSpec {
  std::string action;
  PatternGraph pattern;
  std::string exclusiveGroup;
  int priority = 0;
  int cooldownMs = 0;
};

enum class SourceConstraint {
  Any,
  Bound,
  BoundIfPresent,
};

struct StateTransitionSpec {
  std::string id;
  std::string fromState;
  std::string toState;
  PatternGraph trigger;
  SourceConstraint sourceConstraint = SourceConstraint::BoundIfPresent;
  bool bindSource = false;
  bool unbindSource = false;
  std::string action;
  int priority = 0;
  // Declarative cooldown/phase guard measured from entry into fromState.
  int minStateDurationMs = 0;
  // Evidence for these trigger relations must begin no earlier than entry
  // into fromState. Other state evidence may span the boundary.
  std::vector<std::string> stateLocalRelations;
};

struct StateGraphSpec {
  std::string id;
  std::string initialState;
  std::vector<std::string> states;
  std::vector<std::string> bindOnRelations;
  int bindingTimeoutMs = 0;
  std::string bindingTimeoutRelation;
  std::vector<StateTransitionSpec> transitions;
};

struct EngineSpec {
  std::string version;
  int historyMs = 5000;
  int observationMaxGapMs = 350;
  std::size_t maxObservationsPerFrame = 64;
  std::size_t maxRelationEvents = 512;
  std::size_t maxObservationWindowStates = 64;
  std::size_t maxActionStates = 4096;
  std::vector<RelationSpec> relations;
  std::vector<ObservationWindowSpec> observationWindows;
  std::vector<ActionSpec> actions;
  std::vector<StateGraphSpec> stateGraphs;
};

struct ActionEvent {
  std::string action;
  EntityRef source;
  std::optional<EntityRef> target;
  std::uint64_t startMs = 0;
  std::uint64_t confirmedMs = 0;
  std::uint64_t endMs = 0;
  float confidence = 0.0f;
  float support = 0.0f;
  std::vector<RelationEvent> evidence;
  std::string specVersion;
};

// Single-owner graph interpreter. All methods must be called from the same
// executor unless the caller provides external serialization.
class TemporalGraphEngine {
public:
  TemporalGraphEngine();
  explicit TemporalGraphEngine(EngineSpec spec);
  ~TemporalGraphEngine();

  TemporalGraphEngine(const TemporalGraphEngine&) = delete;
  TemporalGraphEngine& operator=(const TemporalGraphEngine&) = delete;
  TemporalGraphEngine(TemporalGraphEngine&&) noexcept;
  TemporalGraphEngine& operator=(TemporalGraphEngine&&) noexcept;

  // Validates the complete graph before replacing the active specification.
  // Throws std::invalid_argument when the graph contract is inconsistent.
  void Configure(EngineSpec spec);

  std::vector<ActionEvent> ProcessFrame(
      const std::vector<Observation>& observations,
      std::chrono::steady_clock::time_point now);

  void Reset();

  [[nodiscard]] const EngineSpec& Spec() const;
  [[nodiscard]] const std::vector<RelationEvent>& Relations() const;
  [[nodiscard]] std::size_t ActionStateCount() const;
  [[nodiscard]] std::string StateOf(const std::string& graphId) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kfcore::thig
