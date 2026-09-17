#include "kfcore/thig/temporal_graph.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kfcore::thig {
namespace {

constexpr std::size_t kMaximumMatchesPerNode = 64;
constexpr char kDefaultSpecVersion[] = "thig-1.0";
constexpr char kKeySeparator = '\x1f';

std::uint64_t ToMilliseconds(std::chrono::steady_clock::time_point time) {
  if (time == std::chrono::steady_clock::time_point{}) {
    return 0;
  }
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count());
}

std::string EntityKey(const EntityRef& entity) {
  return entity.kind + ":" + std::to_string(entity.id);
}

std::string RelationKey(const Observation& observation) {
  std::string key = EntityKey(observation.source) + "\x1f" + observation.relation;
  if (observation.target.has_value()) {
    key += "\x1f" + EntityKey(*observation.target);
  }
  return key;
}

std::string ExclusiveScopeKey(const EntityRef& source,
                              const std::optional<EntityRef>& target,
                              const std::string& exclusiveGroup) {
  std::string key = EntityKey(source) + kKeySeparator;
  if (target.has_value()) {
    key += EntityKey(*target);
  }
  return key + kKeySeparator + exclusiveGroup;
}

bool Contains(const std::vector<std::string>& values, const std::string& value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

void ValidatePattern(const PatternGraph& graph, const std::string& owner) {
  if (graph.nodes.empty()) {
    throw std::invalid_argument(owner + " has an empty pattern graph");
  }
  if (graph.root >= graph.nodes.size()) {
    throw std::invalid_argument(owner + " has an invalid pattern root");
  }

  std::vector<int> visit(graph.nodes.size(), 0);
  std::function<void(std::size_t)> validateNode = [&](std::size_t index) {
    if (index >= graph.nodes.size()) {
      throw std::invalid_argument(owner + " references a missing pattern node");
    }
    if (visit[index] == 1) {
      throw std::invalid_argument(owner + " contains a pattern cycle");
    }
    if (visit[index] == 2) {
      return;
    }
    visit[index] = 1;
    const auto& node = graph.nodes[index];
    const auto requireInputs = [&](std::size_t expected) {
      if (node.inputs.size() != expected) {
        throw std::invalid_argument(owner + " node '" + node.id +
                                    "' has invalid input arity");
      }
    };
    switch (node.operation) {
      case PatternOperator::Atom:
        requireInputs(0);
        if (node.relation.empty()) {
          throw std::invalid_argument(owner + " contains an atom without a relation");
        }
        break;
      case PatternOperator::Seq:
      case PatternOperator::Both:
      case PatternOperator::During:
      case PatternOperator::Within:
      case PatternOperator::OnsetWithin:
        requireInputs(2);
        break;
      case PatternOperator::Repeat:
        requireInputs(1);
        if (node.repeatCount < 2) {
          throw std::invalid_argument(owner + " repeat count must be at least two");
        }
        if (node.sourceJoin != PatternSourceJoin::Same) {
          throw std::invalid_argument(owner +
                                      " repeat nodes require same-source evidence");
        }
        break;
    }
    if (node.minDurationMs < 0 || node.windowMs < 0 ||
        node.minOverlapMs < 0) {
      throw std::invalid_argument(owner + " contains a negative duration");
    }
    if (node.minOverlapMs > 0 && node.operation != PatternOperator::Both) {
      throw std::invalid_argument(owner + " node '" + node.id +
                                  "' sets overlap on a non-Both operator");
    }
    for (const auto input : node.inputs) {
      validateNode(input);
    }
    visit[index] = 2;
  };
  validateNode(graph.root);
}

void ValidateSpec(const EngineSpec& spec) {
  if (spec.historyMs <= 0) {
    throw std::invalid_argument("THIG historyMs must be positive");
  }
  if (spec.observationMaxGapMs < 0) {
    throw std::invalid_argument("THIG observationMaxGapMs cannot be negative");
  }
  if (spec.maxObservationsPerFrame == 0 || spec.maxRelationEvents == 0 ||
      spec.maxObservationWindowStates == 0 || spec.maxActionStates == 0) {
    throw std::invalid_argument("THIG graph capacity limits must be positive");
  }
  if (!spec.actions.empty() &&
      spec.maxRelationEvents > spec.maxActionStates / spec.actions.size()) {
    throw std::invalid_argument(
        "THIG maxActionStates must cover the relation/action bound");
  }

  std::unordered_set<std::string> relations;
  std::unordered_map<std::string, std::string> relationGroups;
  for (const auto& relation : spec.relations) {
    if (relation.relation.empty() ||
        relation.relation.find(kKeySeparator) != std::string::npos ||
        !relations.insert(relation.relation).second) {
      throw std::invalid_argument("THIG relation names must be non-empty and unique");
    }
    relationGroups.emplace(relation.relation, relation.exclusiveGroup);
  }

  std::unordered_set<std::string> stabilizedGroups;
  for (const auto& window : spec.observationWindows) {
    if (window.exclusiveGroup.empty() ||
        !stabilizedGroups.insert(window.exclusiveGroup).second) {
      throw std::invalid_argument(
          "THIG observation window groups must be non-empty and unique");
    }
    if (window.windowMs <= 0 || window.minimumSupportingObservations <= 0 ||
        window.maxSamplesPerSource == 0 ||
        static_cast<std::size_t>(window.minimumSupportingObservations) >
            window.maxSamplesPerSource ||
        !std::isfinite(window.minimumSupportRatio) ||
        window.minimumSupportRatio <= 0.0f ||
        window.minimumSupportRatio > 1.0f ||
        !std::isfinite(window.switchMargin) || window.switchMargin < 0.0f ||
        window.switchMargin >= 1.0f) {
      throw std::invalid_argument(
          "THIG observation window has invalid bounds or thresholds");
    }
    const bool knownGroup = std::any_of(
        spec.relations.begin(), spec.relations.end(), [&](const auto& relation) {
          return relation.exclusiveGroup == window.exclusiveGroup;
        });
    if (!knownGroup) {
      throw std::invalid_argument(
          "THIG observation window references an unknown exclusive group '" +
          window.exclusiveGroup + "'");
    }
    std::unordered_set<std::string> ignored;
    for (const auto& relation : window.ignoredRelations) {
      const auto group = relationGroups.find(relation);
      if (group == relationGroups.end() ||
          group->second != window.exclusiveGroup ||
          !ignored.insert(relation).second) {
        throw std::invalid_argument(
            "THIG observation window contains an invalid ignored relation '" +
            relation + "'");
      }
    }
  }
  const auto validateKnownAtoms = [&](const PatternGraph& pattern,
                                      const std::string& owner) {
    ValidatePattern(pattern, owner);
    std::unordered_set<std::string> required;
    for (const auto& node : pattern.nodes) {
      if (node.operation == PatternOperator::Atom &&
          relations.find(node.relation) == relations.end()) {
        throw std::invalid_argument(owner + " references unknown relation '" +
                                    node.relation + "'");
      }
      if (node.operation == PatternOperator::Atom) {
        required.insert(node.relation);
      }
    }
    std::unordered_set<std::string> forbidden;
    for (const auto& relation : pattern.forbiddenRelations) {
      if (relations.find(relation) == relations.end() ||
          required.find(relation) != required.end() ||
          !forbidden.insert(relation).second) {
        throw std::invalid_argument(
            owner + " contains an invalid forbidden relation '" + relation +
            "'");
      }
    }
  };

  std::unordered_set<std::string> actions;
  for (const auto& action : spec.actions) {
    if (action.action.empty() || !actions.insert(action.action).second) {
      throw std::invalid_argument("THIG action names must be non-empty and unique");
    }
    if (action.cooldownMs < 0) {
      throw std::invalid_argument("THIG action cooldown cannot be negative");
    }
    validateKnownAtoms(action.pattern, "action '" + action.action + "'");
  }

  std::unordered_set<std::string> graphIds;
  for (const auto& graph : spec.stateGraphs) {
    if (graph.id.empty() || !graphIds.insert(graph.id).second) {
      throw std::invalid_argument("THIG state graph ids must be non-empty and unique");
    }
    std::unordered_set<std::string> states(graph.states.begin(), graph.states.end());
    if (states.size() != graph.states.size() ||
        states.find(graph.initialState) == states.end()) {
      throw std::invalid_argument("THIG state graph '" + graph.id +
                                  "' has invalid states or initial state");
    }
    if (graph.bindingTimeoutMs < 0) {
      throw std::invalid_argument("THIG binding timeout cannot be negative");
    }
    if (!graph.bindingTimeoutRelation.empty() &&
        relations.find(graph.bindingTimeoutRelation) == relations.end()) {
      throw std::invalid_argument("THIG state graph '" + graph.id +
                                  "' uses an unknown timeout relation");
    }
    for (const auto& relation : graph.bindOnRelations) {
      if (relations.find(relation) == relations.end()) {
        throw std::invalid_argument("THIG state graph '" + graph.id +
                                    "' binds an unknown relation");
      }
    }
    std::unordered_set<std::string> transitionIds;
    for (const auto& transition : graph.transitions) {
      if (transition.id.empty() || !transitionIds.insert(transition.id).second ||
          states.find(transition.fromState) == states.end() ||
          states.find(transition.toState) == states.end()) {
        throw std::invalid_argument("THIG state graph '" + graph.id +
                                    "' has an invalid transition");
      }
      if (transition.minStateDurationMs < 0) {
        throw std::invalid_argument("THIG state graph '" + graph.id +
                                    "' has a negative state duration");
      }
      validateKnownAtoms(transition.trigger,
                         "transition '" + graph.id + "." + transition.id + "'");
      std::unordered_set<std::string> stateLocalRelations;
      for (const auto& relation : transition.stateLocalRelations) {
        const bool isTriggerAtom = std::any_of(
            transition.trigger.nodes.begin(), transition.trigger.nodes.end(),
            [&](const PatternNode& node) {
              return node.operation == PatternOperator::Atom &&
                     node.relation == relation;
            });
        if (!stateLocalRelations.insert(relation).second || !isTriggerAtom) {
          throw std::invalid_argument(
              "THIG transition '" + graph.id + "." + transition.id +
              "' has an invalid state-local relation");
        }
      }
    }
  }
}

struct PatternMatch {
  EntityRef source;
  std::optional<EntityRef> target;
  std::uint64_t startMs = 0;
  std::uint64_t endMs = 0;
  std::uint64_t latestObservedMs = 0;
  float confidence = 0.0f;
  float support = 0.0f;
  std::vector<std::size_t> evidenceIndices;
};

bool CompatibleTargets(const PatternMatch& left, const PatternMatch& right) {
  return !left.target.has_value() || !right.target.has_value() ||
         *left.target == *right.target;
}

PatternMatch MergeMatches(const PatternMatch& left, const PatternMatch& right) {
  PatternMatch merged;
  merged.source = left.source;
  merged.target = left.target.has_value() ? left.target : right.target;
  merged.startMs = std::min(left.startMs, right.startMs);
  merged.endMs = std::max(left.endMs, right.endMs);
  merged.latestObservedMs = std::max(left.latestObservedMs, right.latestObservedMs);
  merged.confidence = std::min(left.confidence, right.confidence);
  merged.support = std::min(left.support, right.support);
  merged.evidenceIndices = left.evidenceIndices;
  for (const auto index : right.evidenceIndices) {
    if (std::find(merged.evidenceIndices.begin(), merged.evidenceIndices.end(), index) ==
        merged.evidenceIndices.end()) {
      merged.evidenceIndices.push_back(index);
    }
  }
  return merged;
}

PatternMatch MergeConcurrentMatches(const PatternMatch& left,
                                    const PatternMatch& right) {
  PatternMatch merged = MergeMatches(left, right);
  merged.startMs = std::max(left.startMs, right.startMs);
  merged.endMs = std::min(left.endMs, right.endMs);
  // Currentness requires an overlap that still reaches a newly observed
  // operand. This permits UNKNOWN on one active side, but rejects both an old
  // overlap revived by newer unrelated evidence and an all-UNKNOWN match.
  merged.latestObservedMs = std::min(
      merged.endMs, std::max(left.latestObservedMs, right.latestObservedMs));
  return merged;
}

PatternMatch MergeDuringMatches(const PatternMatch& contained,
                                const PatternMatch& container) {
  PatternMatch merged = MergeMatches(contained, container);
  merged.startMs = contained.startMs;
  merged.endMs = contained.endMs;
  return merged;
}

PatternMatch MergeDistinctMatches(const PatternMatch& left,
                                  const PatternMatch& right,
                                  bool concurrent) {
  PatternMatch merged = concurrent ? MergeConcurrentMatches(left, right)
                                   : MergeMatches(left, right);
  if (concurrent) {
    merged.latestObservedMs = std::min(
        {merged.endMs, left.latestObservedMs, right.latestObservedMs});
  }
  if (EntityKey(right.source) < EntityKey(left.source)) {
    merged.source = right.source;
    merged.target = left.source;
  } else {
    merged.source = left.source;
    merged.target = right.source;
  }
  return merged;
}

std::string MatchFingerprint(const PatternMatch& match,
                             const std::vector<RelationEvent>& relations) {
  std::vector<std::uint64_t> ids;
  ids.reserve(match.evidenceIndices.size());
  for (const auto index : match.evidenceIndices) {
    ids.push_back(relations[index].eventId);
  }
  std::sort(ids.begin(), ids.end());
  std::string fingerprint = EntityKey(match.source);
  for (const auto id : ids) {
    fingerprint += ":" + std::to_string(id);
  }
  return fingerprint;
}

struct EffectiveObservation {
  Observation observation;
  std::uint64_t intervalStartMs = 0;
};

class TemporalStabilizer {
public:
  void Configure(const EngineSpec& spec) {
    relationGroups_.clear();
    windows_.clear();
    for (const auto& relation : spec.relations) {
      relationGroups_.emplace(relation.relation, relation.exclusiveGroup);
    }
    for (const auto& window : spec.observationWindows) {
      windows_.emplace(window.exclusiveGroup, window);
    }
    maximumStates_ = spec.maxObservationWindowStates;
    Reset();
  }

  void Reset() { states_.clear(); }

  std::vector<EffectiveObservation> ProcessFrame(
      const std::vector<Observation>& observations, std::uint64_t nowMs) {
    Prune(nowMs);

    std::vector<EffectiveObservation> output;
    output.reserve(observations.size());
    std::unordered_map<std::string, std::vector<const Observation*>> grouped;
    std::unordered_map<std::string, std::string> groupByKey;

    for (const auto& observation : observations) {
      const auto relationGroup = relationGroups_.find(observation.relation);
      const auto window =
          relationGroup == relationGroups_.end()
              ? windows_.end()
              : windows_.find(relationGroup->second);
      if (window == windows_.end()) {
        output.push_back({observation, nowMs});
        continue;
      }

      const std::string key = StateKey(observation.source, window->first);
      groupByKey.emplace(key, window->first);
      if (observation.truth == TruthValue::False) {
        RejectRelation(key, observation.relation);
        output.push_back({observation, nowMs});
        continue;
      }
      if (observation.truth == TruthValue::Unknown ||
          Contains(window->second.ignoredRelations, observation.relation)) {
        const auto state = states_.find(key);
        if (state != states_.end()) {
          state->second.lastInputMs = nowMs;
        }
        continue;
      }
      grouped[key].push_back(&observation);
    }

    std::size_t newStates = 0;
    for (const auto& [key, values] : grouped) {
      (void)values;
      if (states_.find(key) == states_.end()) {
        ++newStates;
      }
    }
    if (newStates > maximumStates_ - std::min(maximumStates_, states_.size())) {
      throw std::length_error("THIG observation window state budget exhausted");
    }

    for (const auto& [key, current] : grouped) {
      const auto& window = windows_.at(groupByKey.at(key));
      if (current.size() != 1) {
        const auto state = states_.find(key);
        if (state != states_.end()) {
          state->second.lastInputMs = nowMs;
        }
        continue;
      }

      auto& state = states_[key];
      state.exclusiveGroup = window.exclusiveGroup;
      state.lastInputMs = nowMs;
      state.samples.push_back({nowMs, *current.front()});
      while (state.samples.size() > window.maxSamplesPerSource) {
        state.samples.pop_front();
      }

      const std::string previousStableRelation = state.stableRelation;
      const bool conflictsWithStable =
          window.rejectStableOnConflict && !previousStableRelation.empty() &&
          current.front()->relation != previousStableRelation;
      if (conflictsWithStable) {
        Observation rejection = *current.front();
        rejection.relation = previousStableRelation;
        rejection.truth = TruthValue::False;
        rejection.diagnostic =
            "stabilizer rejected stable relation after observing " +
            current.front()->relation;
        output.push_back({std::move(rejection), nowMs});
      }

      const auto candidate = SelectCandidate(state, window, nowMs);
      if (candidate.has_value() &&
          (state.stableRelation.empty() ||
           state.stableRelation != candidate->relation)) {
        state.stableRelation = candidate->relation;
      }

      if (state.stableRelation.empty() ||
          current.front()->relation != state.stableRelation) {
        continue;
      }
      const auto intervalStart = candidate.has_value() &&
                                         candidate->relation == state.stableRelation
                                     ? candidate->firstMs
                                     : nowMs;
      output.push_back({*current.front(), intervalStart});
    }
    return output;
  }

  struct WindowSample {
    std::uint64_t observedMs = 0;
    Observation observation;
  };

  struct WindowState {
    std::string exclusiveGroup;
    std::string stableRelation;
    std::uint64_t lastInputMs = 0;
    std::deque<WindowSample> samples;
  };

  struct Candidate {
    std::string relation;
    std::uint64_t firstMs = 0;
  };

  struct CandidateStats {
    std::string relation;
    std::size_t count = 0;
    float confidenceSum = 0.0f;
    std::uint64_t firstMs = 0;
    std::uint64_t latestMs = 0;
  };

  static std::string StateKey(const EntityRef& source,
                              const std::string& group) {
    return EntityKey(source) + kKeySeparator + group;
  }

  void Prune(std::uint64_t nowMs) {
    for (auto iterator = states_.begin(); iterator != states_.end();) {
      auto& state = iterator->second;
      const auto& window = windows_.at(state.exclusiveGroup);
      const auto cutoff =
          nowMs > static_cast<std::uint64_t>(window.windowMs)
              ? nowMs - static_cast<std::uint64_t>(window.windowMs)
              : 0;
      while (!state.samples.empty() &&
             state.samples.front().observedMs < cutoff) {
        state.samples.pop_front();
      }
      if (state.samples.empty() &&
          nowMs - state.lastInputMs >
              static_cast<std::uint64_t>(window.windowMs)) {
        iterator = states_.erase(iterator);
      } else {
        ++iterator;
      }
    }
  }

  void RejectRelation(const std::string& key, const std::string& relation) {
    const auto iterator = states_.find(key);
    if (iterator == states_.end()) {
      return;
    }
    auto& state = iterator->second;
    state.samples.erase(
        std::remove_if(state.samples.begin(), state.samples.end(),
                       [&](const auto& sample) {
                         return sample.observation.relation == relation;
                       }),
        state.samples.end());
    if (state.stableRelation == relation) {
      state.stableRelation.clear();
    }
  }

  static std::optional<Candidate> SelectCandidate(
      const WindowState& state, const ObservationWindowSpec& window,
      std::uint64_t nowMs) {
    std::unordered_map<std::string, CandidateStats> byRelation;
    for (const auto& sample : state.samples) {
      auto& stats = byRelation[sample.observation.relation];
      if (stats.count == 0) {
        stats.relation = sample.observation.relation;
        stats.firstMs = sample.observedMs;
      }
      ++stats.count;
      stats.confidenceSum += sample.observation.confidence;
      stats.latestMs = sample.observedMs;
    }
    if (byRelation.empty()) {
      return std::nullopt;
    }

    std::vector<CandidateStats> candidates;
    candidates.reserve(byRelation.size());
    for (auto& [relation, stats] : byRelation) {
      (void)relation;
      candidates.push_back(std::move(stats));
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const auto& left, const auto& right) {
                       if (left.count != right.count) {
                         return left.count > right.count;
                       }
                       const float leftMean =
                           left.confidenceSum / static_cast<float>(left.count);
                       const float rightMean =
                           right.confidenceSum / static_cast<float>(right.count);
                       return leftMean == rightMean
                                  ? left.relation < right.relation
                                  : leftMean > rightMean;
                     });

    const auto& winner = candidates.front();
    const float total = static_cast<float>(state.samples.size());
    const float winnerSupport = static_cast<float>(winner.count) / total;
    const float runnerSupport =
        candidates.size() > 1
            ? static_cast<float>(candidates[1].count) / total
            : 0.0f;
    const bool separated =
        candidates.size() == 1 ||
        winnerSupport - runnerSupport > window.switchMargin;
    if (winner.count <
            static_cast<std::size_t>(window.minimumSupportingObservations) ||
        winnerSupport < window.minimumSupportRatio || !separated ||
        winner.latestMs != nowMs) {
      return std::nullopt;
    }
    return Candidate{winner.relation, winner.firstMs};
  }

  std::unordered_map<std::string, std::string> relationGroups_;
  std::unordered_map<std::string, ObservationWindowSpec> windows_;
  std::unordered_map<std::string, WindowState> states_;
  std::size_t maximumStates_ = 0;
};

}  // namespace

PatternGraph PatternGraph::Atom(std::string relation, int minDurationMs) {
  PatternGraph graph;
  PatternNode atom;
  atom.id = "atom";
  atom.operation = PatternOperator::Atom;
  atom.relation = std::move(relation);
  atom.minDurationMs = minDurationMs;
  graph.nodes.push_back(std::move(atom));
  graph.root = 0;
  return graph;
}

class TemporalGraphEngine::Impl {
public:
  struct StateRuntime {
    std::string state;
    std::uint64_t stateEnteredMs = 0;
    std::optional<EntityRef> boundSource;
    std::uint64_t boundSourceLastObservedMs = 0;
    std::unordered_map<std::string, std::string> lastFingerprintByTransition;
  };

  void Configure(EngineSpec spec) {
    ValidateSpec(spec);
    if (spec.version.empty()) {
      spec.version = kDefaultSpecVersion;
    }
    spec_ = std::move(spec);
    relationSpecs_.clear();
    for (const auto& relation : spec_.relations) {
      relationSpecs_.emplace(relation.relation, relation);
    }
    stabilizer_.Configure(spec_);
    configured_ = true;
    Reset();
  }

  void Reset() {
    relations_.clear();
    activeRelationByKey_.clear();
    stabilizer_.Reset();
    actionStates_.clear();
    stateRuntimes_.clear();
    nextEventId_ = 1;
    lastProcessMs_ = 0;
    hasProcessed_ = false;
    for (const auto& graph : spec_.stateGraphs) {
      StateRuntime runtime;
      runtime.state = graph.initialState;
      stateRuntimes_.emplace(graph.id, std::move(runtime));
    }
  }

  std::vector<ActionEvent> ProcessFrame(const std::vector<Observation>& input,
                                        std::uint64_t nowMs) {
    if (!configured_) {
      throw std::logic_error("THIG engine must be configured before processing");
    }
    if (hasProcessed_ && nowMs < lastProcessMs_) {
      throw std::invalid_argument("THIG timestamps must be monotonic");
    }

    std::vector<Observation> observations = input;
    ValidateObservations(observations);
    InsertBindingTimeoutObservations(observations, nowMs);
    ValidateObservations(observations);
    // Stabilization is staged so a later graph-capacity rejection cannot
    // consume temporal-window history from a frame the engine did not accept.
    auto stagedStabilizer = stabilizer_;
    auto effectiveObservations =
        stagedStabilizer.ProcessFrame(observations, nowMs);
    EnsureRelationCapacity(effectiveObservations, nowMs);
    PruneHistory(nowMs);
    stabilizer_ = std::move(stagedStabilizer);
    lastProcessMs_ = nowMs;
    hasProcessed_ = true;
    UpdateRelationGraph(effectiveObservations, nowMs);
    UpdateBindingsFromObservations(effectiveObservations, nowMs);

    std::vector<ActionEvent> output;
    MatchActions(nowMs, output);
    MatchStateGraphs(nowMs, output);
    Arbitrate(output);
    PruneHistory(nowMs);
    return output;
  }

  const EngineSpec& Spec() const { return spec_; }
  const std::vector<RelationEvent>& Relations() const { return relations_; }
  std::size_t ActionStateCount() const { return actionStates_.size(); }

  std::string StateOf(const std::string& graphId) const {
    const auto iterator = stateRuntimes_.find(graphId);
    return iterator == stateRuntimes_.end() ? std::string{} : iterator->second.state;
  }

private:
  void ValidateObservations(const std::vector<Observation>& observations) const {
    if (observations.size() > spec_.maxObservationsPerFrame) {
      throw std::length_error("THIG frame exceeds maxObservationsPerFrame");
    }
    std::unordered_set<std::string> keys;
    std::unordered_set<std::string> exclusiveKeys;
    for (const auto& observation : observations) {
      const auto relationSpec = relationSpecs_.find(observation.relation);
      if (relationSpec == relationSpecs_.end()) {
        throw std::invalid_argument(
            "THIG observation references unknown relation '" +
            observation.relation + "'");
      }
      const std::string key = RelationKey(observation);
      if (!keys.insert(key).second) {
        throw std::invalid_argument(
            "THIG frame contains duplicate relation observation '" +
            observation.relation + "'");
      }
      if (observation.truth == TruthValue::True &&
          !relationSpec->second.exclusiveGroup.empty()) {
        std::string exclusiveKey = ExclusiveScopeKey(
            observation.source, observation.target,
            relationSpec->second.exclusiveGroup);
        if (!exclusiveKeys.insert(std::move(exclusiveKey)).second) {
          throw std::invalid_argument(
              "THIG frame contains conflicting true relations in exclusive "
              "group '" +
              relationSpec->second.exclusiveGroup + "'");
        }
      }
      if (!std::isfinite(observation.confidence) ||
          observation.confidence < 0.0f || observation.confidence > 1.0f) {
        throw std::invalid_argument("THIG confidence must be within [0, 1]");
      }
      if (observation.source.kind.empty() ||
          observation.source.kind.find(kKeySeparator) != std::string::npos ||
          (observation.target.has_value() &&
           (observation.target->kind.empty() ||
            observation.target->kind.find(kKeySeparator) !=
                std::string::npos))) {
        throw std::invalid_argument(
            "THIG entity kinds must be non-empty canonical keys");
      }
    }
  }

  void EnsureRelationCapacity(
      const std::vector<EffectiveObservation>& observations,
      std::uint64_t nowMs) const {
    const std::uint64_t cutoff =
        nowMs > static_cast<std::uint64_t>(spec_.historyMs)
            ? nowMs - static_cast<std::uint64_t>(spec_.historyMs)
            : 0;
    const std::size_t retainedEvents = static_cast<std::size_t>(std::count_if(
        relations_.begin(), relations_.end(), [&](const auto& relation) {
          return relation.active || relation.endMs >= cutoff;
        }));
    std::size_t newEvents = 0;
    for (const auto& effective : observations) {
      const auto& observation = effective.observation;
      if (observation.truth == TruthValue::True &&
          activeRelationByKey_.find(RelationKey(observation)) ==
              activeRelationByKey_.end()) {
        ++newEvents;
      }
    }
    if (retainedEvents + newEvents > spec_.maxRelationEvents) {
      throw std::length_error("THIG rolling graph exceeds maxRelationEvents");
    }
  }

  void InsertBindingTimeoutObservations(std::vector<Observation>& observations,
                                        std::uint64_t nowMs) {
    for (const auto& graph : spec_.stateGraphs) {
      auto& runtime = stateRuntimes_.at(graph.id);
      if (!runtime.boundSource.has_value() || graph.bindingTimeoutMs <= 0 ||
          graph.bindingTimeoutRelation.empty() ||
          nowMs - runtime.boundSourceLastObservedMs <=
              static_cast<std::uint64_t>(graph.bindingTimeoutMs)) {
        continue;
      }
      Observation timeout;
      timeout.source = *runtime.boundSource;
      timeout.relation = graph.bindingTimeoutRelation;
      timeout.truth = TruthValue::True;
      timeout.evidenceType = EvidenceType::Transition;
      timeout.confidence = 1.0f;
      timeout.producer = "thig";
      timeout.diagnostic = "binding timeout in state graph " + graph.id;
      const auto duplicate = std::find_if(
          observations.begin(), observations.end(), [&](const auto& existing) {
            return RelationKey(existing) == RelationKey(timeout);
          });
      if (duplicate == observations.end()) {
        observations.push_back(std::move(timeout));
      }
    }
  }

  void UpdateRelationGraph(
      const std::vector<EffectiveObservation>& observations,
                           std::uint64_t nowMs) {
    std::unordered_map<std::string, const EffectiveObservation*> observedByKey;
    std::unordered_map<std::string, std::unordered_set<std::string>>
        trueRelationsByExclusiveScope;
    observedByKey.reserve(observations.size());
    for (const auto& effective : observations) {
      const auto& observation = effective.observation;
      observedByKey.emplace(RelationKey(observation), &effective);
      if (observation.truth == TruthValue::True) {
        const auto& relationSpec = relationSpecs_.at(observation.relation);
        if (!relationSpec.exclusiveGroup.empty()) {
          trueRelationsByExclusiveScope[ExclusiveScopeKey(
              observation.source, observation.target,
              relationSpec.exclusiveGroup)]
              .insert(observation.relation);
        }
      }
    }

    for (auto iterator = activeRelationByKey_.begin();
         iterator != activeRelationByKey_.end();) {
      auto& event = relations_[iterator->second];
      ++event.rawFrames;
      const auto observed = observedByKey.find(iterator->first);
      const auto& relationSpec = relationSpecs_.at(event.relation);
      bool conflict = false;
      if (!relationSpec.exclusiveGroup.empty()) {
        const auto scopeRelations = trueRelationsByExclusiveScope.find(
            ExclusiveScopeKey(event.source, event.target,
                              relationSpec.exclusiveGroup));
        if (scopeRelations != trueRelationsByExclusiveScope.end()) {
          for (const auto& relation : scopeRelations->second) {
            if (relation != event.relation) {
              conflict = true;
              break;
            }
          }
        }
      }
      const bool explicitlyFalse =
          observed != observedByKey.end() &&
          observed->second->observation.truth == TruthValue::False;
      const bool supporting =
          observed != observedByKey.end() &&
          observed->second->observation.truth == TruthValue::True;
      const bool expired =
          nowMs - event.lastObservedMs >
          static_cast<std::uint64_t>(spec_.observationMaxGapMs);

      if (supporting) {
        const auto& observation = observed->second->observation;
        ++event.observedFrames;
        ++event.supportFrames;
        event.confidence =
            ((event.confidence * static_cast<float>(event.supportFrames - 1)) +
             observation.confidence) /
            static_cast<float>(event.supportFrames);
        event.support = static_cast<float>(event.supportFrames) /
                        static_cast<float>(event.observedFrames);
        event.endMs = nowMs;
        event.lastObservedMs = nowMs;
        event.lastSourceSerial = observation.serial;
        event.producer = observation.producer;
        event.diagnostic = observation.diagnostic;
        ++iterator;
      } else if (explicitlyFalse || conflict || expired) {
        if (explicitlyFalse || conflict) {
          ++event.observedFrames;
          event.support = static_cast<float>(event.supportFrames) /
                          static_cast<float>(event.observedFrames);
        }
        event.active = false;
        event.endMs = event.lastObservedMs;
        iterator = activeRelationByKey_.erase(iterator);
      } else {
        ++event.unknownFrames;
        ++iterator;
      }
    }

    for (const auto& effective : observations) {
      const auto& observation = effective.observation;
      if (observation.truth != TruthValue::True) {
        continue;
      }
      const std::string key = RelationKey(observation);
      if (activeRelationByKey_.find(key) != activeRelationByKey_.end()) {
        continue;
      }
      RelationEvent event;
      event.eventId = nextEventId_++;
      event.relation = observation.relation;
      event.source = observation.source;
      event.target = observation.target;
      event.startMs = std::min(effective.intervalStartMs, nowMs);
      event.endMs = nowMs;
      event.lastObservedMs = nowMs;
      event.evidenceType = observation.evidenceType;
      event.confidence = observation.confidence;
      event.support = 1.0f;
      event.firstSourceSerial = observation.serial;
      event.lastSourceSerial = observation.serial;
      event.rawFrames = 1;
      event.observedFrames = 1;
      event.supportFrames = 1;
      event.active = true;
      event.producer = observation.producer;
      event.diagnostic = observation.diagnostic;
      event.specVersion = spec_.version;
      relations_.push_back(std::move(event));
      activeRelationByKey_[key] = relations_.size() - 1;
    }
  }

  void UpdateBindingsFromObservations(
      const std::vector<EffectiveObservation>& observations,
      std::uint64_t nowMs) {
    for (const auto& graph : spec_.stateGraphs) {
      auto& runtime = stateRuntimes_.at(graph.id);
      for (const auto& effective : observations) {
        const auto& observation = effective.observation;
        if (observation.truth != TruthValue::True ||
            observation.relation == graph.bindingTimeoutRelation) {
          continue;
        }
        if (runtime.boundSource.has_value() &&
            observation.source == *runtime.boundSource) {
          runtime.boundSourceLastObservedMs = nowMs;
        }
        if (!runtime.boundSource.has_value() &&
            Contains(graph.bindOnRelations, observation.relation)) {
          runtime.boundSource = observation.source;
          runtime.boundSourceLastObservedMs = nowMs;
          break;
        }
      }
    }
  }

  std::vector<PatternMatch> EvaluatePattern(const PatternGraph& graph,
                                            std::size_t nodeIndex,
                                            const EntityRef& source,
                                            std::uint64_t nowMs,
                                            const std::vector<std::string>*
                                                stateLocalRelations,
                                            std::uint64_t stateEnteredMs) const {
    const auto& node = graph.nodes[nodeIndex];
    if (node.operation == PatternOperator::Atom) {
      std::vector<PatternMatch> matches;
      for (std::size_t index = 0; index < relations_.size(); ++index) {
        const auto& relation = relations_[index];
        if (relation.relation != node.relation || relation.source != source) {
          continue;
        }
        if (stateLocalRelations != nullptr &&
            Contains(*stateLocalRelations, relation.relation) &&
            relation.startMs < stateEnteredMs) {
          continue;
        }
        const std::uint64_t effectiveEnd = relation.active ? nowMs : relation.endMs;
        if (effectiveEnd - relation.startMs <
            static_cast<std::uint64_t>(node.minDurationMs)) {
          continue;
        }
        PatternMatch match;
        match.source = relation.source;
        match.target = relation.target;
        match.startMs = relation.startMs;
        match.endMs = effectiveEnd;
        match.latestObservedMs = relation.lastObservedMs;
        match.confidence = relation.confidence;
        match.support = relation.support;
        match.evidenceIndices.push_back(index);
        matches.push_back(std::move(match));
      }
      return matches;
    }

    const auto left = EvaluatePattern(graph, node.inputs[0], source, nowMs,
                                      stateLocalRelations, stateEnteredMs);
    if (node.operation == PatternOperator::Repeat) {
      if (left.size() < static_cast<std::size_t>(node.repeatCount)) {
        return {};
      }
      std::vector<PatternMatch> ordered = left;
      std::sort(ordered.begin(), ordered.end(), [](const auto& first, const auto& second) {
        return first.startMs < second.startMs;
      });
      for (std::size_t offset = ordered.size() - node.repeatCount + 1; offset-- > 0;) {
        PatternMatch combined = ordered[offset];
        bool valid = true;
        for (int count = 1; count < node.repeatCount; ++count) {
          const auto& next = ordered[offset + static_cast<std::size_t>(count)];
          if (combined.endMs > next.startMs || !CompatibleTargets(combined, next)) {
            valid = false;
            break;
          }
          combined = MergeMatches(combined, next);
        }
        if (valid && (node.windowMs == 0 ||
                      combined.endMs - combined.startMs <=
                          static_cast<std::uint64_t>(node.windowMs))) {
          return {std::move(combined)};
        }
      }
      return {};
    }

    std::vector<PatternMatch> right;
    if (node.sourceJoin == PatternSourceJoin::Distinct) {
      for (const auto& candidateSource : KnownSources()) {
        if (candidateSource == source) {
          continue;
        }
        auto candidateMatches =
            EvaluatePattern(graph, node.inputs[1], candidateSource, nowMs,
                            stateLocalRelations, stateEnteredMs);
        right.insert(right.end(),
                     std::make_move_iterator(candidateMatches.begin()),
                     std::make_move_iterator(candidateMatches.end()));
      }
    } else {
      right = EvaluatePattern(graph, node.inputs[1], source, nowMs,
                              stateLocalRelations, stateEnteredMs);
    }
    std::vector<PatternMatch> output;
    for (const auto& first : left) {
      for (const auto& second : right) {
        const bool distinctSources = first.source != second.source;
        if (node.sourceJoin == PatternSourceJoin::Distinct) {
          if (!distinctSources || first.target.has_value() ||
              second.target.has_value()) {
            continue;
          }
        } else if (distinctSources || !CompatibleTargets(first, second)) {
          continue;
        }
        bool matches = false;
        switch (node.operation) {
          case PatternOperator::Seq:
            matches = first.endMs <= second.startMs &&
                      (node.windowMs == 0 ||
                       second.startMs - first.endMs <=
                           static_cast<std::uint64_t>(node.windowMs));
            break;
          case PatternOperator::Both:
            if (first.startMs <= second.endMs &&
                second.startMs <= first.endMs) {
              const auto overlapStart =
                  std::max(first.startMs, second.startMs);
              const auto overlapEnd = std::min(first.endMs, second.endMs);
              const auto overlapMs = overlapEnd - overlapStart;
              matches =
                  overlapMs >= static_cast<std::uint64_t>(node.minOverlapMs) &&
                  (node.windowMs == 0 ||
                   (std::max(first.startMs, second.startMs) -
                    std::min(first.startMs, second.startMs)) <=
                       static_cast<std::uint64_t>(node.windowMs));
            }
            break;
          case PatternOperator::During:
            matches = first.startMs >= second.startMs &&
                      first.endMs <= second.endMs;
            break;
          case PatternOperator::Within: {
            const auto start = std::min(first.startMs, second.startMs);
            const auto end = std::max(first.endMs, second.endMs);
            matches = end - start <= static_cast<std::uint64_t>(node.windowMs);
            break;
          }
          case PatternOperator::OnsetWithin:
            matches = first.startMs <= second.startMs &&
                      second.startMs - first.startMs <=
                          static_cast<std::uint64_t>(node.windowMs);
            break;
          case PatternOperator::Atom:
          case PatternOperator::Repeat:
            break;
        }
        if (matches) {
          if (node.sourceJoin == PatternSourceJoin::Distinct) {
            output.push_back(MergeDistinctMatches(
                first, second, node.operation == PatternOperator::Both));
          } else {
            output.push_back(
                node.operation == PatternOperator::Both
                    ? MergeConcurrentMatches(first, second)
                    : node.operation == PatternOperator::During
                          ? MergeDuringMatches(first, second)
                          : MergeMatches(first, second));
          }
          if (output.size() >= kMaximumMatchesPerNode) {
            return output;
          }
        }
      }
    }
    return output;
  }

  std::vector<PatternMatch> CurrentMatches(const PatternGraph& pattern,
                                           std::uint64_t nowMs,
                                           const std::vector<std::string>*
                                               stateLocalRelations = nullptr,
                                           std::uint64_t stateEnteredMs = 0) const {
    std::vector<PatternMatch> matches;
    for (const auto& source : KnownSources()) {
      auto sourceMatches = EvaluatePattern(pattern, pattern.root, source, nowMs,
                                           stateLocalRelations, stateEnteredMs);
      for (auto& match : sourceMatches) {
        if (match.latestObservedMs == nowMs &&
            !HasForbiddenRelation(pattern, match, nowMs)) {
          matches.push_back(std::move(match));
        }
      }
    }
    return matches;
  }

  bool HasForbiddenRelation(const PatternGraph& pattern,
                            const PatternMatch& match,
                            std::uint64_t nowMs) const {
    if (pattern.forbiddenRelations.empty()) {
      return false;
    }
    const auto participantMatches = [&](const EntityRef& source) {
      return source == match.source ||
             (match.target.has_value() && source == *match.target);
    };
    return std::any_of(relations_.begin(), relations_.end(),
                       [&](const auto& relation) {
      if (!participantMatches(relation.source) ||
          !Contains(pattern.forbiddenRelations, relation.relation)) {
        return false;
      }
      const auto relationEnd = relation.active ? nowMs : relation.endMs;
      return relation.startMs <= match.endMs && match.startMs <= relationEnd;
    });
  }

  std::vector<EntityRef> KnownSources() const {
    std::vector<EntityRef> sources;
    for (const auto& relation : relations_) {
      if (std::find(sources.begin(), sources.end(), relation.source) ==
          sources.end()) {
        sources.push_back(relation.source);
      }
    }
    return sources;
  }

  ActionEvent MakeActionEvent(const std::string& action, const PatternMatch& match,
                              std::uint64_t nowMs) const {
    ActionEvent event;
    event.action = action;
    event.source = match.source;
    event.target = match.target;
    event.startMs = match.startMs;
    event.confirmedMs = nowMs;
    event.endMs = match.endMs;
    event.confidence = match.confidence;
    event.support = match.support;
    event.specVersion = spec_.version;
    event.evidence.reserve(match.evidenceIndices.size());
    for (const auto index : match.evidenceIndices) {
      event.evidence.push_back(relations_[index]);
    }
    return event;
  }

  void MatchActions(std::uint64_t nowMs, std::vector<ActionEvent>& output) {
    PruneActionStates(nowMs);
    for (const auto& action : spec_.actions) {
      for (const auto& match : CurrentMatches(action.pattern, nowMs)) {
        const std::string fingerprint = MatchFingerprint(match, relations_);
        const std::string key =
            action.action + "\x1f" + EntityKey(match.source);
        const auto existing = actionStates_.find(key);
        if (existing != actionStates_.end() &&
            existing->second.fingerprint == fingerprint) {
          existing->second.lastMatchedMs = nowMs;
          continue;
        }
        if (existing != actionStates_.end() && action.cooldownMs > 0 &&
            nowMs - existing->second.lastEmittedMs <
                static_cast<std::uint64_t>(action.cooldownMs)) {
          existing->second.lastMatchedMs = nowMs;
          continue;
        }
        if (existing == actionStates_.end() &&
            actionStates_.size() >= spec_.maxActionStates) {
          throw std::length_error("THIG action state budget exhausted");
        }
        actionStates_[key] = {fingerprint, nowMs, nowMs};
        output.push_back(MakeActionEvent(action.action, match, nowMs));
      }
    }
  }

  void PruneActionStates(std::uint64_t nowMs) {
    const std::uint64_t cutoff =
        nowMs > static_cast<std::uint64_t>(spec_.historyMs)
            ? nowMs - static_cast<std::uint64_t>(spec_.historyMs)
            : 0;
    for (auto iterator = actionStates_.begin();
         iterator != actionStates_.end();) {
      if (iterator->second.lastMatchedMs < cutoff) {
        iterator = actionStates_.erase(iterator);
      } else {
        ++iterator;
      }
    }
  }

  bool SourceAllowed(const StateRuntime& runtime, SourceConstraint constraint,
                     const EntityRef& source) const {
    switch (constraint) {
      case SourceConstraint::Any:
        return true;
      case SourceConstraint::Bound:
        return runtime.boundSource.has_value() && source == *runtime.boundSource;
      case SourceConstraint::BoundIfPresent:
        return !runtime.boundSource.has_value() || source == *runtime.boundSource;
    }
    return false;
  }

  void MatchStateGraphs(std::uint64_t nowMs, std::vector<ActionEvent>& output) {
    for (const auto& graph : spec_.stateGraphs) {
      auto& runtime = stateRuntimes_.at(graph.id);
      std::unordered_set<std::string> consumedEvidence;
      // A timeout may release a binding and make one current observation legal
      // in the same frame. Bound the transition chain by the graph size.
      for (std::size_t step = 0; step <= graph.transitions.size(); ++step) {
        std::vector<const StateTransitionSpec*> transitions;
        for (const auto& transition : graph.transitions) {
          if (transition.fromState == runtime.state) {
            transitions.push_back(&transition);
          }
        }
        std::stable_sort(transitions.begin(), transitions.end(), [](const auto* left,
                                                                    const auto* right) {
          return left->priority > right->priority;
        });

        bool transitioned = false;
        for (const auto* transition : transitions) {
          if (nowMs - runtime.stateEnteredMs <
              static_cast<std::uint64_t>(transition->minStateDurationMs)) {
            continue;
          }
          for (const auto& match : CurrentMatches(
                   transition->trigger, nowMs,
                   &transition->stateLocalRelations, runtime.stateEnteredMs)) {
            if (!SourceAllowed(runtime, transition->sourceConstraint, match.source)) {
              continue;
            }
            const std::string fingerprint = MatchFingerprint(match, relations_);
            if (consumedEvidence.find(fingerprint) != consumedEvidence.end()) {
              continue;
            }
            if (runtime.lastFingerprintByTransition[transition->id] == fingerprint) {
              continue;
            }
            runtime.lastFingerprintByTransition[transition->id] = fingerprint;
            consumedEvidence.insert(fingerprint);
            const bool changesState = runtime.state != transition->toState;
            runtime.state = transition->toState;
            if (changesState) {
              runtime.stateEnteredMs = nowMs;
            }
            if (transition->bindSource) {
              runtime.boundSource = match.source;
              runtime.boundSourceLastObservedMs = nowMs;
            }
            if (transition->unbindSource) {
              runtime.boundSource.reset();
              runtime.boundSourceLastObservedMs = 0;
            }
            if (!transition->action.empty()) {
              output.push_back(MakeActionEvent(transition->action, match, nowMs));
            }
            transitioned = true;
            break;
          }
          if (transitioned) {
            break;
          }
        }
        if (!transitioned) {
          break;
        }
      }
    }
  }

  void Arbitrate(std::vector<ActionEvent>& output) const {
    const auto actionSpecOf = [&](const std::string& action) -> const ActionSpec* {
      const auto iterator = std::find_if(
          spec_.actions.begin(), spec_.actions.end(),
          [&](const auto& candidate) { return candidate.action == action; });
      return iterator == spec_.actions.end() ? nullptr : &*iterator;
    };
    std::stable_sort(output.begin(), output.end(), [&](const auto& left, const auto& right) {
      const auto priorityOf = [&](const std::string& action) {
        const auto* actionSpec = actionSpecOf(action);
        return actionSpec == nullptr ? 0 : actionSpec->priority;
      };
      const int leftPriority = priorityOf(left.action);
      const int rightPriority = priorityOf(right.action);
      return leftPriority == rightPriority ? left.confidence > right.confidence
                                           : leftPriority > rightPriority;
    });

    std::unordered_set<std::string> occupiedGroups;
    output.erase(std::remove_if(output.begin(), output.end(), [&](const auto& event) {
                   const auto* actionSpec = actionSpecOf(event.action);
                   if (actionSpec == nullptr || actionSpec->exclusiveGroup.empty()) {
                     return false;
                   }
                   return !occupiedGroups
                               .insert(actionSpec->exclusiveGroup + "\x1f" +
                                       EntityKey(event.source))
                               .second;
                 }),
                 output.end());
  }

  void PruneHistory(std::uint64_t nowMs) {
    const std::uint64_t cutoff =
        nowMs > static_cast<std::uint64_t>(spec_.historyMs)
            ? nowMs - static_cast<std::uint64_t>(spec_.historyMs)
            : 0;
    if (std::none_of(relations_.begin(), relations_.end(), [&](const auto& relation) {
          return !relation.active && relation.endMs < cutoff;
        })) {
      return;
    }
    std::vector<RelationEvent> retained;
    retained.reserve(relations_.size());
    for (auto& relation : relations_) {
      if (relation.active || relation.endMs >= cutoff) {
        retained.push_back(std::move(relation));
      }
    }
    relations_ = std::move(retained);
    activeRelationByKey_.clear();
    for (std::size_t index = 0; index < relations_.size(); ++index) {
      if (!relations_[index].active) {
        continue;
      }
      Observation observation;
      observation.source = relations_[index].source;
      observation.target = relations_[index].target;
      observation.relation = relations_[index].relation;
      activeRelationByKey_[RelationKey(observation)] = index;
    }
  }

  EngineSpec spec_;
  bool configured_ = false;
  std::unordered_map<std::string, RelationSpec> relationSpecs_;
  TemporalStabilizer stabilizer_;
  std::vector<RelationEvent> relations_;
  std::unordered_map<std::string, std::size_t> activeRelationByKey_;
  struct ActionState {
    std::string fingerprint;
    std::uint64_t lastEmittedMs = 0;
    std::uint64_t lastMatchedMs = 0;
  };
  std::unordered_map<std::string, ActionState> actionStates_;
  std::unordered_map<std::string, StateRuntime> stateRuntimes_;
  std::uint64_t nextEventId_ = 1;
  std::uint64_t lastProcessMs_ = 0;
  bool hasProcessed_ = false;
};

TemporalGraphEngine::TemporalGraphEngine() : impl_(std::make_unique<Impl>()) {}

TemporalGraphEngine::TemporalGraphEngine(EngineSpec spec) : TemporalGraphEngine() {
  Configure(std::move(spec));
}

TemporalGraphEngine::~TemporalGraphEngine() = default;
TemporalGraphEngine::TemporalGraphEngine(TemporalGraphEngine&&) noexcept = default;
TemporalGraphEngine& TemporalGraphEngine::operator=(TemporalGraphEngine&&) noexcept = default;

void TemporalGraphEngine::Configure(EngineSpec spec) {
  impl_->Configure(std::move(spec));
}

std::vector<ActionEvent> TemporalGraphEngine::ProcessFrame(
    const std::vector<Observation>& observations,
    std::chrono::steady_clock::time_point now) {
  return impl_->ProcessFrame(observations, ToMilliseconds(now));
}

void TemporalGraphEngine::Reset() { impl_->Reset(); }
const EngineSpec& TemporalGraphEngine::Spec() const { return impl_->Spec(); }
const std::vector<RelationEvent>& TemporalGraphEngine::Relations() const {
  return impl_->Relations();
}
std::size_t TemporalGraphEngine::ActionStateCount() const {
  return impl_->ActionStateCount();
}
std::string TemporalGraphEngine::StateOf(const std::string& graphId) const {
  return impl_->StateOf(graphId);
}

}  // namespace kfcore::thig
