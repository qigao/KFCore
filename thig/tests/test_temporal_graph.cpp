#include "kfcore/thig/temporal_graph.hpp"

#include <algorithm>
#include <chrono>
#include <string>

#include "tinytest.hpp"

namespace {

kfcore::thig::Observation Observe(std::string relation, std::int64_t sourceId,
                          std::uint64_t serial = 1) {
  kfcore::thig::Observation observation;
  observation.serial = serial;
  observation.source = {"test_subject", sourceId};
  observation.relation = std::move(relation);
  observation.confidence = 0.9f;
  observation.producer = "test";
  return observation;
}

kfcore::thig::Observation Reject(std::string relation, std::int64_t sourceId,
                         std::uint64_t serial) {
  auto observation = Observe(std::move(relation), sourceId, serial);
  observation.truth = kfcore::thig::TruthValue::False;
  return observation;
}

kfcore::thig::EngineSpec BasicSpec(std::string relation, int dwellMs) {
  kfcore::thig::EngineSpec spec;
  spec.version = "test-v1";
  spec.historyMs = 5000;
  spec.observationMaxGapMs = 150;
  spec.relations.push_back({relation, "pose"});
  spec.actions.push_back({"confirmed", kfcore::thig::PatternGraph::Atom(relation, dwellMs)});
  return spec;
}

kfcore::thig::EngineSpec BinaryPatternSpec(kfcore::thig::PatternOperator operation,
                                   int windowMs = 0,
                                   kfcore::thig::PatternSourceJoin sourceJoin =
                                       kfcore::thig::PatternSourceJoin::Same) {
  kfcore::thig::EngineSpec spec;
  spec.version = "binary-v1";
  spec.relations = {{"first", {}}, {"second", {}}};
  kfcore::thig::PatternGraph pattern;
  pattern.nodes = {
      {"first", kfcore::thig::PatternOperator::Atom, {}, "first"},
      {"second", kfcore::thig::PatternOperator::Atom, {}, "second"},
      {"root", operation, {0, 1}, {}, 0, windowMs, 0, sourceJoin},
  };
  pattern.root = 2;
  spec.actions.push_back({"composed", std::move(pattern)});
  return spec;
}

kfcore::thig::EngineSpec WindowedShapeSpec() {
  kfcore::thig::EngineSpec spec;
  spec.version = "windowed-shape-v1";
  spec.observationMaxGapMs = 250;
  spec.relations = {{"open", "shape"},
                    {"fist", "shape"},
                    {"pointer", "shape"},
                    {"unclassified", "shape"}};
  kfcore::thig::ObservationWindowSpec window;
  window.exclusiveGroup = "shape";
  window.windowMs = 300;
  window.minimumSupportingObservations = 2;
  window.minimumSupportRatio = 0.60f;
  window.switchMargin = 0.15f;
  window.maxSamplesPerSource = 8;
  window.ignoredRelations = {"unclassified"};
  spec.observationWindows.push_back(std::move(window));
  spec.actions.push_back({"stable_open", kfcore::thig::PatternGraph::Atom("open")});
  return spec;
}

const kfcore::thig::RelationEvent* ActiveRelation(
    const kfcore::thig::TemporalGraphEngine& engine, const std::string& relation) {
  const auto& relations = engine.Relations();
  const auto iterator = std::find_if(
      relations.begin(), relations.end(), [&](const auto& candidate) {
        return candidate.active && candidate.relation == relation;
      });
  return iterator == relations.end() ? nullptr : &*iterator;
}

}  // namespace

spec("THIG temporal graph") {
  it("derives an action from graph data and preserves UNKNOWN evidence") {
    kfcore::thig::TemporalGraphEngine engine(BasicSpec("candidate", 100));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("candidate", 11, 10)}, now));
    check_empty(engine.ProcessFrame({}, now + std::chrono::milliseconds(50)));
    const auto actions = engine.ProcessFrame(
        {Observe("candidate", 11, 12)}, now + std::chrono::milliseconds(100));

    check_size(actions, 1);
    check(actions[0].action == "confirmed");
    check_size(actions[0].evidence, 1);
    check(actions[0].evidence[0].rawFrames == 3);
    check(actions[0].evidence[0].observedFrames == 2);
    check(actions[0].evidence[0].unknownFrames == 1);
    check_within(actions[0].support, 1.0f, 0.0001f);
    check(actions[0].specVersion == "test-v1");
  }

  it("treats an exclusive observed relation as FALSE and starts new evidence") {
    auto spec = BasicSpec("circle", 200);
    spec.observationMaxGapMs = 1000;
    spec.relations.push_back({"open", "pose"});
    kfcore::thig::TemporalGraphEngine engine(std::move(spec));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("circle", 8, 1)}, now));
    check_empty(engine.ProcessFrame({Observe("open", 8, 2)},
                                    now + std::chrono::milliseconds(100)));
    check_empty(engine.ProcessFrame({Observe("circle", 8, 3)},
                                    now + std::chrono::milliseconds(200)));
    const auto actions = engine.ProcessFrame(
        {Observe("circle", 8, 4)}, now + std::chrono::milliseconds(400));

    check_size(actions, 1);
    check(static_cast<int>(actions[0].evidence[0].firstSourceSerial) == 3);
    check(actions[0].evidence[0].unknownFrames == 0);
  }

  it("matches a declarative sequence with consistent entity binding") {
    kfcore::thig::EngineSpec spec;
    spec.version = "sequence-v1";
    spec.relations = {{"start", "phase"}, {"finish", "phase"}};
    kfcore::thig::PatternGraph sequence;
    sequence.nodes = {
        {"start", kfcore::thig::PatternOperator::Atom, {}, "start"},
        {"finish", kfcore::thig::PatternOperator::Atom, {}, "finish"},
        {"sequence", kfcore::thig::PatternOperator::Seq, {0, 1}},
    };
    sequence.root = 2;
    spec.actions.push_back({"completed", sequence});
    kfcore::thig::TemporalGraphEngine engine(std::move(spec));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("start", 4)}, now));
    const auto actions = engine.ProcessFrame(
        {Observe("finish", 4, 2)}, now + std::chrono::milliseconds(50));

    check_size(actions, 1);
    check(actions[0].action == "completed");
    check_size(actions[0].evidence, 2);
  }

  it("bounds Seq by the gap between consecutive relations") {
    kfcore::thig::TemporalGraphEngine engine(
        BinaryPatternSpec(kfcore::thig::PatternOperator::Seq, 50));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("first", 4, 1)}, now));
    check_empty(engine.ProcessFrame(
        {Reject("first", 4, 2)}, now + std::chrono::milliseconds(10)));
    check_empty(engine.ProcessFrame(
        {Observe("second", 4, 3)}, now + std::chrono::milliseconds(100)));
    check_empty(engine.ProcessFrame(
        {Observe("first", 4, 4), Reject("second", 4, 5)},
        now + std::chrono::milliseconds(110)));
    const auto actions = engine.ProcessFrame(
        {Reject("first", 4, 6), Observe("second", 4, 7)},
        now + std::chrono::milliseconds(120));

    check_size(actions, 1);
    check_size(actions[0].evidence, 2);
  }

  it("matches concurrent evidence with Both") {
    kfcore::thig::TemporalGraphEngine engine(
        BinaryPatternSpec(kfcore::thig::PatternOperator::Both));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("first", 4)}, now));
    const auto actions = engine.ProcessFrame(
        {Observe("first", 4, 2), Observe("second", 4, 3)},
        now + std::chrono::milliseconds(10));

    check_size(actions, 1);
    check_size(actions[0].evidence, 2);
  }

  it("requires configured overlap instead of boundary contact for Both") {
    auto spec = BinaryPatternSpec(kfcore::thig::PatternOperator::Both);
    spec.actions[0].pattern.nodes[2].minOverlapMs = 10;
    kfcore::thig::TemporalGraphEngine engine(std::move(spec));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("first", 4, 1)}, now));
    check_empty(engine.ProcessFrame(
        {Observe("first", 4, 2), Observe("second", 4, 3)},
        now + std::chrono::milliseconds(20)));
    const auto actions = engine.ProcessFrame(
        {Observe("first", 4, 4), Observe("second", 4, 5)},
        now + std::chrono::milliseconds(30));

    check_size(actions, 1);
  }

  it("does not revive historical concurrent evidence in a later state") {
    kfcore::thig::EngineSpec spec;
    spec.version = "concurrent-recency-v1";
    spec.observationMaxGapMs = 1000;
    spec.relations = {{"first", {}}, {"second", {}}, {"unlock", {}}};

    kfcore::thig::PatternGraph concurrent;
    concurrent.nodes = {
        {"first", kfcore::thig::PatternOperator::Atom, {}, "first"},
        {"second", kfcore::thig::PatternOperator::Atom, {}, "second"},
        {"both", kfcore::thig::PatternOperator::Both, {0, 1}},
    };
    concurrent.root = 2;

    kfcore::thig::StateGraphSpec graph;
    graph.id = "recency";
    graph.initialState = "blocked";
    graph.states = {"blocked", "ready", "done"};
    graph.transitions = {
        {"unlock", "blocked", "ready", kfcore::thig::PatternGraph::Atom("unlock")},
        {"consume", "ready", "done", std::move(concurrent),
         kfcore::thig::SourceConstraint::Any, false, false, "stale_concurrent"},
    };
    spec.stateGraphs.push_back(std::move(graph));

    kfcore::thig::TemporalGraphEngine engine(std::move(spec));
    const auto now = std::chrono::steady_clock::now();
    check_empty(engine.ProcessFrame(
        {Observe("first", 4, 1), Observe("second", 4, 2)}, now));

    const auto actions = engine.ProcessFrame(
        {Reject("first", 4, 3), Observe("second", 4, 4),
         Observe("unlock", 4, 5)},
        now + std::chrono::milliseconds(10));

    check_empty(actions);
    check(engine.StateOf("recency") == "ready");
  }

  it("can require selected relation evidence to begin in the current state") {
    kfcore::thig::EngineSpec spec;
    spec.version = "state-local-evidence-v1";
    spec.observationMaxGapMs = 1000;
    spec.relations = {{"candidate", {}}, {"unlock", {}}};

    kfcore::thig::StateGraphSpec graph;
    graph.id = "state_local";
    graph.initialState = "blocked";
    graph.states = {"blocked", "ready", "done"};
    graph.transitions.push_back(
        {"unlock", "blocked", "ready", kfcore::thig::PatternGraph::Atom("unlock")});
    kfcore::thig::StateTransitionSpec consume;
    consume.id = "consume";
    consume.fromState = "ready";
    consume.toState = "done";
    consume.trigger = kfcore::thig::PatternGraph::Atom("candidate");
    consume.sourceConstraint = kfcore::thig::SourceConstraint::Any;
    consume.action = "accepted";
    consume.stateLocalRelations = {"candidate"};
    graph.transitions.push_back(std::move(consume));
    spec.stateGraphs.push_back(std::move(graph));

    kfcore::thig::TemporalGraphEngine engine(std::move(spec));
    const auto now = std::chrono::steady_clock::now();
    check_empty(engine.ProcessFrame({Observe("candidate", 4, 1)}, now));
    check_empty(engine.ProcessFrame(
        {Reject("candidate", 4, 2), Observe("unlock", 4, 3)},
        now + std::chrono::milliseconds(10)));
    check(engine.StateOf("state_local") == "ready");
  }

  it("keeps active concurrent evidence through an observed dropout") {
    auto spec = BinaryPatternSpec(kfcore::thig::PatternOperator::Both);
    spec.observationMaxGapMs = 100;
    spec.actions[0].pattern.nodes[0].minDurationMs = 20;
    spec.actions[0].pattern.nodes[1].minDurationMs = 20;
    spec.actions[0].pattern.nodes[2].minOverlapMs = 20;
    kfcore::thig::TemporalGraphEngine engine(std::move(spec));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame(
        {Observe("first", 4, 1), Observe("second", 4, 2)}, now));
    check_empty(engine.ProcessFrame(
        {Observe("first", 4, 3)}, now + std::chrono::milliseconds(10)));
    const auto actions = engine.ProcessFrame(
        {Observe("first", 4, 4)}, now + std::chrono::milliseconds(20));

    check_size(actions, 1);
    check_size(actions[0].evidence, 2);
    check(actions[0].evidence[1].unknownFrames == 2);
  }

  it("filters transient and ignored shape ambiguity in a time window") {
    kfcore::thig::TemporalGraphEngine engine(WindowedShapeSpec());
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("open", 4, 1)}, now));
    check_size(engine.ProcessFrame(
                   {Observe("open", 4, 2)},
                   now + std::chrono::milliseconds(100)),
               1);
    check_empty(engine.ProcessFrame(
        {Observe("pointer", 4, 3)}, now + std::chrono::milliseconds(150)));
    check_empty(engine.ProcessFrame(
        {Observe("unclassified", 4, 4)},
        now + std::chrono::milliseconds(175)));
    check_empty(engine.ProcessFrame(
        {Observe("open", 4, 5)}, now + std::chrono::milliseconds(200)));

    const auto* open = ActiveRelation(engine, "open");
    check_not_null(open);
    check(open->unknownFrames == 2);
    check_null(ActiveRelation(engine, "pointer"));
    check_null(ActiveRelation(engine, "unclassified"));
  }

  it("switches an exclusive relation only after window support wins") {
    kfcore::thig::TemporalGraphEngine engine(WindowedShapeSpec());
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("open", 4, 1)}, now));
    check_size(engine.ProcessFrame(
                   {Observe("open", 4, 2)},
                   now + std::chrono::milliseconds(100)),
               1);
    check_empty(engine.ProcessFrame(
        {Observe("pointer", 4, 3)}, now + std::chrono::milliseconds(200)));
    check_empty(engine.ProcessFrame(
        {Observe("pointer", 4, 4)}, now + std::chrono::milliseconds(300)));
    check_empty(engine.ProcessFrame(
        {Observe("pointer", 4, 5)}, now + std::chrono::milliseconds(400)));

    check_null(ActiveRelation(engine, "open"));
    const auto* pointer = ActiveRelation(engine, "pointer");
    check_not_null(pointer);
    const auto expectedStart = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch())
            .count() +
        200);
    check(pointer->startMs == expectedStart);
  }

  it("rejects a stable transition relation on observed counter-evidence") {
    auto spec = WindowedShapeSpec();
    spec.observationWindows[0].rejectStableOnConflict = true;
    kfcore::thig::TemporalGraphEngine engine(std::move(spec));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("open", 4, 1)}, now));
    check_size(engine.ProcessFrame(
                   {Observe("open", 4, 2)},
                   now + std::chrono::milliseconds(100)),
               1);
    check_empty(engine.ProcessFrame(
        {Observe("pointer", 4, 3)},
        now + std::chrono::milliseconds(150)));

    check_null(ActiveRelation(engine, "open"));
    check_null(ActiveRelation(engine, "pointer"));
  }

  it("invalidates a sequence containing a stable forbidden relation") {
    kfcore::thig::EngineSpec spec;
    spec.version = "forbidden-v1";
    spec.relations = {{"start", "phase"},
                      {"finish", "phase"},
                      {"wrong", {}}};
    kfcore::thig::PatternGraph sequence;
    sequence.nodes = {
        {"start", kfcore::thig::PatternOperator::Atom, {}, "start"},
        {"finish", kfcore::thig::PatternOperator::Atom, {}, "finish"},
        {"sequence", kfcore::thig::PatternOperator::Seq, {0, 1}},
    };
    sequence.root = 2;
    sequence.forbiddenRelations = {"wrong"};
    spec.actions.push_back({"completed", std::move(sequence)});
    kfcore::thig::TemporalGraphEngine engine(std::move(spec));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("start", 4, 1)}, now));
    check_empty(engine.ProcessFrame(
        {Observe("wrong", 4, 2)}, now + std::chrono::milliseconds(10)));
    check_empty(engine.ProcessFrame(
        {Observe("finish", 4, 3)}, now + std::chrono::milliseconds(20)));
  }

  it("fails fast when observation window state capacity is exhausted") {
    auto spec = WindowedShapeSpec();
    spec.maxObservationWindowStates = 1;
    kfcore::thig::TemporalGraphEngine engine(std::move(spec));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("open", 1, 1)}, now));
    check_throws_as(
        engine.ProcessFrame({Observe("open", 2, 2)},
                            now + std::chrono::milliseconds(1)),
        std::length_error);
  }

  it("rejects overlap constraints on non-concurrent operators") {
    auto spec = BasicSpec("candidate", 0);
    spec.actions[0].pattern.nodes[0].minOverlapMs = 1;

    check_throws_as(kfcore::thig::TemporalGraphEngine(std::move(spec)),
                    std::invalid_argument);
  }

  it("joins concurrent evidence from two distinct sources") {
    kfcore::thig::TemporalGraphEngine engine(BinaryPatternSpec(
        kfcore::thig::PatternOperator::Both, 100,
        kfcore::thig::PatternSourceJoin::Distinct));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame(
        {Observe("first", 4, 1), Observe("second", 4, 2)}, now));
    const auto actions = engine.ProcessFrame(
        {Observe("first", 4, 3), Observe("second", 8, 4)},
        now + std::chrono::milliseconds(20));

    check_size(actions, 1);
    check(static_cast<int>(actions[0].source.id) == 4);
    check_true(actions[0].target.has_value());
    check(static_cast<int>(actions[0].target->id) == 8);
    check_size(actions[0].evidence, 2);
  }

  it("rejects distinct-source evidence outside the onset window") {
    kfcore::thig::TemporalGraphEngine engine(BinaryPatternSpec(
        kfcore::thig::PatternOperator::Both, 50,
        kfcore::thig::PatternSourceJoin::Distinct));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("first", 4, 1)}, now));
    check_empty(engine.ProcessFrame(
        {Observe("first", 4, 2), Observe("second", 8, 3)},
        now + std::chrono::milliseconds(80)));
  }

  it("requires a common overlap when Both operators are nested") {
    kfcore::thig::EngineSpec spec;
    spec.version = "nested-both-v1";
    spec.observationMaxGapMs = 1000;
    spec.relations = {{"first", {}}, {"second", {}}, {"third", {}}};
    kfcore::thig::PatternGraph pattern;
    pattern.nodes = {
        {"first", kfcore::thig::PatternOperator::Atom, {}, "first"},
        {"second", kfcore::thig::PatternOperator::Atom, {}, "second"},
        {"first_and_second", kfcore::thig::PatternOperator::Both, {0, 1}},
        {"third", kfcore::thig::PatternOperator::Atom, {}, "third"},
        {"all", kfcore::thig::PatternOperator::Both, {2, 3}},
    };
    pattern.root = 4;
    spec.actions.push_back({"composed", std::move(pattern)});
    kfcore::thig::TemporalGraphEngine engine(std::move(spec));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("first", 4, 1)}, now));
    check_empty(engine.ProcessFrame(
        {Observe("first", 4, 2), Observe("second", 4, 3)},
        now + std::chrono::milliseconds(10)));
    check_empty(engine.ProcessFrame(
        {Reject("first", 4, 4), Observe("second", 4, 5),
         Observe("third", 4, 6)},
        now + std::chrono::milliseconds(20)));

    const auto actions = engine.ProcessFrame(
        {Observe("first", 4, 7), Observe("second", 4, 8),
         Observe("third", 4, 9)},
        now + std::chrono::milliseconds(30));
    check_size(actions, 1);
    check_size(actions[0].evidence, 3);
  }

  it("matches a relation contained by another with During") {
    kfcore::thig::TemporalGraphEngine engine(
        BinaryPatternSpec(kfcore::thig::PatternOperator::During));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("second", 4)}, now));
    const auto actions = engine.ProcessFrame(
        {Observe("first", 4, 2), Observe("second", 4, 3)},
        now + std::chrono::milliseconds(10));

    check_size(actions, 1);
    check_size(actions[0].evidence, 2);
    check(static_cast<int>(actions[0].endMs - actions[0].startMs) == 0);
  }

  it("keeps forbidden evidence inside a During action interval") {
    kfcore::thig::EngineSpec spec;
    spec.version = "during-forbidden-v1";
    spec.relations = {{"left", "direction"},
                      {"right", "direction"},
                      {"neutral", "direction"},
                      {"open", {}}};
    kfcore::thig::PatternGraph wave;
    wave.nodes = {
        {"first_left", kfcore::thig::PatternOperator::Atom, {}, "left"},
        {"right", kfcore::thig::PatternOperator::Atom, {}, "right"},
        {"last_left", kfcore::thig::PatternOperator::Atom, {}, "left"},
        {"first_reversal", kfcore::thig::PatternOperator::Seq, {0, 1}},
        {"second_reversal", kfcore::thig::PatternOperator::Seq, {3, 2}},
        {"open", kfcore::thig::PatternOperator::Atom, {}, "open"},
        {"open_wave", kfcore::thig::PatternOperator::During, {4, 5}},
    };
    wave.root = 6;
    wave.forbiddenRelations = {"neutral"};
    spec.actions.push_back({"wave", std::move(wave)});
    kfcore::thig::TemporalGraphEngine engine(std::move(spec));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame(
        {Observe("left", 4, 1), Observe("open", 4, 2)}, now));
    check_empty(engine.ProcessFrame(
        {Observe("neutral", 4, 3)}, now + std::chrono::milliseconds(50)));
    check_empty(engine.ProcessFrame(
        {Observe("right", 4, 4), Observe("open", 4, 5)},
        now + std::chrono::milliseconds(100)));
    check_empty(engine.ProcessFrame(
        {Observe("left", 4, 6), Observe("open", 4, 7)},
        now + std::chrono::milliseconds(200)));
  }

  it("matches two relations inside a bounded combined window") {
    kfcore::thig::TemporalGraphEngine engine(
        BinaryPatternSpec(kfcore::thig::PatternOperator::Within, 50));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("first", 4)}, now));
    const auto actions = engine.ProcessFrame(
        {Observe("second", 4, 2)}, now + std::chrono::milliseconds(40));

    check_size(actions, 1);
  }

  it("matches ordered onsets inside an OnsetWithin window") {
    kfcore::thig::TemporalGraphEngine engine(
        BinaryPatternSpec(kfcore::thig::PatternOperator::OnsetWithin, 50));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("first", 4)}, now));
    const auto actions = engine.ProcessFrame(
        {Observe("first", 4, 2), Observe("second", 4, 3)},
        now + std::chrono::milliseconds(40));

    check_size(actions, 1);
  }

  it("matches distinct repeated relation intervals") {
    auto spec = BasicSpec("pulse", 0);
    kfcore::thig::PatternGraph repeat;
    repeat.nodes = {
        {"pulse", kfcore::thig::PatternOperator::Atom, {}, "pulse"},
        {"twice", kfcore::thig::PatternOperator::Repeat, {0}, {}, 0, 100, 2},
    };
    repeat.root = 1;
    spec.actions[0].pattern = std::move(repeat);
    kfcore::thig::TemporalGraphEngine engine(std::move(spec));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("pulse", 4)}, now));
    auto ended = Observe("pulse", 4, 2);
    ended.truth = kfcore::thig::TruthValue::False;
    check_empty(engine.ProcessFrame({ended}, now + std::chrono::milliseconds(10)));
    const auto actions = engine.ProcessFrame(
        {Observe("pulse", 4, 3)}, now + std::chrono::milliseconds(20));

    check_size(actions, 1);
    check_size(actions[0].evidence, 2);
  }

  it("advances a state graph only through declared transitions") {
    kfcore::thig::EngineSpec spec;
    spec.version = "state-v1";
    spec.relations = {{"advance", "motion"}, {"return", "motion"}};
    kfcore::thig::StateGraphSpec graph;
    graph.id = "cycle";
    graph.initialState = "ready";
    graph.states = {"ready", "waiting"};
    graph.transitions = {
        {"advance", "ready", "waiting", kfcore::thig::PatternGraph::Atom("advance"),
         kfcore::thig::SourceConstraint::Any, true, false, "accepted"},
        {"return", "waiting", "ready", kfcore::thig::PatternGraph::Atom("return"),
         kfcore::thig::SourceConstraint::Bound, false, true},
    };
    spec.stateGraphs.push_back(std::move(graph));
    kfcore::thig::TemporalGraphEngine engine(std::move(spec));
    const auto now = std::chrono::steady_clock::now();

    const auto accepted = engine.ProcessFrame({Observe("advance", 7)}, now);
    check_size(accepted, 1);
    check(accepted[0].action == "accepted");
    check(engine.StateOf("cycle") == "waiting");

    check_empty(engine.ProcessFrame({Observe("return", 8, 2)},
                                    now + std::chrono::milliseconds(10)));
    check(engine.StateOf("cycle") == "waiting");
    check_empty(engine.ProcessFrame({Observe("return", 7, 3)},
                                    now + std::chrono::milliseconds(20)));
    check(engine.StateOf("cycle") == "ready");
  }

  it("guards a transition by time spent in its source state") {
    kfcore::thig::EngineSpec spec;
    spec.version = "state-dwell-v1";
    spec.relations = {{"advance", "motion"}, {"return", "motion"}};
    kfcore::thig::StateGraphSpec graph;
    graph.id = "cycle";
    graph.initialState = "ready";
    graph.states = {"ready", "waiting"};
    graph.transitions = {
        {"advance", "ready", "waiting", kfcore::thig::PatternGraph::Atom("advance"),
         kfcore::thig::SourceConstraint::Any, true, false},
        {"return", "waiting", "ready", kfcore::thig::PatternGraph::Atom("return"),
         kfcore::thig::SourceConstraint::Bound, false, true, {}, 0, 100},
    };
    spec.stateGraphs.push_back(std::move(graph));
    kfcore::thig::TemporalGraphEngine engine(std::move(spec));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("advance", 7, 1)}, now));
    check_empty(engine.ProcessFrame(
        {Observe("return", 7, 2)}, now + std::chrono::milliseconds(50)));
    check(engine.StateOf("cycle") == "waiting");
    check_empty(engine.ProcessFrame(
        {Observe("return", 7, 3)}, now + std::chrono::milliseconds(100)));
    check(engine.StateOf("cycle") == "ready");
  }

  it("rejects a negative state-duration guard") {
    auto spec = BasicSpec("candidate", 0);
    kfcore::thig::StateGraphSpec graph;
    graph.id = "invalid";
    graph.initialState = "ready";
    graph.states = {"ready"};
    kfcore::thig::StateTransitionSpec transition;
    transition.id = "invalid_duration";
    transition.fromState = "ready";
    transition.toState = "ready";
    transition.trigger = kfcore::thig::PatternGraph::Atom("candidate");
    transition.minStateDurationMs = -1;
    graph.transitions.push_back(std::move(transition));
    spec.stateGraphs.push_back(std::move(graph));

    check_throws_as(kfcore::thig::TemporalGraphEngine(std::move(spec)),
                    std::invalid_argument);
  }

  it("arbitrates same-axis action conflicts by declared priority") {
    auto graph = BasicSpec("candidate", 0);
    graph.actions.clear();
    kfcore::thig::ActionSpec low;
    low.action = "low";
    low.pattern = kfcore::thig::PatternGraph::Atom("candidate");
    low.exclusiveGroup = "axis";
    low.priority = 10;
    kfcore::thig::ActionSpec high = low;
    high.action = "high";
    high.priority = 20;
    graph.actions = {low, high};
    kfcore::thig::TemporalGraphEngine engine(std::move(graph));

    const auto actions = engine.ProcessFrame(
        {Observe("candidate", 7)}, std::chrono::steady_clock::now());

    check_size(actions, 1);
    check(actions[0].action == "high");
  }

  it("rejects a cyclic pattern graph before it can run") {
    auto spec = BasicSpec("candidate", 0);
    spec.actions[0].pattern.nodes = {
        {"cycle", kfcore::thig::PatternOperator::Seq, {0, 0}},
    };
    check_throws_as(kfcore::thig::TemporalGraphEngine(std::move(spec)), std::invalid_argument);
  }

  it("rejects non-monotonic frames without mutating relation evidence") {
    kfcore::thig::TemporalGraphEngine engine(BasicSpec("candidate", 100));
    const auto now = std::chrono::steady_clock::now();

    check_empty(engine.ProcessFrame({Observe("candidate", 3)}, now));
    const auto relationCount = engine.Relations().size();
    check_throws_as(
        engine.ProcessFrame({Observe("candidate", 3, 2)},
                            now - std::chrono::milliseconds(1)),
        std::invalid_argument);
    check_size(engine.Relations(), relationCount);
  }

  it("fails fast when the rolling graph item budget is exhausted") {
    auto graph = BasicSpec("candidate", 0);
    graph.maxRelationEvents = 1;
    kfcore::thig::TemporalGraphEngine engine(std::move(graph));
    const auto now = std::chrono::steady_clock::now();

    check_size(engine.ProcessFrame({Observe("candidate", 1)}, now), 1);
    check_throws_as(
        engine.ProcessFrame({Observe("candidate", 2, 2)},
                            now + std::chrono::milliseconds(1)),
        std::length_error);
    check_size(engine.Relations(), 1);
  }
}

