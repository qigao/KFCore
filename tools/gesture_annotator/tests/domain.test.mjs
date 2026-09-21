import test from 'node:test';
import assert from 'node:assert/strict';
import { LIMITS, createProject, validateProject, validateAnnotation, parseProject,
  sampleTimes, nearestFrame, upsertFrame, AnnotationHistory } from '../dist/domain.mjs';

const video = { name: 'sample.mp4', size: 1000, sha256: 'a'.repeat(64), width: 640, height: 480, duration: 3.2 };
const annotation = () => ({ id: 'one', label: 'swipe_left', start: 0.1, end: 1.5,
  phase: 'whole', entities: ['hand-A'], flags: [], note: '' });
const hand = () => ({ entity: null, handedness: 'Left', handednessScore: 0.9,
  landmarks: Array.from({ length: 21 }, () => [0.2, 0.5, -0.01]),
  worldLandmarks: Array.from({ length: 21 }, () => [0.1, -0.1, 0]) });
const frame = (timestampUs = 0) => ({ timestampUs, hands: [hand()] });

test('project JSON round-trip retains raw data, independent intervals and null identity', () => {
  const p = createProject(video); p.annotations.push(annotation()); p.frames.push(frame());
  assert.deepEqual(parseProject(JSON.stringify(p)), p);
  assert.equal(p.frames[0].hands[0].entity, null);
});
test('reject incompatible schema and model provenance', () => {
  const p = createProject(video); p.schema = 'kfcore-temporal-gesture-classes/1';
  assert.throws(() => validateProject(p));
  p.schema = createProject(video).schema; p.inference.version = 'latest';
  assert.throws(() => validateProject(p));
});
test('video metadata is bounded and finite', () => {
  for (const change of [{ duration: NaN }, { duration: Infinity }, { duration: 0 },
    { size: LIMITS.videoBytes + 1 }, { sha256: 'no' }, { width: 0 }, { height: 8193 }]) {
    assert.throws(() => createProject({ ...video, ...change }));
  }
});
test('interval boundaries reject zero/reversed/out-of-range and nonfinite times', () => {
  for (const change of [{ start: -1 }, { end: 4 }, { end: 0.1 }, { start: 2 }, { start: NaN }, { end: Infinity }]) {
    assert.throws(() => validateAnnotation({ ...annotation(), ...change }, video.duration));
  }
  assert.doesNotThrow(() => validateAnnotation({ ...annotation(), start: 0, end: video.duration }, video.duration));
});
test('reject invalid identity, phase, flags, text and duplicate interval IDs', () => {
  for (const change of [{ entities: [] }, { entities: ['left'] }, { entities: ['hand-A', 'hand-A'] },
    { phase: 'unknown' }, { flags: ['unknown'] }, { label: ' ' }, { note: 'x'.repeat(2001) }]) {
    assert.throws(() => validateAnnotation({ ...annotation(), ...change }, video.duration));
  }
  const p = createProject(video); p.annotations = [annotation(), annotation()];
  assert.throws(() => validateProject(p), /重复/);
});
test('overlapping phases and two-hand actions are preserved, not silently resolved', () => {
  const p = createProject(video); p.annotations = [annotation(), { ...annotation(), id: 'two', phase: 'active', entities: ['hand-A', 'hand-B'] }];
  assert.equal(validateProject(p).annotations.length, 2);
});
test('sampling is bounded and excludes duration endpoint', () => {
  assert.deepEqual(sampleTimes(0.25, 10), [0, 0.1, 0.2]);
  assert.equal(sampleTimes(300, 10).length, LIMITS.frames);
  assert.throws(() => sampleTimes(301, 10), /最多/);
  assert.throws(() => sampleTimes(1, 0));
});
test('nearest sample uses bounded tolerance and handles empty/boundary positions', () => {
  const frames = [frame(0), frame(100000), frame(200000)];
  assert.equal(nearestFrame([], 0, 10), null);
  assert.equal(nearestFrame(frames, 0.11, 10), frames[1]);
  assert.equal(nearestFrame(frames, 0.26, 10), null);
  assert.equal(nearestFrame(frames, 0, 10), frames[0]);
});
test('frame upsert does not mutate old results and replaces only exact timestamp', () => {
  const frames = [frame(0), frame(200000)], replacement = { timestampUs: 0, hands: [] };
  const next = upsertFrame(frames, replacement);
  assert.equal(frames[0].hands.length, 1); assert.equal(next[0], replacement);
  assert.deepEqual(upsertFrame(next, frame(100000)).map(f => f.timestampUs), [0, 100000, 200000]);
});
test('timestamps are strictly increasing integer microseconds within video', () => {
  for (const frames of [[frame(1.5)], [frame(-1)], [frame(3300000)], [frame(1), frame(1)], [frame(2), frame(1)]]) {
    const p = createProject(video); p.frames = frames; assert.throws(() => validateProject(p));
  }
});
test('duplicate per-sample identity rejected; equal handedness allowed', () => {
  const p = createProject(video); p.frames = [{ timestampUs: 0, hands: [hand(), hand()] }];
  assert.doesNotThrow(() => validateProject(p));
  p.frames[0].hands.forEach(h => { h.entity = 'hand-A'; });
  assert.throws(() => validateProject(p), /同一身份/);
  p.frames[0].hands[1].entity = 'hand-B'; assert.doesNotThrow(() => validateProject(p));
});
test('raw keypoints must contain exactly 21 finite XYZ tuples', () => {
  for (const change of [h => h.landmarks.pop(), h => { h.landmarks[0][0] = NaN; },
    h => { h.worldLandmarks[0][2] = Infinity; }, h => { h.handednessScore = 1.1; }]) {
    const p = createProject(video); p.frames = [frame()]; change(p.frames[0].hands[0]);
    assert.throws(() => validateProject(p));
  }
});
test('empty detections are valid negative evidence, not invented hands', () => {
  const p = createProject(video); p.frames = [{ timestampUs: 0, hands: [] }];
  assert.doesNotThrow(() => validateProject(p));
});
test('history clones labels, supports redo and clears redo after new edit', () => {
  const history = new AnnotationHistory(), initial = [annotation()];
  history.record(initial); initial[0].label = 'changed';
  const previous = history.undo(initial); assert.equal(previous[0].label, 'swipe_left');
  assert.equal(history.redo(previous)[0].label, 'changed');
  history.undo(initial); history.record(previous); assert.equal(history.future.length, 0);
});
test('history storage is bounded', () => {
  const history = new AnnotationHistory();
  for (let i = 0; i < LIMITS.history + 5; i++) history.record([{ ...annotation(), label: String(i) }]);
  assert.equal(history.past.length, LIMITS.history); assert.equal(history.past[0][0].label, '5');
});
test('malformed JSON fails without changing existing project', () => {
  const p = createProject(video), before = structuredClone(p);
  for (const json of ['{', 'null', '{}', '[]']) assert.throws(() => parseProject(json));
  assert.deepEqual(p, before);
});
