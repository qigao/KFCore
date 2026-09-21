export const SCHEMA = 'kfcore-gesture-annotation/1';
export const LIMITS = Object.freeze({ videoBytes: 256 * 1024 * 1024, jsonBytes: 48 * 1024 * 1024,
  frames: 3000, annotations: 1000, duration: 3600, history: 30, hands: 2 });
export const MODEL = Object.freeze({ package: '@mediapipe/tasks-vision', version: '0.10.21',
  model: 'hand_landmarker/float16/1', runningMode: 'IMAGE', delegate: 'CPU',
  detectionConfidence: 0.5, presenceConfidence: 0.5,
  url: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' });
export const PHASES = ['whole', 'start', 'active', 'end', 'idle'];
export const ENTITIES = ['hand-A', 'hand-B'];
export const FLAGS = ['occluded', 'incomplete', 'negative'];
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const finite = (x, min, max) => typeof x === 'number' && Number.isFinite(x) && x >= min && x <= max;
const text = (s, max, empty = false) => typeof s === 'string' && s.length <= max && (empty || s.trim().length > 0);

export function createProject(video) {
  return validateProject({ schema: SCHEMA, video, subject: '', session: '',
    inference: { ...MODEL }, sampling: { hz: 10, timestampSource: 'html-video-currentTime' },
    annotations: [], frames: [] });
}

export function validateAnnotation(a, duration) {
  assert(a && text(a.id, 80), '标注 ID 无效');
  assert(text(a.label, 80), '动作名称须为 1–80 个字符');
  assert(finite(a.start, 0, duration) && finite(a.end, 0, duration) && a.end > a.start,
    '时间段必须满足 0 ≤ 起点 < 终点 ≤ 视频时长');
  assert(PHASES.includes(a.phase), '阶段无效');
  assert(Array.isArray(a.entities) && a.entities.length >= 1 && a.entities.length <= LIMITS.hands &&
    a.entities.every(x => ENTITIES.includes(x)) && new Set(a.entities).size === a.entities.length, '手身份无效');
  assert(Array.isArray(a.flags) && a.flags.every(x => FLAGS.includes(x)) && new Set(a.flags).size === a.flags.length, '标记无效');
  assert(text(a.note, 2000, true), '备注最多 2000 个字符');
  return a;
}

export function validateProject(p) {
  assert(p?.schema === SCHEMA, '不支持的标注格式或版本');
  const v = p.video;
  assert(v && text(v.name, 255) && Number.isSafeInteger(v.size) && v.size > 0 && v.size <= LIMITS.videoBytes &&
    /^[a-f0-9]{64}$/.test(v.sha256), '视频身份信息无效');
  assert(finite(v.duration, Number.EPSILON, LIMITS.duration) &&
    Number.isInteger(v.width) && v.width > 0 && v.width <= 8192 &&
    Number.isInteger(v.height) && v.height > 0 && v.height <= 8192, '视频尺寸或时长超限');
  assert(text(p.subject, 120, true) && text(p.session, 120, true), '被试或会话字段无效');
  assert(p.inference && Object.entries(MODEL).every(([k, value]) => p.inference[k] === value), '模型来源或配置不匹配');
  assert([5, 10, 15, 30].includes(p.sampling?.hz) && p.sampling.timestampSource === 'html-video-currentTime', '采样配置无效');
  assert(Array.isArray(p.annotations) && p.annotations.length <= LIMITS.annotations, '标注数量超限');
  const ids = new Set();
  for (const a of p.annotations) {
    validateAnnotation(a, v.duration);
    assert(!ids.has(a.id), '标注 ID 重复'); ids.add(a.id);
  }
  assert(Array.isArray(p.frames) && p.frames.length <= LIMITS.frames, '采样数量超限');
  let previous = -1;
  for (const f of p.frames) {
    assert(Number.isSafeInteger(f.timestampUs) && f.timestampUs >= 0 &&
      f.timestampUs <= Math.round(v.duration * 1e6) && f.timestampUs > previous, '采样时间必须递增且位于视频内');
    previous = f.timestampUs;
    assert(Array.isArray(f.hands) && f.hands.length <= LIMITS.hands, '检测手数量无效');
    const entities = new Set();
    for (const hand of f.hands) {
      assert(hand.entity === null || ENTITIES.includes(hand.entity), '采样身份无效');
      if (hand.entity !== null) { assert(!entities.has(hand.entity), '同一采样的两只手不能绑定同一身份'); entities.add(hand.entity); }
      assert(['Left', 'Right'].includes(hand.handedness) && finite(hand.handednessScore, 0, 1), '左右手分类无效');
      for (const key of ['landmarks', 'worldLandmarks']) {
        assert(Array.isArray(hand[key]) && hand[key].length === 21 && hand[key].every(point =>
          Array.isArray(point) && point.length === 3 && point.every(n => finite(n, -100, 100))), '关键点须包含 21 组有限 XYZ');
      }
    }
  }
  return p;
}

export function parseProject(json) {
  assert(new TextEncoder().encode(json).byteLength <= LIMITS.jsonBytes, 'JSON 文件超限');
  return validateProject(JSON.parse(json));
}

export function sampleTimes(duration, hz) {
  assert(finite(duration, Number.EPSILON, LIMITS.duration) && [5, 10, 15, 30].includes(hz), '采样参数无效');
  const count = Math.ceil(duration * hz);
  assert(count <= LIMITS.frames, `最多 ${LIMITS.frames} 个采样，请降低采样率或裁短视频`);
  return Array.from({ length: count }, (_, i) => i / hz);
}

export function nearestFrame(frames, seconds, hz) {
  const time = Math.round(seconds * 1e6);
  let lo = 0, hi = frames.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (frames[mid].timestampUs < time) lo = mid + 1; else hi = mid; }
  const candidates = [frames[lo - 1], frames[lo]].filter(Boolean);
  const best = candidates.sort((a, b) => Math.abs(a.timestampUs - time) - Math.abs(b.timestampUs - time))[0];
  return best && Math.abs(best.timestampUs - time) <= 0.51e6 / hz ? best : null;
}

export function upsertFrame(frames, frame) {
  const next = frames.filter(f => f.timestampUs !== frame.timestampUs);
  assert(next.length < LIMITS.frames, '采样数量超限');
  return [...next, frame].sort((a, b) => a.timestampUs - b.timestampUs);
}

// Only human intervals enter this bounded history; inference arrays are not copied per edit.
export class AnnotationHistory {
  past = []; future = [];
  record(annotations) {
    this.past.push(structuredClone(annotations));
    if (this.past.length > LIMITS.history) this.past.shift();
    this.future = [];
  }
  undo(current) {
    if (!this.past.length) return current;
    this.future.push(structuredClone(current)); return this.past.pop();
  }
  redo(current) {
    if (!this.future.length) return current;
    this.past.push(structuredClone(current)); return this.future.pop();
  }
}
