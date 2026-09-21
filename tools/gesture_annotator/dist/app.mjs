import { LIMITS, ENTITIES, FLAGS, createProject, validateProject, validateAnnotation,
  parseProject, sampleTimes, nearestFrame, upsertFrame, AnnotationHistory } from './domain.mjs';
import { loadProject, saveProject } from './storage.mjs';

const $ = id => document.getElementById(id);
const video = $('video');
const colors = ['#a4e8bc', '#a7cfff'];
const edges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
const WORKER_TIMEOUT_MS = 120000;
const MEDIA_TIMEOUT_MS = 15000;
let project = null, history = new AnnotationHistory(), editing = null, objectUrl = null;
let busy = false, cancelled = false, worker = null, pending = null, requestId = 0;
let saveTimer, revision = 0, savedRevision = 0, saveChain = Promise.resolve(), shownFrame;

function status(message, error = false) { $('status').textContent = message; $('status').classList.toggle('error', error); }
function safe(fn) { return async event => { try { await fn(event); } catch (error) { status(error.message, true); } }; }
function requireReady() { if (!project || busy) throw new Error('请先打开视频，并等待当前操作完成'); }
function setBusy(value) {
  busy = value;
  for (const id of ['transport', 'editor', 'metadata', 'jsonFile', 'export']) $(id).disabled = value || !project;
  $('videoFile').disabled = value;
  $('undo').disabled = value || !history.past.length;
  $('redo').disabled = value || !history.future.length;
  for (const element of document.querySelectorAll('#annotations button,#hands select,#timeline button')) element.disabled = value;
}
function changed() {
  revision++;
  $('storageStatus').textContent = '有未保存更改…';
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => void persist(), 350);
}
async function persist() {
  if (!project) return;
  const snapshot = structuredClone(project), version = revision;
  saveChain = saveChain.catch(() => {}).then(() => saveProject(snapshot));
  try {
    await saveChain;
    savedRevision = Math.max(savedRevision, version);
    if (revision === version) $('storageStatus').textContent = '已自动保存到此浏览器 · 请导出 JSON 备份';
  } catch (error) {
    $('storageStatus').textContent = `本地保存失败：${error.message}。请立即导出 JSON。`;
  }
}
window.addEventListener('beforeunload', event => {
  if (revision !== savedRevision || busy) { event.preventDefault(); event.returnValue = ''; }
});

function waitMedia(eventName, action) {
  return new Promise((resolve, reject) => {
    const cleanup = () => { clearTimeout(timer); video.removeEventListener(eventName, done); video.removeEventListener('error', fail); };
    const done = () => { cleanup(); resolve(); };
    const fail = () => { cleanup(); reject(new Error('视频解码失败，请使用浏览器支持的格式（如 H.264 MP4）')); };
    const timer = setTimeout(() => { cleanup(); reject(new Error('视频加载或定位超时')); }, MEDIA_TIMEOUT_MS);
    video.addEventListener(eventName, done, { once: true });
    video.addEventListener('error', fail, { once: true });
    try { action(); } catch (error) { cleanup(); reject(error); }
  });
}
async function seek(time) {
  const target = Math.max(0, Math.min(video.duration, time));
  if (Math.abs(video.currentTime - target) < 1e-6 && video.readyState >= 2 && !video.seeking) return;
  await waitMedia('seeked', () => { video.currentTime = target; });
}

$('videoFile').addEventListener('change', safe(async event => {
  const file = event.target.files[0]; event.target.value = '';
  if (!file) return;
  if (file.size > LIMITS.videoBytes || file.size === 0) throw new Error('视频必须介于 1 字节与 256 MiB 之间');
  if (project && !confirm('切换视频？当前项目将先保存到此浏览器；建议先导出 JSON 备份。')) return;
  setBusy(true); video.pause(); status('正在校验视频并查找本地项目…');
  const oldUrl = objectUrl;
  const oldTime = video.currentTime;
  let candidateUrl;
  try {
    clearTimeout(saveTimer); await persist();
    if (revision !== savedRevision) throw new Error('当前项目尚未保存，先导出 JSON 后再切换');
    const hash = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
    const sha256 = [...new Uint8Array(hash)].map(x => x.toString(16).padStart(2, '0')).join('');
    candidateUrl = URL.createObjectURL(file);
    await waitMedia('loadeddata', () => { video.src = candidateUrl; video.load(); });
    const fresh = createProject({ name: file.name, size: file.size, sha256,
      width: video.videoWidth, height: video.videoHeight, duration: video.duration });
    let restored;
    try { restored = await loadProject(sha256); }
    catch (error) { throw new Error(`无法读取本地项目：${error.message}`); }
    if (restored) {
      validateProject(restored);
      if (restored.video.sha256 !== sha256) throw new Error('本地项目视频哈希不匹配');
    }
    project = restored ?? fresh;
    objectUrl = candidateUrl; candidateUrl = null;
    if (oldUrl) URL.revokeObjectURL(oldUrl);
    history = new AnnotationHistory(); resetForm();
    $('empty').hidden = true; $('media').hidden = false;
    $('videoName').textContent = file.name;
    $('dimensions').textContent = `${video.videoWidth} × ${video.videoHeight}`;
    $('seek').max = video.duration;
    for (const id of ['start', 'end']) $(id).max = video.duration;
    fillMetadata(); render(); changed();
    status(restored ? '已恢复该视频的本地项目。' : '视频已就绪。可以直接标注，或先检测手部关键点。');
  } catch (error) {
    if (candidateUrl) {
      URL.revokeObjectURL(candidateUrl);
      if (oldUrl) {
        await waitMedia('loadeddata', () => { video.src = oldUrl; video.load(); });
        await seek(oldTime);
      } else { video.removeAttribute('src'); video.load(); }
    }
    throw error;
  } finally { setBusy(false); }
}));

function fillMetadata() {
  $('subject').value = project.subject; $('session').value = project.session; $('hz').value = project.sampling.hz;
}
for (const id of ['subject', 'session']) $(id).addEventListener('change', safe(() => {
  requireReady(); project[id] = $(id).value; changed();
}));
$('hz').addEventListener('change', safe(() => {
  requireReady(); project.sampling.hz = Number($('hz').value); shownFrame = undefined; changed(); renderTime();
}));

function resetForm() {
  editing = null; $('annotationForm').reset(); $('editMode').textContent = '新建'; $('saveAnnotation').textContent = '添加区间';
}
function commitAnnotations(next) {
  validateProject({ ...project, annotations: next });
  history.record(project.annotations); project.annotations = next;
  changed(); render();
}
function addAnnotation(a) {
  requireReady(); validateAnnotation(a, project.video.duration);
  const next = editing ? project.annotations.map(old => old.id === editing ? a : old) : [...project.annotations, a];
  commitAnnotations(next); resetForm(); status('区间已保存。');
}
$('annotationForm').addEventListener('submit', safe(event => {
  event.preventDefault();
  addAnnotation({ id: editing ?? crypto.randomUUID(), label: $('label').value.trim(),
    start: Number($('start').value), end: Number($('end').value), phase: $('phase').value,
    entities: $('entities').value === 'both' ? [...ENTITIES] : [$('entities').value],
    flags: FLAGS.filter(flag => $(flag).checked), note: $('note').value });
}));
$('resetAnnotation').addEventListener('click', resetForm);
function mark(id) { requireReady(); $(id).value = Math.min(Number(video.currentTime.toFixed(3)), project.video.duration); }
$('markStart').addEventListener('click', safe(() => mark('start')));
$('markEnd').addEventListener('click', safe(() => mark('end')));
for (const operation of ['undo', 'redo']) $(operation).addEventListener('click', safe(() => {
  requireReady(); project.annotations = history[operation](project.annotations); resetForm(); changed(); render();
}));
function button(label, action) {
  const b = document.createElement('button'); b.type = 'button'; b.textContent = label;
  b.addEventListener('click', safe(action)); return b;
}
function edit(a) {
  requireReady(); editing = a.id;
  for (const id of ['label', 'start', 'end', 'phase', 'note']) $(id).value = a[id];
  $('entities').value = a.entities.length === 2 ? 'both' : a.entities[0];
  for (const flag of FLAGS) $(flag).checked = a.flags.includes(flag);
  $('editMode').textContent = '编辑'; $('saveAnnotation').textContent = '保存修改'; $('label').focus();
}
function render() {
  $('count').textContent = project.annotations.length;
  $('annotations').replaceChildren(); $('timeline').replaceChildren();
  for (const a of [...project.annotations].sort((a, b) => a.start - b.start)) {
    const li = document.createElement('li'); li.className = 'annotation';
    const title = document.createElement('strong'); title.textContent = a.label;
    const detail = document.createElement('p'); detail.className = 'muted';
    detail.textContent = `${a.start.toFixed(3)}–${a.end.toFixed(3)} s · ${a.entities.join(' + ')} · ${a.phase}${a.flags.length ? ' · ' + a.flags.join(', ') : ''}`;
    const actions = document.createElement('div'); actions.className = 'actions';
    actions.append(button('定位', async () => { requireReady(); video.pause(); await seek(a.start); }),
      button('编辑', () => edit(a)), button('删除', () => {
        requireReady(); if (!confirm(`删除区间「${a.label}」？可通过撤销恢复。`)) return;
        commitAnnotations(project.annotations.filter(item => item.id !== a.id));
        if (editing === a.id) resetForm();
      }));
    li.append(title, detail, actions); $('annotations').append(li);
    const bar = button('', async () => { requireReady(); video.pause(); await seek(a.start); });
    bar.className = 'timeline-item'; bar.setAttribute('aria-label', `${a.label}，${a.start} 到 ${a.end} 秒，点击定位`);
    const canvas = document.createElement('canvas'); canvas.width = 1000; canvas.height = 28;
    const ctx = canvas.getContext('2d'); ctx.fillStyle = '#365747';
    ctx.fillRect(a.start / project.video.duration * canvas.width, 0, (a.end - a.start) / project.video.duration * canvas.width, canvas.height);
    const caption = document.createElement('span'); caption.textContent = `${a.label} · ${a.start.toFixed(2)}–${a.end.toFixed(2)}`;
    bar.append(canvas, caption); $('timeline').append(bar);
  }
  if (!project.annotations.length) { const li = document.createElement('li'); li.className = 'muted'; li.textContent = '尚无标注。设置起止时间后添加。'; $('annotations').append(li); }
  const assigned = project.frames.reduce((sum, f) => sum + f.hands.filter(h => h.entity !== null).length, 0);
  $('stats').textContent = `${project.frames.length} 个采样 · ${assigned} 个已绑定检测`;
  shownFrame = undefined; setBusy(busy); renderTime();
}

function renderTime() {
  if (!project) return;
  $('clock').textContent = `${video.currentTime.toFixed(3)} / ${project.video.duration.toFixed(3)} s`;
  $('seek').value = video.currentTime;
  const frame = nearestFrame(project.frames, video.currentTime, project.sampling.hz);
  if (shownFrame !== frame) { shownFrame = frame; renderHands(frame); }
  draw(frame);
}
function renderHands(frame) {
  $('hands').replaceChildren();
  $('frameTime').textContent = frame ? `${(frame.timestampUs / 1e6).toFixed(3)} s · 最近采样` : '无邻近采样';
  if (!frame || !frame.hands.length) {
    const p = document.createElement('p'); p.className = 'muted';
    p.textContent = frame ? '此采样未检测到手。可人工标注遮挡或负样本。' : '该位置尚无检测。检测结果顺序不代表稳定身份。';
    $('hands').append(p); return;
  }
  frame.hands.forEach((hand, index) => {
    const row = document.createElement('div'); row.className = 'hand';
    const info = document.createElement('div');
    const title = document.createElement('strong'); title.textContent = `检测 ${index + 1} · ${hand.handedness}`;
    const hint = document.createElement('p'); hint.className = 'muted'; hint.textContent = `左右手分类分数 ${hand.handednessScore.toFixed(3)}（非关键点置信度）`;
    info.append(title, hint);
    const label = document.createElement('label'); label.textContent = '此采样的身份';
    const select = document.createElement('select');
    for (const [value, name] of [['', '未校对'], ...ENTITIES.map(x => [x, x])]) {
      const option = document.createElement('option'); option.value = value; option.textContent = name; select.append(option);
    }
    select.value = hand.entity ?? ''; select.disabled = busy;
    select.addEventListener('change', safe(() => {
      requireReady(); const entity = select.value || null;
      if (entity && frame.hands.some((other, i) => i !== index && other.entity === entity)) {
        select.value = hand.entity ?? ''; throw new Error('该身份已分配给此采样中的另一只手');
      }
      hand.entity = entity; changed(); render();
    }));
    label.append(select); row.append(info, label); $('hands').append(row);
  });
}
function draw(frame) {
  const canvas = $('overlay'); canvas.width = video.clientWidth; canvas.height = video.clientHeight;
  const ctx = canvas.getContext('2d'); if (!frame) return;
  const scale = Math.min(canvas.width / project.video.width, canvas.height / project.video.height);
  const width = project.video.width * scale, height = project.video.height * scale;
  const ox = (canvas.width - width) / 2, oy = (canvas.height - height) / 2;
  frame.hands.forEach((hand, index) => {
    const points = hand.landmarks.map(([x, y]) => [ox + x * width, oy + y * height]);
    ctx.strokeStyle = ctx.fillStyle = colors[index]; ctx.lineWidth = 2;
    for (const [a, b] of edges) { ctx.beginPath(); ctx.moveTo(...points[a]); ctx.lineTo(...points[b]); ctx.stroke(); }
    for (const [x, y] of points) { ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill(); }
    ctx.font = '14px system-ui'; ctx.fillText(`${index + 1}: ${hand.entity ?? '?'}`, points[0][0] + 8, points[0][1]);
  });
}
video.addEventListener('timeupdate', renderTime);
video.addEventListener('seeked', renderTime);
video.addEventListener('play', () => { $('play').textContent = '暂停'; });
video.addEventListener('pause', () => { $('play').textContent = '播放'; });
new ResizeObserver(renderTime).observe($('media'));
async function togglePlay() { requireReady(); if (video.paused) await video.play(); else video.pause(); }
$('play').addEventListener('click', safe(togglePlay));
$('seek').addEventListener('input', safe(() => { requireReady(); video.pause(); video.currentTime = Number($('seek').value); }));
async function step(direction) { requireReady(); video.pause(); await seek(video.currentTime + direction / project.sampling.hz); }
$('back').addEventListener('click', safe(() => step(-1)));
$('forward').addEventListener('click', safe(() => step(1)));
document.addEventListener('keydown', safe(async event => {
  if (event.target.closest('input,select,textarea,button,summary,[contenteditable]') || event.ctrlKey || event.metaKey || event.altKey || !project || busy) return;
  const action = { ' ': togglePlay, ArrowLeft: () => step(-1), ArrowRight: () => step(1), i: () => mark('start'), o: () => mark('end') }[event.key];
  if (action) { event.preventDefault(); await action(); }
}));

function stopWorker(reason) {
  worker?.terminate(); worker = null;
  if (pending) { const p = pending; pending = null; clearTimeout(p.timer); p.reject(new Error(reason)); }
}
async function infer() {
  const bitmap = await createImageBitmap(video);
  if (cancelled) { bitmap.close(); throw new Error('已取消采样'); }
  if (!worker) {
    worker = new Worker('/inference-worker.mjs');
    worker.onmessage = ({ data }) => {
      if (!pending || data.id !== pending.id) return;
      const p = pending; pending = null; clearTimeout(p.timer);
      if (data.error) p.reject(new Error(`MediaPipe：${data.error}`)); else p.resolve(data.hands);
    };
    worker.onerror = event => { event.preventDefault(); stopWorker(`推理 Worker 加载失败：${event.message}`); };
  }
  return new Promise((resolve, reject) => {
    const id = ++requestId;
    pending = { id, resolve, reject, timer: setTimeout(() => stopWorker('推理超时，请检查网络或浏览器兼容性'), WORKER_TIMEOUT_MS) };
    try { worker.postMessage({ id, bitmap }, [bitmap]); }
    catch (error) { bitmap.close(); stopWorker(error.message); }
  });
}
async function analyze(all) {
  requireReady(); video.pause();
  const times = all ? sampleTimes(project.video.duration, project.sampling.hz) : [video.currentTime];
  const oldTime = video.currentTime;
  if (all && project.frames.length && !confirm('重新采样会替换全部检测和逐采样身份绑定；人工动作区间不变。继续？')) return;
  if (!all && project.frames.some(f => f.timestampUs === Math.round(oldTime * 1e6)) &&
    !confirm('覆盖当前采样的检测和身份绑定？人工动作区间不变。')) return;
  setBusy(true); cancelled = false; $('cancel').hidden = false; $('progress').hidden = false; $('progress').value = 0;
  const frames = [];
  try {
    for (let i = 0; i < times.length; i++) {
      if (cancelled) throw new Error('已取消采样');
      status(`采样 ${i + 1} / ${times.length}。首次检测需下载模型；可取消，原数据会保留。`);
      await seek(times[i]);
      if (cancelled) throw new Error('已取消采样');
      const timestampUs = Math.round(video.currentTime * 1e6);
      const hands = await infer();
      if (cancelled) throw new Error('已取消采样');
      frames.push({ timestampUs, hands }); $('progress').value = (i + 1) / times.length;
    }
    const next = all ? frames : upsertFrame(project.frames, frames[0]);
    validateProject({ ...project, frames: next });
    project.frames = next; changed();
    status(`完成 ${frames.length} 个采样。请校对身份后导出；左右手分类不是跟踪 ID。`);
  } catch (error) { status(`${error.message}。本次结果未提交，原有检测和标注已保留。`, !cancelled); }
  finally {
    $('cancel').hidden = true; $('progress').hidden = true;
    try { await seek(oldTime); } finally { setBusy(false); render(); }
  }
}
$('detect').addEventListener('click', safe(() => analyze(false)));
$('analyze').addEventListener('click', safe(() => analyze(true)));
$('cancel').addEventListener('click', () => { cancelled = true; stopWorker('已取消采样'); status('正在取消，原有数据将保留…'); });

$('jsonFile').addEventListener('change', safe(async event => {
  const file = event.target.files[0]; event.target.value = ''; if (!file) return;
  requireReady();
  if (file.size > LIMITS.jsonBytes) throw new Error('JSON 最大 48 MiB');
  setBusy(true);
  try {
    const imported = parseProject(await file.text());
    if (imported.video.sha256 !== project.video.sha256 || imported.video.size !== project.video.size ||
      imported.video.width !== project.video.width || imported.video.height !== project.video.height ||
      Math.abs(imported.video.duration - project.video.duration) > 0.001) throw new Error('标注文件不属于当前视频');
    if (!confirm('导入将替换当前视频的标注和检测。请确认已有 JSON 备份。')) return;
    project = imported; history = new AnnotationHistory(); resetForm(); fillMetadata(); changed(); render(); status('项目导入成功。');
  } finally { setBusy(false); }
}));
$('export').addEventListener('click', safe(() => {
  requireReady(); validateProject(project);
  const json = JSON.stringify(project);
  const blob = new Blob([json], { type: 'application/json' });
  if (blob.size > LIMITS.jsonBytes) throw new Error('导出体积超过可导入上限');
  const url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = url; a.download = `${project.video.name}.annotations.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000); status('JSON 下载已发起。请确认文件已保存；视频文件不包含在项目中。');
}));

// Optional WebMCP exposes the same validated actions and state, never a separate model.
if (document.modelContext?.registerTool) {
  document.modelContext.registerTool({ name: 'gesture_project_summary', description: '读取当前本地动作标注项目摘要，不含视频或关键点数据',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true, untrustedContentHint: true },
    execute: async () => ({ content: [{ type: 'text', text: JSON.stringify(project ? {
      video: project.video, subject: project.subject, session: project.session,
      annotations: project.annotations, samples: project.frames.length, busy,
    } : { loaded: false }) }] }) });
}
