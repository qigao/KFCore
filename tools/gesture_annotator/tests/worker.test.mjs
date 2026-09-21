import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { runInNewContext } from 'node:vm';

const source = await readFile(new URL('../dist/inference-worker.mjs', import.meta.url), 'utf8');
function adapter(detect) {
  const messages = [];
  const context = { self: { postMessage: data => messages.push(data) }, fakeDetector: { detect } };
  // Isolate result/error adaptation from network, WASM and browser graphics capabilities.
  runInNewContext(`${source}\ninitialize = async () => { landmarker = fakeDetector; };`, context);
  return { receive: context.self.onmessage, messages };
}
test('worker maps raw results without inventing tracking identity or landmark confidence', async () => {
  const points = Array.from({ length: 21 }, () => ({ x: 0.1, y: 0.2, z: -0.01 }));
  const { receive, messages } = adapter(() => ({ landmarks: [points], worldLandmarks: [points],
    handedness: [[{ categoryName: 'Left', score: 0.95 }]] }));
  let closed = false;
  await receive({ data: { id: 17, bitmap: { close() { closed = true; } } } });
  assert.equal(messages[0].id, 17); assert.equal(messages[0].hands[0].entity, null);
  assert.equal(messages[0].hands[0].landmarks.length, 21);
  assert.equal(messages[0].hands[0].handednessScore, 0.95);
  assert.equal('landmarkConfidence' in messages[0].hands[0], false);
  assert.equal(closed, true);
});
test('worker propagates detector failure and releases bitmap', async () => {
  const { receive, messages } = adapter(() => { throw new Error('inference failed'); });
  let closed = false;
  await receive({ data: { id: 3, bitmap: { close() { closed = true; } } } });
  assert.equal(messages[0].error, 'inference failed'); assert.equal(messages[0].id, 3);
  assert.equal('hands' in messages[0], false); assert.equal(closed, true);
});
