let landmarker;
async function initialize() {
  // Classic worker is intentional: the pinned Emscripten WASM loader uses importScripts.
  const { MODEL, LIMITS } = await import('./domain.mjs');
  const root = `https://cdn.jsdelivr.net/npm/${MODEL.package}@${MODEL.version}`;
  const { FilesetResolver, HandLandmarker } = await import(`${root}/vision_bundle.mjs`);
  const vision = await FilesetResolver.forVisionTasks(`${root}/wasm`);
  landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL.url, delegate: MODEL.delegate },
    runningMode: MODEL.runningMode, numHands: LIMITS.hands,
    minHandDetectionConfidence: MODEL.detectionConfidence,
    minHandPresenceConfidence: MODEL.presenceConfidence,
  });
}
self.onmessage = async ({ data }) => {
  try {
    if (!landmarker) await initialize();
    // IMAGE mode intentionally avoids temporal state contamination when seeking backwards.
    const result = landmarker.detect(data.bitmap);
    const points = list => list.map(({ x, y, z }) => [x, y, z]);
    const hands = result.landmarks.map((landmarks, i) => ({
      entity: null, landmarks: points(landmarks), worldLandmarks: points(result.worldLandmarks[i]),
      handedness: result.handedness[i][0].categoryName,
      handednessScore: result.handedness[i][0].score,
    }));
    self.postMessage({ id: data.id, hands });
  } catch (error) { self.postMessage({ id: data.id, error: error.message }); }
  finally { data.bitmap?.close(); }
};
