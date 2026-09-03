# LAM Python Prototype Design

## Objective

Validate a no-training path from audio to LAM-compatible facial motion and from one portrait to a LAM Gaussian avatar before integrating either model into KFCore.

The prototype is deliberately external to the KFCore CMake graph. Upstream repositories, Python environments, model weights, and generated media live under the ignored `.cache/lam-prototype/` tree. Only the reproducible probe, tests, and measured report belong to this branch.

## Boundaries

- Use the released LAM and LAM-Audio2Expression inference code and checkpoints without training or identity fine-tuning.
- Keep LAM and LAM-Audio2Expression in separate Python 3.10 environments because their official install scripts pin PyTorch 2.3.0 and 2.1.2 respectively.
- Run on the available RTX 4060 Laptop GPU with 8 GiB VRAM. An out-of-memory result is a measured outcome, not a reason to add an undocumented CPU fallback.
- Do not modify the root CMake dependency migration. The current branch baseline cannot configure because its committed TurboUtils release root is absent.
- Do not vendor upstream source code or model weights into KFCore.
- Treat published model weights as non-commercial research assets until their applicable licenses are reviewed for the intended deployment.

## Data Flow

```text
sample WAV -> LAM-A2E -> ARKit expression sequence -> metrics report

sample portrait -> LAM -> canonical Gaussian avatar -> preview/video -> metrics report

ARKit expression sequence -> [route A: exported ARKit-rigged LAM avatar]
                          -> runtime renderer (not yet validated here)

ARKit expression sequence -> [route B: ARKit52-to-FLAME retargeter]
                          -> offline LAM renderer (not implemented upstream)
```

The first integration target is LAM-A2E because it has the smaller download and isolates audio preprocessing, model loading, CUDA inference, and output timing. LAM is attempted only after the A2E environment and measurement path are reproducible.

## Measurements

Each runnable stage records:

- environment and upstream commit;
- cold model-load time;
- first inference time;
- repeated warm inference time when supported;
- input duration and generated motion-frame count;
- peak allocated and reserved CUDA memory;
- total GPU memory before and after the process;
- output paths and failure stage.

## Failure Semantics

The repository-owned probes terminate with a non-zero exit code for missing
files, malformed outputs, incompatible dependencies, CUDA errors, and
out-of-memory failures. They do not silently change device, precision,
resolution, or model. The upstream LAM application does not provide this
guarantee: its UI callback catches broad exceptions and can return process exit
code zero without an output, so an integration adapter must validate the output
contract independently.

## Acceptance Gates

1. A preflight command reports Python, PyTorch, CUDA, GPU, upstream revisions, disk space, and checkpoint presence as machine-readable JSON.
2. The A2E official sample produces a finite 52-value expression vector for every frame and a report with timing and CUDA memory.
3. Only after gate 2 passes, the LAM model-load probe runs; single-image inference is attempted only if loading leaves enough GPU memory for the official path.
4. No generated environment, checkpoint, upstream checkout, or media file appears in `git status`.

## Measured A2E Result (2026-09-03)

Evidence was collected from the official
[LAM-Audio2Expression](https://github.com/3DAIGC/LAM_Audio2Expression) checkout at
commit `02a703c3ea7d8e360eb43098eca85ee98a083529`.

### Environment and assets

- **Fact:** Python 3.10.21, PyTorch 2.1.2+cu121, CUDA runtime 12.1, and the RTX
  4060 Laptop GPU completed a real CUDA tensor operation. `pip check` reported
  no broken requirements.
- **Fact:** `LAM_audio2exp_assets.tar` SHA-256 is
  `f5759f080ec1352d454d88d5a32f09c0fc8aeaf50b9ac1fd9c34a4acd82acfa1`.
- **Fact:** `LAM_audio2exp_streaming.tar` SHA-256 is
  `0877f419b1b7b6856479b743b71625ad478f97434404e91288f9f0d6d516b97a`.
- **Fact:** The loaded streaming model reports 97,912,596 trainable parameters.

### Timing and memory

The reproducible run explicitly set `ex_vol=False`; this prevents the upstream
configuration from invoking an undeclared `spleeter` executable and then
silently continuing with the original audio.

- **Fact:** A clean process took 10.075 seconds from interpreter launch through
  model construction, checkpoint load, inference, JSON export, and shutdown.
- **Fact:** An in-process instrumented run took 4.671 seconds after importing
  PyTorch. The upstream log interval from model construction through checkpoint
  load was 1.556 seconds; the model forward interval was 0.317 seconds.
- **Fact:** Peak CUDA allocated memory was 554,708,992 bytes (529.0 MiB); peak
  reserved memory was 765,460,480 bytes (730.0 MiB).
- **Calculation:** For the 7.9953125-second WAV, forward real-time factor was
  `0.317 / 7.9953125 = 0.0396`, or about 25.2 times faster than real time. This
  excludes process startup, model loading, JSON post-processing, and rendering.

### Output contract

- **Fact:** The sample produced 240 frames at 30 fps; expected frame count was
  `ceil(7.9953125 * 30) = 240`.
- **Fact:** Every frame contained 52 finite expression weights. Across the
  output, the observed range was `[0.0, 0.9513577222824097]`.
- **Fact:** The repository-owned validator passed all contract checks. Its unit
  tests cover frame width, non-finite values, range violations, and audio/frame
  count mismatch.

### Gate decision

**Inference:** A persistent Python A2E worker is feasible on this machine. The
0.317-second forward time is small relative to an 8-second clip, while the
10.075-second one-shot process time shows why launching Python per utterance is
the wrong integration boundary. Gate 2 passes, so the separate LAM environment
and single-image model-load gate may proceed.

**MED:** The upstream default `ex_vol=True` depends on `spleeter`, which is not
declared in its requirements. Its failed shell command does not fail inference.
KFCore integration must either require a validated vocal stem or explicitly
disable separation; it must not inherit this silent fallback.

## Measured LAM Result (2026-09-03)

Evidence was collected with the official Windows one-click archive linked by
the [LAM repository](https://github.com/aigc3d/LAM). The archive SHA-256 is
`81ca564f84df14db995868685af42d612b51b3c82204d2f6fcd366d45a4a1121`.
Its embedded Git metadata is incomplete, so the archive hash, rather than an
unverifiable commit identifier, is the reproducibility key.

### Environment and model load

- **Fact:** The bundled environment uses Python 3.10.11 and PyTorch 2.7.0+cu128.
  `pytorch3d`, `nvdiffrast.torch`, `diff_gaussian_rasterization`, `simple_knn._C`,
  and the FaceBoxes NMS extension all imported successfully when PyTorch was
  loaded first.
- **Fact:** The loaded LAM model contains 557,642,254 parameters. CPU model
  construction and checkpoint load took 17.590 seconds; transfer to CUDA took
  1.218 seconds, for 18.808 seconds total.
- **Fact:** Model-load peak CUDA allocated memory was 2,359,741,440 bytes
  (2.20 GiB), and peak reserved memory was 2,376,073,216 bytes (2.21 GiB).
- **Fact:** The first `nvdiffrast` CUDA context compilation succeeded with the
  CUDA 12.8 toolkit and `MAX_JOBS=1`, taking 303.815 seconds. Selecting the
  machine's CUDA 13.0 toolkit and parallel Ninja build failed with MSVC C1060
  on the 16 GiB host.

### Official single-image run

The run used the bundled `assets/sample_input/status.png`, the bundled
`Look_In_My_Eyes` motion, the official 512x512 FP32 path, and no training. A
small compatibility patch fixed Windows path handling and streamed only the
requested render tensors to CPU. It also converted the completed RGB tensor to
`uint8` in place before MoviePy encoding.

- **Fact:** The 519-frame LAM render interval took 17.556 seconds. The measured
  application interval, including model loading, tracking, preprocessing,
  FLAME fitting, rendering, and encoding, was 136.384 seconds; clean process
  wall time was 148.537 seconds.
- **Fact:** Within that run, tracking-model setup took 9.13 seconds, image
  preprocessing 3.67 seconds, FLAME optimization 49.22 seconds, and tracking
  export 1.93 seconds.
- **Fact:** Peak CUDA allocated memory was 4,190,469,120 bytes (3.90 GiB), and
  peak reserved memory was 4,674,551,808 bytes (4.35 GiB).
- **Fact:** `ffprobe` validated a 512x512 H.264 video at 30 fps with 520 encoded
  frames and duration 17.333 seconds. The muxed result also contains a
  17.300-second AAC stream. The silent and muxed files have SHA-256 values
  `358c5bd5347da33f8d236937c659146b59605cc7613179edc630530897acb64c`
  and `82a2a5939d5cc836e21d4eeb8cbe92b87720e0854dd17bdb211adde130ed3706`
  respectively.
- **Fact:** The renderer constructed a canonical Gaussian list in memory and
  logged 20,018 upsampled vertices. This run deliberately disabled PLY export;
  therefore the video path is validated, but finite-value validation of a
  persisted canonical Gaussian artifact remains open.

### Required compatibility patch

The unmodified 519-frame path accumulated all RGB, mask, depth, and per-frame
Gaussian tensors on the GPU, then failed while requesting another 1.52 GiB;
peak allocation had reached 5,971,208,192 bytes. After streaming selected
outputs to CPU, rendering completed, but NumPy's full-size `clip` and cast
copies exhausted host memory. The repository patch
`tools/lam_python_probe/patches/lam-one-click-windows.patch` addresses both
bounded-memory failures and the Windows path separator bugs without changing
the default behavior of `ModelLAM.infer_single_view` for existing callers.

**HIGH:** The upstream UI catches broad exceptions and returned exit code zero
for both the CUDA extension failure and the GPU/CPU out-of-memory failures.
KFCore must treat the Python worker's typed response and validated output as the
fact source; process exit status alone is insufficient.

**HIGH:** The stock renderer does not fit the tested 8 GiB GPU for a 519-frame
clip because output memory grows with frame count. The compatibility patch is
required unless the production worker renders bounded chunks and encodes them
incrementally.

**MED:** First-use extension compilation is a deployment step, not request
work. The worker image must pin CUDA 12.8, set `CUDA_HOME`/`CUDA_PATH`, restrict
the compile to `MAX_JOBS=1` on this host class, and warm the extension cache
before accepting traffic.

**MED:** The current MoviePy boundary still holds the final `uint8` RGB clip in
host memory. At 512x512 RGB its lower-bound payload is
`frames * 512 * 512 * 3` bytes (about 389 MiB for 519 frames). Production must
stream frames to FFmpeg so memory is bounded independently of utterance length.

## Integration Decision

**Decision: proceed with a persistent Python worker prototype, but do not yet
call it an end-to-end audio-driven LAM pipeline.** Python orchestration is not
the measured bottleneck: A2E forward took 0.317 seconds for 7.995 seconds of
audio, while LAM image/FLAME preparation and cold model loading dominate the
one-shot latency. Keeping both models resident avoids roughly 10 seconds of A2E
process startup and 19 seconds of LAM construction/load per request.

The proposed ownership boundary is:

```text
KFCore C++
  request validation / job state / cancellation / output publication
       |
       | versioned IPC messages; paths plus metadata, never Python objects
       v
persistent Python model worker
  identity prepare -> canonical LAM / ARKit export (cached, bounded, versioned)
  audio infer      -> ARKit52 sequence
  runtime animate  -> exported ARKit rig + Gaussian offsets
  frame stream     -> FFmpeg stdin -> atomic MP4 publication
```

- **Architecture:** KFCore remains the durable job/state owner. The Python
  worker owns loaded model and CUDA state only; restarting it must not advance a
  job or publish a partial result.
- **Interface:** Requests carry input paths, output path, model/config version,
  identity-cache key, and limits. Responses carry stage, typed error, timings,
  frame count, dimensions, hashes, and peak memory. Outputs are written to a
  temporary path and atomically renamed only after media validation.
- **Error semantics:** Missing models, invalid audio/image, CUDA failure, OOM,
  cancellation, and malformed output are distinct terminal errors. No CPU,
  lower-resolution, alternate-precision, or vocal-separation fallback is
  implicit.
- **Concurrency:** One worker owns one GPU and serializes LAM renders initially.
  A2E can be scheduled separately only after measurements show it does not
  perturb the renderer's memory budget.
- **Migration cost:** No KFCore CMake or public API change is part of this
  worktree. A later integration needs a versioned IPC schema, worker lifecycle,
  validated LAM avatar export/runtime adapter, chunked encoder, and
  contract/integration tests.

### Remaining blocking gap

**Fact:** LAM-A2E exports 52 named ARKit blendshape weights. The upstream LAM
repository also contains an avatar-export path that writes `offset.ply`, an
ARKit-rigged `skin.glb`, and a reference `animation.glb` for OpenAvatarChat. Its
documentation says the ARKit blendshapes were manually adapted to FLAME's
topology. This is the preferred bridge because it preserves the released A2E
contract and avoids inventing a 52-to-50 coefficient mapping.

**HIGH:** The ARKit avatar-export and runtime animation path has not yet been
executed or media-validated on this Windows machine. Until it passes, arbitrary
spoken sentences are not end-to-end proven. The alternative offline LAM runner
expects per-frame FLAME tensors (`expr`, `rotation`, `neck_pose`, `jaw_pose`,
`eyes_pose`, `translation`, and identity `betas`); the A2E README still lists a
FLAME-expression model as unreleased. Using that route would require a separate
retargeter with validation for mouth closure, jaw rotation, eye blinks, head
pose, temporal continuity, and identity preservation. This worktree proves the
two inference halves and the single-image video renderer separately; it does
not claim either bridge is complete.
