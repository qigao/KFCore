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

Missing files, incompatible Python/CUDA packages, malformed checkpoints, CUDA errors, and out-of-memory failures terminate the current stage with a non-zero exit code and a specific diagnostic. The probe does not silently change device, precision, resolution, or model.

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
