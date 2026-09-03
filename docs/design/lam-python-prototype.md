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

