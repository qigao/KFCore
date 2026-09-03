# LAM Python Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce measured evidence that the released LAM-Audio2Expression and LAM inference paths can run without training on this Windows RTX 4060 Laptop system.

**Architecture:** Keep all heavyweight upstream code, environments, weights, and generated outputs under ignored `.cache/lam-prototype/`. Add a small repository-owned Python probe that validates the environment and emits a stable JSON report, then execute the official A2E path before attempting LAM.

**Tech Stack:** Python 3.10, PyTorch/CUDA versions pinned by each upstream project, pytest for the repository-owned probe, official LAM and LAM-Audio2Expression sources.

**Spec:** `docs/design/lam-python-prototype.md`

## Global Constraints

- No model training or identity fine-tuning.
- Do not modify the root CMake dependency migration.
- Do not commit upstream repositories, environments, checkpoints, or generated media.
- Fail fast on dependency, checkpoint, CUDA, input, or output-contract errors.
- Measure before making performance claims.

---

### Task 1: Reproducible environment probe

**Files:**
- Create: `tools/lam_python_probe/preflight.py`
- Create: `tools/lam_python_probe/tests/test_preflight.py`
- Create: `tools/lam_python_probe/README.md`

**Interfaces:**
- Consumes: optional `--lam-root`, `--a2e-root`, and `--output` paths.
- Produces: `collect_preflight(lam_root: Path, a2e_root: Path) -> dict[str, object]` and a JSON report.

- [ ] **Step 1: Write a failing test for stable repository and runtime fields**

```python
def test_collect_preflight_reports_required_fields(tmp_path):
    lam = tmp_path / "LAM"
    a2e = tmp_path / "LAM_Audio2Expression"
    lam.mkdir()
    a2e.mkdir()
    report = collect_preflight(lam, a2e)
    assert report["schema_version"] == 1
    assert report["python"]["version"]
    assert report["upstreams"]["lam"]["path"] == str(lam.resolve())
    assert report["upstreams"]["a2e"]["path"] == str(a2e.resolve())
```

- [ ] **Step 2: Run the test and verify it fails because `preflight` does not exist**

Run: `python -m pytest tools/lam_python_probe/tests/test_preflight.py -q`

Expected: import failure for `tools.lam_python_probe.preflight`.

- [ ] **Step 3: Implement the minimal preflight collector and JSON CLI**

The collector must use standard-library APIs for Python, platform, paths, subprocess execution, and disk space. PyTorch and `nvidia-smi` are optional observations represented by explicit availability and error fields; their absence must not crash preflight collection.

- [ ] **Step 4: Add failing tests for nonexistent upstream roots and atomic JSON output**

The tests must assert that a missing root raises `ValueError` and that the CLI writes a valid report only after collection succeeds.

- [ ] **Step 5: Implement validation and atomic report replacement**

Write the report to a sibling temporary file, close it, then replace the destination. Reject an existing directory as the output path.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tools/lam_python_probe/tests/test_preflight.py -q`

Expected: all preflight tests pass.

- [ ] **Step 7: Commit**

```bash
git add tools/lam_python_probe
git commit -m "feat: add LAM runtime preflight probe"
```

### Task 2: Isolated LAM-A2E environment and smoke inference

**Files:**
- Modify: `docs/design/lam-python-prototype.md`
- Create outside Git: `.cache/lam-prototype/envs/lam-a2e/`
- Create outside Git: `.cache/lam-prototype/models/lam-a2e/`
- Create outside Git: `.cache/lam-prototype/results/a2e/`

**Interfaces:**
- Consumes: official sample WAV and `lam_audio2exp_streaming.tar`.
- Produces: official expression output plus measured cold-start, inference, frame-count, and CUDA-memory evidence.

- [ ] **Step 1: Create a Python 3.10 prefix environment**

Run: `conda.exe create -p .cache/lam-prototype/envs/lam-a2e python=3.10 -y`

Expected: the prefix interpreter reports Python 3.10.

- [ ] **Step 2: Install the upstream CUDA 12.1 dependency set**

Install PyTorch 2.1.2, torchvision 0.16.2, torchaudio 2.1.2, then the exact A2E requirements. Do not reuse the LAM environment because LAM pins PyTorch 2.3.0.

- [ ] **Step 3: Run preflight in the A2E environment**

Expected: `torch_available=true`, `cuda_available=true`, and the detected GPU is recorded.

- [ ] **Step 4: Download and verify the official A2E assets**

Download `3DAIGC/LAM_audio2exp` into `.cache/lam-prototype/models/lam-a2e/`, retain the upstream revision metadata, and verify every required archive exists and is non-empty before extraction.

- [ ] **Step 5: Run official streaming inference on its sample audio**

Invoke the upstream `inference.py` with `configs/lam_audio2exp_config_streaming.py`, an explicit checkpoint path, explicit audio path, and an output directory under `.cache/lam-prototype/results/a2e/`.

- [ ] **Step 6: Validate outputs and record measurements**

Reject missing outputs, zero frames, any frame other than 52 values, non-finite values, or values outside the model's sigmoid output interval `[0, 1]`. Record the exact failure if the official output representation differs and update the adapter contract only from observed evidence.

- [ ] **Step 7: Commit the measured report**

```bash
git add docs/design/lam-python-prototype.md
git commit -m "docs: record LAM-A2E feasibility measurements"
```

### Task 3: LAM model-load and single-image feasibility gate

**Files:**
- Modify: `docs/design/lam-python-prototype.md`
- Create outside Git: `.cache/lam-prototype/envs/lam/`
- Create outside Git: `.cache/lam-prototype/models/lam/`
- Create outside Git: `.cache/lam-prototype/results/lam/`

**Interfaces:**
- Consumes: official LAM-20K checkpoint, LAM assets, and a repository-external consented portrait or upstream sample.
- Produces: either an avatar/preview with timing and peak-memory evidence, or a reproducible fail-fast compatibility result.

- [ ] **Step 1: Create a separate Python 3.10 LAM environment**

Install PyTorch 2.3.0, torchvision 0.18.0, torchaudio 2.3.0, xformers 0.0.26.post1, and the Windows-compatible upstream dependencies.

- [ ] **Step 2: Verify native CUDA extensions independently**

Import `pytorch3d`, `nvdiffrast`, `diff_gaussian_rasterization`, and `simple_knn` one at a time. Stop at the first failed import and record compiler, CUDA, and package versions.

- [ ] **Step 3: Download official assets and LAM-20K weights**

Verify expected assets and checkpoint files before loading; keep all downloads outside Git.

- [ ] **Step 4: Measure model construction and checkpoint load**

Record process time and CUDA allocated/reserved memory. If this stage raises CUDA out-of-memory, do not attempt image inference.

- [ ] **Step 5: Run one official single-image inference**

Use the official preprocessing and inference path with batch size one and its configured resolution. Do not add CPU offload or lower precision unless separately authorized and measured as a distinct configuration.

- [ ] **Step 6: Validate avatar and render output**

Require a non-empty canonical Gaussian representation, finite Gaussian attributes, a non-empty rendered image/video, and an explicit timing/memory report.

- [ ] **Step 7: Commit the measured report**

```bash
git add docs/design/lam-python-prototype.md
git commit -m "docs: record LAM single-image feasibility"
```

### Task 4: Decision and integration boundary

**Files:**
- Modify: `docs/design/lam-python-prototype.md`

**Interfaces:**
- Consumes: measured A2E and LAM results.
- Produces: a proceed/stop decision for a production Python service and its KFCore boundary.

- [ ] **Step 1: Compare measured stages**

Separate model load, preprocessing, inference, rendering, encoding, and Python orchestration times. Do not attribute CUDA time to Python overhead.

- [ ] **Step 2: State the architecture decision**

Choose a persistent Python service only if the official inference path succeeds within the measured memory budget. Otherwise stop and document which hardware, dependency, or model constraint failed.

- [ ] **Step 3: Run repository-owned probe tests and inspect Git scope**

Run: `python -m pytest tools/lam_python_probe/tests/test_preflight.py -q`

Run: `git status --short`

Expected: tests pass; only the probe and design/plan documentation are tracked.

- [ ] **Step 4: Commit**

```bash
git add docs/design/lam-python-prototype.md docs/superpowers/plans/2026-09-03-lam-python-prototype.md
git commit -m "docs: conclude LAM Python prototype"
```

