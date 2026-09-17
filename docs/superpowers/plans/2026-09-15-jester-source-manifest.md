# Jester Source Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-fast CLI that selects compatible clips from an already-downloaded Jester dataset into an auditable KFCore five-class source manifest rooted at `F:\KFCoreDatasets\temporal_gesture`.

**Architecture:** Keep public-dataset selection separate from production feature extraction. The new module parses Jester metadata, maps only exact compatible labels, verifies every selected frame directory, and writes JSONL records that retain source provenance while explicitly declaring that subject identity is unavailable; a later production `HandDetector -> HandTracker` pass remains responsible for creating trainable 78-D frame records and phase labels.

**Tech Stack:** Python 3.10+, standard-library `argparse`, `csv`, `json`, `pathlib`, and `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-15-temporal-gesture-gru-v1-design.md`

## Global Constraints

- Gesture labels remain exactly `0 none, 1 swipe_left, 2 swipe_right, 3 grab, 4 release` under `kfcore-temporal-gesture-classes/1`.
- Public-dataset clips are not trainable JSONL until they pass through the production `HandDetector -> HandTracker` path and receive consistent phase annotations.
- Jester subject identities are unavailable in its public split metadata, so generated records must use `subject_id: null` and `usage: pretrain_only`; no synthetic subject IDs or subject-separated quality claims are allowed.
- Only exact Jester labels `Swiping Left`, `Swiping Right`, `No gesture`, and `Doing other things` are selected; all other source labels are counted as skipped and never silently remapped.
- The CLI consumes only local files. It does not log in, accept dataset terms, or download licensed data.
- Existing output files are rejected rather than overwritten.

---

### Task 1: Jester selection and validation core

**Files:**
- Create: `tools/temporal_gesture/label_contract.py`
- Create: `tools/temporal_gesture/prepare_dataset.py`
- Create: `tools/temporal_gesture/test_prepare_dataset.py`
- Modify: `tools/temporal_gesture/model.py`

**Interfaces:**
- Consumes: Jester semicolon-delimited rows `source_sample_id;source_label` and a frame root containing one directory per sample.
- Produces: dependency-free label constants in `label_contract.py`, plus `SourceSample`, `SelectionResult`, and `select_jester_samples(labels_csv: Path, frames_root: Path, source_split: str) -> SelectionResult`.

- [x] **Step 1: Write failing mapping and provenance tests**

```python
def test_selects_only_compatible_jester_labels(self) -> None:
    result = select_jester_samples(labels_csv, frames_root, "train")
    self.assertEqual([sample.target_gesture_label for sample in result.samples], [0, 0, 1, 2])
    self.assertTrue(all(sample.subject_id is None for sample in result.samples))
    self.assertTrue(all(sample.usage == "pretrain_only" for sample in result.samples))
    self.assertEqual(result.skipped_by_label, {"Thumb Up": 1})
```

- [x] **Step 2: Run the focused test and confirm the module is absent**

Run from `tools/temporal_gesture`: `python -m unittest test_prepare_dataset.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'prepare_dataset'`.

- [x] **Step 3: Implement exact label mapping and complete pre-write validation**

```python
MANIFEST_SCHEMA = "kfcore-gesture-source-manifest/1"
JESTER_LABEL_MAP = {
    "No gesture": (0, "none", "hard_negative"),
    "Doing other things": (0, "none", "hard_negative"),
    "Swiping Left": (1, "swipe_left", "positive"),
    "Swiping Right": (2, "swipe_right", "positive"),
}

def select_jester_samples(
    labels_csv: Path, frames_root: Path, source_split: str
) -> SelectionResult:
    """Validate all selected samples before returning an immutable result."""
```

Reject empty or path-like sample IDs, duplicate IDs, empty split names, missing directories, and selected directories without `.jpg`, `.jpeg`, or `.png` frames. Preserve the original label, absolute source path, frame count, target label/name, role, `subject_id=None`, and `usage="pretrain_only"`; count unselected labels without mapping them.

- [x] **Step 4: Run the focused tests**

Run from `tools/temporal_gesture`: `python -m unittest test_prepare_dataset.py -v`

Expected: PASS for exact mapping, skipped-label accounting, duplicate-ID rejection, invalid-ID rejection, missing-frame rejection, and deterministic source-ID ordering.

### Task 2: Atomic manifest CLI and F-drive default

**Files:**
- Modify: `tools/temporal_gesture/prepare_dataset.py`
- Modify: `tools/temporal_gesture/test_prepare_dataset.py`

**Interfaces:**
- Consumes: `write_manifest(result: SelectionResult, output: Path) -> None` and CLI arguments `--labels`, `--frames-root`, `--source-split`, optional `--dataset-root`.
- Produces: JSONL at `<dataset-root>/manifests/jester-<source-split>.jsonl` and a JSON summary on stdout.

- [x] **Step 1: Write failing serialization and no-overwrite tests**

```python
def test_write_manifest_preserves_contract_and_refuses_overwrite(self) -> None:
    write_manifest(result, output)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    self.assertEqual(records[0]["manifest_schema"], MANIFEST_SCHEMA)
    self.assertEqual(records[0]["gesture_label_contract"], GESTURE_LABEL_CONTRACT)
    self.assertEqual(records[0]["subject_id"], None)
    with self.assertRaisesRegex(FileExistsError, "already exists"):
        write_manifest(result, output)
```

- [x] **Step 2: Run the serialization test and verify it fails**

Run from `tools/temporal_gesture`: `python -m unittest test_prepare_dataset.JesterManifestTests.test_write_manifest_preserves_contract_and_refuses_overwrite -v`

Expected: FAIL because `write_manifest` is not implemented.

- [x] **Step 3: Implement prevalidated JSONL writing and CLI defaults**

```python
DEFAULT_DATASET_ROOT = Path(r"F:\KFCoreDatasets\temporal_gesture")

def write_manifest(result: SelectionResult, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    # Write every validated SourceSample, flush and fsync, then replace the absent output.
```

The CLI derives `jester-<source-split>.jsonl`, validates before opening the temporary file, removes only its own temporary file after a write failure, and prints selected/skipped counts. It must not emit landmarks, phase labels, or claim the manifest is ready for `train.py`.

- [x] **Step 4: Run the complete preparation test module**

Run from `tools/temporal_gesture`: `python -m unittest test_prepare_dataset.py -v`

Expected: PASS.

### Task 3: Document the two-stage public-data workflow and run regression checks

**Files:**
- Modify: `tools/temporal_gesture/README.md`
- Modify: `docs/superpowers/plans/2026-09-15-jester-source-manifest.md`

**Interfaces:**
- Consumes: `prepare_dataset.py` CLI from Task 2.
- Produces: A copy-pasteable command for the F-drive layout and an explicit boundary between source manifest creation and trainable KFCore JSONL extraction.

- [x] **Step 1: Add the documented invocation and limitations**

```text
python tools/temporal_gesture/prepare_dataset.py ^
  --labels F:\KFCoreDatasets\temporal_gesture\raw\jester\jester-v1-train.csv ^
  --frames-root F:\KFCoreDatasets\temporal_gesture\raw\jester\20bn-jester-v1 ^
  --source-split train
```

State that Jester contributes swipe positives and hard negatives only, cannot supply grab/release, has no public subject IDs for leakage-safe final evaluation, and still requires production landmark/tracker extraction plus phase annotation.

- [x] **Step 2: Run the new tests and existing temporal-gesture contract tests**

Run from `tools/temporal_gesture`: `python -m unittest test_prepare_dataset.py test_contract.py -v`

Expected: all tests PASS.

- [x] **Step 3: Exercise `--help` and an invalid empty raw directory**

Run: `python tools/temporal_gesture/prepare_dataset.py --help`

Expected: exit code 0 and all required arguments documented.

Run: `python tools/temporal_gesture/prepare_dataset.py --labels F:\KFCoreDatasets\temporal_gesture\raw\jester\missing.csv --frames-root F:\KFCoreDatasets\temporal_gesture\raw\jester\missing --source-split train`

Expected: non-zero exit with the missing labels path; no manifest created.

- [x] **Step 4: Mark completed plan checkboxes and inspect the scoped diff**

Run: `git diff --check -- tools/temporal_gesture/prepare_dataset.py tools/temporal_gesture/test_prepare_dataset.py tools/temporal_gesture/README.md docs/superpowers/plans/2026-09-15-jester-source-manifest.md`

Expected: no whitespace errors.
