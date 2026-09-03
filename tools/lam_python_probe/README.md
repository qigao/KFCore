# LAM Python runtime probe

This tool records the Python, PyTorch, CUDA, GPU, disk, and upstream revision
facts used by the LAM/LAM-Audio2Expression prototype. It does not install
packages, download model weights, or run inference.

Run its tests from the repository root:

```powershell
python -m pytest tools/lam_python_probe/tests/test_preflight.py
```

Collect a report:

```powershell
python tools/lam_python_probe/preflight.py `
  --lam-root .cache/lam-prototype/runtime/lam-one-click/LAM `
  --a2e-root .cache/lam-upstream/LAM_Audio2Expression `
  --output .cache/lam-prototype/results/preflight.json
```

The command fails when either upstream is missing its required entry points or
when the default LAM/A2E checkpoint is missing or empty. It records checkpoint
paths and sizes. Reports use schema version 1 and are written atomically.

Validate an A2E expression export against its source WAV:

```powershell
python tools/lam_python_probe/a2e_output.py `
  --expression .cache/lam-prototype/results/a2e/no-extraction/bsData.json `
  --audio .cache/lam-upstream/LAM_Audio2Expression/assets/sample_audio/BarackObama_english.wav
```

The validator requires a non-empty 30 fps sequence, exactly 52 finite weights
per frame, weights in `[0, 1]`, the official ordered ARKit52 names, consistent
metadata/time/rotation fields, and a frame count matching the WAV duration.

The command and measurement snapshot used for the official A2E and LAM samples
is committed as
[`docs/design/lam-python-prototype-results.json`](../../docs/design/lam-python-prototype-results.json).
The A2E command is non-interactive. The recorded LAM command launches the
official Gradio path; select `assets/sample_input/status.png` and
`Look_In_My_Eyes` to reproduce the recorded input pair. The report explicitly
marks stage timing instrumentation that was manually transcribed rather than
retained as an automated harness.

## LAM Windows one-click compatibility patch

The official CUDA 12.8 one-click archive needs a bounded-output patch to render
the bundled 519-frame sample on the tested 8 GiB GPU / 16 GiB host. Run these
commands from the KFCore worktree after setting the two placeholder paths:

```powershell
$repoRoot = git rev-parse --show-toplevel
$patchPath = Join-Path $repoRoot 'tools\lam_python_probe\patches\lam-one-click-windows.patch'
$archivePath = '<download-directory>\LAM-windows-one-click-install.zip'
$lamRoot = '<extracted-directory>\LAM'
$expectedSha256 = '81CA564F84DF14DB995868685AF42D612B51B3C82204D2F6FCD366D45A4A1121'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash -ne $expectedSha256) {
    throw 'LAM one-click archive SHA-256 mismatch'
}
Push-Location -LiteralPath $lamRoot
try {
    git apply --check $patchPath
    git apply $patchPath
} finally {
    Pop-Location
}
```

Before first launch, use the CUDA toolkit matching bundled PyTorch and restrict
native-extension compilation on a 16 GiB host:

```powershell
$env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:CUDA_PATH = $env:CUDA_HOME
$env:MAX_JOBS = '1'
$env:TORCH_EXTENSIONS_DIR = '<writable-extension-cache>'
$env:TORCH_HOME = '<extracted-LAM>\models'
```

Launch the patched official application from `$lamRoot`:

```powershell
& .\.glut\python.exe app_lam.py --blender_path blender
```

The patch keeps the model method's prior defaults for existing callers. The
tested `save_ply=False` path retains only requested tensor outputs and places
them directly on CPU, fixes backslash-safe path derivation, and avoids full-size
NumPy intermediates. `save_ply=True` still retains per-frame non-tensor 3DGS
objects and has not been validated as bounded-memory.
The archive used to validate the patch has SHA-256
`81ca564f84df14db995868685af42d612b51b3c82204d2f6fcd366d45a4a1121`.
