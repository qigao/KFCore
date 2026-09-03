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
  --lam-root .cache/lam-upstream/LAM `
  --a2e-root .cache/lam-upstream/LAM_Audio2Expression `
  --output .cache/lam-prototype/results/preflight.json
```

The command fails when either upstream checkout is missing or when the output
path is a directory. Reports use schema version 1 and are written atomically.

Validate an A2E expression export against its source WAV:

```powershell
python tools/lam_python_probe/a2e_output.py `
  --expression .cache/lam-prototype/results/a2e/no-extraction/bsData.json `
  --audio .cache/lam-upstream/LAM_Audio2Expression/assets/sample_audio/BarackObama_english.wav
```

The validator requires a non-empty 30 fps sequence, exactly 52 finite weights
per frame, weights in `[0, 1]`, and a frame count matching the WAV duration.

## LAM Windows one-click compatibility patch

The official CUDA 12.8 one-click archive needs a bounded-output patch to render
the bundled 519-frame sample on the tested 8 GiB GPU / 16 GiB host. From the
extracted LAM directory, verify and apply:

```powershell
git apply --unidiff-zero --check C:\projects\cpp\KFCore\.worktrees\lam-python-prototype\tools\lam_python_probe\patches\lam-one-click-windows.patch
git apply --unidiff-zero C:\projects\cpp\KFCore\.worktrees\lam-python-prototype\tools\lam_python_probe\patches\lam-one-click-windows.patch
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

The patch keeps the model method's prior defaults for existing callers. The
runner opts into retaining only requested outputs and placing them directly on
CPU, fixes backslash-safe path derivation, and avoids full-size NumPy copies.
The archive used to validate the patch has SHA-256
`81ca564f84df14db995868685af42d612b51b3c82204d2f6fcd366d45a4a1121`.
