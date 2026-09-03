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
