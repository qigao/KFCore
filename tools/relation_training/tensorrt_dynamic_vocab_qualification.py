from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCHEMA = "kfcore.tensorrt-dynamic-vocab-qualification/1"
EXPECTED_INPUTS = ("image", "boxes", "box_counts", "W", "alpha")
EXPECTED_OUTPUTS = (
    "pred_logits",
    "pair_logits",
    "sub_idx",
    "obj_idx",
    "valid_mask",
)
DEFAULT_SEED = 20260929
DEFAULT_ABS_TOL = 1.0e-4
DEFAULT_REL_TOL = 1.0e-4


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def qualification_cases(
    engine_metadata: Mapping[str, object],
) -> list[dict[str, int | str]]:
    if engine_metadata.get("model_type") != "relation.open-vocabulary":
        raise ValueError("qualification requires relation.open-vocabulary")
    tensorrt = engine_metadata.get("tensorrt")
    if not isinstance(tensorrt, Mapping):
        raise ValueError("engine metadata is missing tensorrt provenance")
    profiles = tensorrt.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("engine metadata is missing optimization profiles")

    bank = profiles.get("W")
    alpha = profiles.get("alpha")
    if not isinstance(bank, Mapping) or not isinstance(alpha, Mapping):
        raise ValueError("engine profile must contain W and alpha")

    def shape(entry: Mapping[str, object], selector: str, rank: int) -> tuple[int, ...]:
        raw = entry.get(selector)
        if (
            not isinstance(raw, list)
            or len(raw) != rank
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in raw)
        ):
            raise ValueError(f"invalid {selector} profile shape")
        return tuple(raw)

    w_min = shape(bank, "min", 2)
    w_opt = shape(bank, "opt", 2)
    w_max = shape(bank, "max", 2)
    a_min = shape(alpha, "min", 1)
    a_opt = shape(alpha, "opt", 1)
    a_max = shape(alpha, "max", 1)

    if w_min[1] != w_opt[1] or w_min[1] != w_max[1]:
        raise ValueError("W embedding dimension must be fixed")
    if (w_min[0], w_opt[0], w_max[0]) != (a_min[0], a_opt[0], a_max[0]):
        raise ValueError("W/alpha vocabulary profile extents disagree")
    if not (w_min[0] <= 1 <= w_max[0]):
        raise ValueError("profile does not contain V=1")
    if not (w_min[0] <= 3 <= w_max[0]):
        raise ValueError("profile does not contain V=3")
    if not (w_min[0] <= w_opt[0] <= w_max[0]):
        raise ValueError("invalid min/opt/max vocabulary profile order")

    return [
        {"label": "v1", "vocabulary_size": 1},
        {"label": "v3", "vocabulary_size": 3},
        {"label": "opt", "vocabulary_size": w_opt[0]},
        {"label": "max", "vocabulary_size": w_max[0]},
    ]


def _dummy_boxes(max_boxes: int) -> np.ndarray:
    boxes = np.zeros((1, max_boxes, 4), dtype=np.float32)
    columns = max(1, int(max_boxes**0.5))
    rows = max(1, (max_boxes + columns - 1) // columns)
    for index in range(max_boxes):
        x = (index % columns + 0.5) / columns
        y = (index // columns + 0.5) / rows
        boxes[0, index] = (x, y, 0.2, 0.2)
    return np.clip(boxes, 0.01, 0.99).astype(np.float32, copy=False)


def deterministic_inputs(
    metadata: Mapping[str, object],
    vocabulary_size: int,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, np.ndarray]:
    v = _require_positive_int(vocabulary_size, "vocabulary_size")
    image_size = _require_positive_int(metadata.get("image_size"), "image_size")
    max_boxes = _require_positive_int(metadata.get("max_boxes"), "max_boxes")
    query_dim = _require_positive_int(metadata.get("query_dim"), "query_dim")

    rng = np.random.default_rng(seed + v * 104729)
    image = rng.random((1, 3, image_size, image_size), dtype=np.float32)
    boxes = _dummy_boxes(max_boxes)
    box_counts = np.asarray([max_boxes], dtype=np.int64)

    bank = rng.standard_normal((v, query_dim), dtype=np.float32)
    norms = np.linalg.norm(bank, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise RuntimeError("deterministic predicate bank generated a zero row")
    bank = (bank / norms).astype(np.float32, copy=False)
    alpha = (
        np.asarray([0.5], dtype=np.float32)
        if v == 1
        else np.linspace(0.0, 1.0, v, dtype=np.float32)
    )
    return {
        "image": np.ascontiguousarray(image),
        "boxes": np.ascontiguousarray(boxes),
        "box_counts": np.ascontiguousarray(box_counts),
        "W": np.ascontiguousarray(bank),
        "alpha": np.ascontiguousarray(alpha),
    }


def _validate_outputs(
    outputs: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = [name for name in EXPECTED_OUTPUTS if name not in outputs]
    if missing:
        raise ValueError("missing outputs: " + ", ".join(missing))

    pred = np.asarray(outputs["pred_logits"])
    pair = np.asarray(outputs["pair_logits"])
    sub = np.asarray(outputs["sub_idx"])
    obj = np.asarray(outputs["obj_idx"])
    valid = np.asarray(outputs["valid_mask"])

    if pred.ndim != 3 or pred.shape[0] != 1:
        raise ValueError("pred_logits must be [1,K,V]")
    k = pred.shape[1]
    if pair.shape != (1, k):
        raise ValueError("pair_logits must be [1,K]")
    if sub.shape != (1, k) or obj.shape != (1, k) or valid.shape != (1, k):
        raise ValueError("pair index/valid outputs must be [1,K]")
    if not np.issubdtype(pred.dtype, np.floating):
        raise ValueError("pred_logits must be floating point")
    if not np.issubdtype(pair.dtype, np.floating):
        raise ValueError("pair_logits must be floating point")
    if not np.issubdtype(sub.dtype, np.integer) or not np.issubdtype(obj.dtype, np.integer):
        raise ValueError("sub_idx/obj_idx must be integer")
    if valid.dtype != np.bool_:
        raise ValueError("valid_mask must be bool")
    return pred[0], pair[0], sub[0], obj[0], valid[0]


def _pair_rows(
    outputs: Mapping[str, np.ndarray],
) -> dict[tuple[int, int], tuple[float, np.ndarray]]:
    pred, pair, sub, obj, valid = _validate_outputs(outputs)
    result: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    for row in np.flatnonzero(valid):
        key = (int(sub[row]), int(obj[row]))
        if key in result:
            raise ValueError(f"duplicate valid pair key: {key}")
        result[key] = (float(pair[row]), np.asarray(pred[row], dtype=np.float64))
    return result


def _error_summary(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    abs_tol: float,
    rel_tol: float,
) -> dict[str, float | bool]:
    if reference.shape != actual.shape:
        raise ValueError("numeric parity arrays have different shapes")
    reference64 = np.asarray(reference, dtype=np.float64)
    actual64 = np.asarray(actual, dtype=np.float64)
    delta = np.abs(reference64 - actual64)
    scale = np.maximum(np.abs(reference64), 1.0e-12)
    relative = delta / scale
    max_abs = float(delta.max(initial=0.0))
    max_rel = float(relative.max(initial=0.0))
    passed = bool(np.all(delta <= abs_tol + rel_tol * np.abs(reference64)))
    return {
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "passed": passed,
    }


def compare_pair_keyed_outputs(
    reference: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> dict[str, object]:
    if abs_tol < 0.0 or rel_tol < 0.0:
        raise ValueError("parity tolerances must be non-negative")

    expected = _pair_rows(reference)
    got = _pair_rows(actual)
    expected_keys = set(expected)
    actual_keys = set(got)
    if expected_keys != actual_keys:
        return {
            "passed": False,
            "pair_set_equal": False,
            "reference_valid_pairs": len(expected_keys),
            "actual_valid_pairs": len(actual_keys),
            "missing_pairs": [list(value) for value in sorted(expected_keys - actual_keys)],
            "extra_pairs": [list(value) for value in sorted(actual_keys - expected_keys)],
            "pair_logits": None,
            "pred_logits": None,
        }

    reference_pair = np.asarray(
        [expected[key][0] for key in sorted(expected_keys)],
        dtype=np.float64,
    )
    actual_pair = np.asarray(
        [got[key][0] for key in sorted(expected_keys)],
        dtype=np.float64,
    )
    reference_pred = np.stack(
        [expected[key][1] for key in sorted(expected_keys)],
        axis=0,
    ) if expected_keys else np.empty((0, 0), dtype=np.float64)
    actual_pred = np.stack(
        [got[key][1] for key in sorted(expected_keys)],
        axis=0,
    ) if expected_keys else np.empty((0, 0), dtype=np.float64)

    pair_report = _error_summary(
        reference_pair,
        actual_pair,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    pred_report = _error_summary(
        reference_pred,
        actual_pred,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    return {
        "passed": bool(pair_report["passed"] and pred_report["passed"]),
        "pair_set_equal": True,
        "reference_valid_pairs": len(expected_keys),
        "actual_valid_pairs": len(actual_keys),
        "missing_pairs": [],
        "extra_pairs": [],
        "pair_logits": pair_report,
        "pred_logits": pred_report,
    }


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("latency sample list must not be empty")
    values = sorted(float(value) for value in samples_ms)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("latency samples must be finite and non-negative")
    p95_index = max(0, math.ceil(0.95 * len(values)) - 1)
    return {
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "p95_ms": values[p95_index],
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def validate_qualification_report(payload: Mapping[str, object]) -> dict[str, object]:
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported TensorRT qualification schema")
    if payload.get("hardware_executed") is not True:
        raise ValueError("qualification requires real hardware execution")
    if payload.get("same_engine_reused") is not True:
        raise ValueError("qualification did not prove one-engine reuse")
    if payload.get("same_context_reused") is not True:
        raise ValueError("qualification did not prove one-context reuse")
    if payload.get("passed") is not True:
        raise ValueError("qualification report is not globally passing")
    if payload.get("engine_sha256_before") != payload.get("engine_sha256_after"):
        raise ValueError("engine bytes changed during vocabulary qualification")

    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("qualification requires v1/v3/opt/max cases")
    labels = [case.get("label") if isinstance(case, Mapping) else None for case in cases]
    if labels != ["v1", "v3", "opt", "max"]:
        raise ValueError("qualification case order must be v1/v3/opt/max")
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or case.get("passed") is not True:
            raise ValueError(f"qualification case {index} did not pass")
        if case.get("pair_set_equal") is not True:
            raise ValueError(f"qualification case {index} pair set differs")
    return dict(payload)


def _time_call(
    call: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be >=0 and iterations must be >0")
    for _ in range(warmup):
        call()
    samples: list[float] = []
    for _ in range(iterations):
        begin = time.perf_counter()
        call()
        samples.append((time.perf_counter() - begin) * 1000.0)
    return latency_summary(samples)


class _TensorRTRunner:
    def __init__(self, engine_path: Path, engine_metadata: Mapping[str, object]) -> None:
        try:
            import tensorrt as trt
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT qualification requires tensorrt and CUDA-enabled torch"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT qualification requires an NVIDIA CUDA device")

        self.trt = trt
        self.torch = torch
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        engine_bytes = engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("TensorRT could not deserialize the engine")
        if self.engine.num_optimization_profiles != 1:
            raise RuntimeError("qualification requires exactly one optimization profile")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context")
        self.stream = torch.cuda.Stream()
        self.engine_load_count = 1
        self.context_create_count = 1
        self._validate_engine_contract(engine_metadata)

    def _validate_engine_contract(self, metadata: Mapping[str, object]) -> None:
        trt = self.trt
        names = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        ]
        inputs = {
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        }
        outputs = set(names) - inputs
        if inputs != set(EXPECTED_INPUTS) or outputs != set(EXPECTED_OUTPUTS):
            raise RuntimeError(
                f"TensorRT engine I/O mismatch: inputs={sorted(inputs)}, outputs={sorted(outputs)}"
            )

        tensorrt_meta = metadata.get("tensorrt")
        if not isinstance(tensorrt_meta, Mapping):
            raise ValueError("engine metadata lacks tensorrt block")
        profiles = tensorrt_meta.get("profiles")
        if not isinstance(profiles, Mapping):
            raise ValueError("engine metadata lacks profiles")

        for name in EXPECTED_INPUTS:
            declared = profiles.get(name)
            if not isinstance(declared, Mapping):
                raise ValueError(f"engine metadata lacks profile for {name}")
            actual = self.engine.get_tensor_profile_shape(name, 0)
            actual_shapes = {
                "min": list(actual[0]),
                "opt": list(actual[1]),
                "max": list(actual[2]),
            }
            expected_shapes = {
                selector: list(declared[selector])
                for selector in ("min", "opt", "max")
            }
            if actual_shapes != expected_shapes:
                raise RuntimeError(
                    f"engine profile for {name} differs from sidecar: "
                    f"{actual_shapes} != {expected_shapes}"
                )

    def provenance(self) -> dict[str, object]:
        torch = self.torch
        device = torch.cuda.current_device()
        return {
            "tensorrt_version": self.trt.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "engine_load_count": self.engine_load_count,
            "context_create_count": self.context_create_count,
        }

    def _torch_dtype(self, name: str):
        np_dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
        mapping = {
            np.dtype(np.float32): self.torch.float32,
            np.dtype(np.float16): self.torch.float16,
            np.dtype(np.int64): self.torch.int64,
            np.dtype(np.int32): self.torch.int32,
            np.dtype(np.bool_): self.torch.bool,
        }
        if np_dtype not in mapping:
            raise RuntimeError(f"unsupported TensorRT tensor dtype for {name}: {np_dtype}")
        return mapping[np_dtype]

    def prepare(
        self,
        inputs: Mapping[str, np.ndarray],
    ) -> tuple[Callable[[], None], Callable[[], dict[str, np.ndarray]]]:
        torch = self.torch

        device_inputs = {}
        for name in EXPECTED_INPUTS:
            array = np.ascontiguousarray(inputs[name])
            expected_dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
            if array.dtype != expected_dtype:
                raise RuntimeError(
                    f"{name} dtype {array.dtype} != TensorRT dtype {expected_dtype}"
                )
            if not self.context.set_input_shape(name, tuple(array.shape)):
                raise RuntimeError(f"TensorRT rejected runtime shape for {name}: {array.shape}")
            tensor = torch.from_numpy(array).to(device="cuda")
            device_inputs[name] = tensor
            if not self.context.set_tensor_address(name, int(tensor.data_ptr())):
                raise RuntimeError(f"TensorRT rejected input address for {name}")

        insufficient = self.context.infer_shapes()
        if insufficient:
            raise RuntimeError(
                "TensorRT shape inference has insufficient tensors: "
                + ", ".join(insufficient)
            )

        device_outputs = {}
        for name in EXPECTED_OUTPUTS:
            shape = tuple(int(value) for value in self.context.get_tensor_shape(name))
            if not shape or any(value < 0 for value in shape):
                raise RuntimeError(f"TensorRT output shape unresolved for {name}: {shape}")
            tensor = torch.empty(
                shape,
                dtype=self._torch_dtype(name),
                device="cuda",
            )
            device_outputs[name] = tensor
            if not self.context.set_tensor_address(name, int(tensor.data_ptr())):
                raise RuntimeError(f"TensorRT rejected output address for {name}")

        torch.cuda.synchronize()

        def execute() -> None:
            if not self.context.execute_async_v3(int(self.stream.cuda_stream)):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            self.stream.synchronize()

        def collect() -> dict[str, np.ndarray]:
            execute()
            return {
                name: tensor.detach().cpu().numpy().copy()
                for name, tensor in device_outputs.items()
            }

        return execute, collect


def qualify(
    *,
    onnx_path: str | Path,
    engine_path: str | Path,
    engine_metadata_path: str | Path,
    output_path: str | Path,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    warmup: int = 2,
    iterations: int = 5,
    seed: int = DEFAULT_SEED,
    ort_provider: str = "CPUExecutionProvider",
) -> dict[str, object]:
    source = Path(onnx_path)
    engine_file = Path(engine_path)
    sidecar_path = Path(engine_metadata_path)
    for path in (source, engine_file, sidecar_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = load_json(sidecar_path)
    cases = qualification_cases(metadata)
    onnx_sha = sha256_file(source)
    engine_sha_before = sha256_file(engine_file)
    sidecar_sha = sha256_file(sidecar_path)

    if _require_sha256(metadata.get("onnx_sha256"), "onnx_sha256") != onnx_sha:
        raise ValueError("ONNX bytes do not match model metadata")
    tensorrt_meta = metadata.get("tensorrt")
    assert isinstance(tensorrt_meta, Mapping)
    if _require_sha256(
        tensorrt_meta.get("source_onnx_sha256"),
        "tensorrt.source_onnx_sha256",
    ) != onnx_sha:
        raise ValueError("engine sidecar source ONNX hash differs from supplied ONNX")
    if _require_sha256(
        metadata.get("tensorrt_engine_sha256"),
        "tensorrt_engine_sha256",
    ) != engine_sha_before:
        raise ValueError("engine bytes do not match engine sidecar")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("TensorRT qualification requires onnxruntime") from exc

    available = ort.get_available_providers()
    if ort_provider not in available:
        raise RuntimeError(
            f"ORT provider {ort_provider!r} is unavailable; available={available}"
        )
    ort_session = ort.InferenceSession(
        str(source.resolve()),
        providers=[ort_provider],
    )
    ort_inputs = {item.name for item in ort_session.get_inputs()}
    ort_outputs = [item.name for item in ort_session.get_outputs()]
    if ort_inputs != set(EXPECTED_INPUTS) or ort_outputs != list(EXPECTED_OUTPUTS):
        raise RuntimeError(
            f"ORT graph contract mismatch: inputs={sorted(ort_inputs)}, outputs={ort_outputs}"
        )

    trt_runner = _TensorRTRunner(engine_file, metadata)
    case_reports: list[dict[str, object]] = []
    for case in cases:
        label = str(case["label"])
        vocabulary_size = int(case["vocabulary_size"])
        feeds = deterministic_inputs(metadata, vocabulary_size, seed=seed)

        def run_ort():
            return ort_session.run(list(EXPECTED_OUTPUTS), feeds)

        ort_values = run_ort()
        reference = {
            name: value
            for name, value in zip(EXPECTED_OUTPUTS, ort_values)
        }
        trt_execute, trt_collect = trt_runner.prepare(feeds)
        actual = trt_collect()

        parity = compare_pair_keyed_outputs(
            reference,
            actual,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        ort_latency = _time_call(
            run_ort,
            warmup=warmup,
            iterations=iterations,
        )
        trt_latency = _time_call(
            trt_execute,
            warmup=warmup,
            iterations=iterations,
        )
        case_report = {
            "label": label,
            "vocabulary_size": vocabulary_size,
            "pair_set_equal": parity["pair_set_equal"],
            "reference_valid_pairs": parity["reference_valid_pairs"],
            "actual_valid_pairs": parity["actual_valid_pairs"],
            "missing_pairs": parity["missing_pairs"],
            "extra_pairs": parity["extra_pairs"],
            "pair_logits": parity["pair_logits"],
            "pred_logits": parity["pred_logits"],
            "ort_latency": ort_latency,
            "tensorrt_latency": trt_latency,
            "passed": parity["passed"],
        }
        if not parity["passed"]:
            raise RuntimeError(
                f"TensorRT parity failed for {label} (V={vocabulary_size})"
            )
        case_reports.append(case_report)

    engine_sha_after = sha256_file(engine_file)
    report = {
        "schema": SCHEMA,
        "hardware_executed": True,
        "source_onnx": source.name,
        "source_onnx_sha256": onnx_sha,
        "engine": engine_file.name,
        "engine_sha256_before": engine_sha_before,
        "engine_sha256_after": engine_sha_after,
        "engine_sidecar": sidecar_path.name,
        "engine_sidecar_sha256": sidecar_sha,
        "same_engine_reused": True,
        "same_context_reused": True,
        "abs_tolerance": abs_tol,
        "rel_tolerance": rel_tol,
        "seed": seed,
        "warmup": warmup,
        "iterations": iterations,
        "ort": {
            "version": ort.__version__,
            "provider": ort_provider,
            "session_load_count": 1,
        },
        "hardware": trt_runner.provenance(),
        "profile": tensorrt_meta["profiles"],
        "cases": case_reports,
        "passed": all(bool(case["passed"]) for case in case_reports),
    }
    validate_qualification_report(report)

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify one prebuilt TensorRT dynamic-vocabulary engine against "
            "one ORT session at V=1/3/opt/max without rebuilding the engine."
        )
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--engine-metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--abs-tol", type=float, default=DEFAULT_ABS_TOL)
    parser.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ort-provider", default="CPUExecutionProvider")
    args = parser.parse_args()

    report = qualify(
        onnx_path=args.onnx,
        engine_path=args.engine,
        engine_metadata_path=args.engine_metadata,
        output_path=args.out,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        warmup=args.warmup,
        iterations=args.iterations,
        seed=args.seed,
        ort_provider=args.ort_provider,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
