from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tensorrt_dynamic_vocab_profile import (
    dynamic_vocabulary_profiles,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_tuple(
    entry: dict[str, list[int]],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(int(v) for v in entry["min"]),
        tuple(int(v) for v in entry["opt"]),
        tuple(int(v) for v in entry["max"]),
    )


def validate_profile_against_graph(
    graph_shape: tuple[int, ...],
    profile: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ],
    *,
    name: str,
) -> None:
    for candidate in profile:
        if len(candidate) != len(graph_shape):
            raise ValueError(
                f"{name}: profile rank does not match graph rank"
            )
        for declared, actual in zip(
            graph_shape,
            candidate,
        ):
            if declared >= 0 and declared != actual:
                raise ValueError(
                    f"{name}: graph shape {graph_shape} disagrees "
                    f"with profile {profile}"
                )


def build_engine(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    opt_vocab: int | None = None,
    max_vocab: int | None = None,
    workspace_gib: float = 4.0,
    device: str = "cuda",
) -> dict[str, object]:
    if workspace_gib <= 0.0:
        raise ValueError("workspace_gib must be positive")

    source = Path(onnx_path)
    output = Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("TensorRT output must differ from ONNX source")
    if output.suffix not in {".engine", ".plan"}:
        raise ValueError(
            "TensorRT output must use .engine or .plan"
        )

    metadata_file = (
        source.with_suffix(".json")
        if metadata_path is None
        else Path(metadata_path)
    )
    metadata = json.loads(
        metadata_file.read_text(encoding="utf-8")
    )
    profiles = dynamic_vocabulary_profiles(
        metadata,
        opt_vocab=opt_vocab,
        max_vocab=max_vocab,
    )

    import torch
    import tensorrt as trt

    target = torch.device(device)
    if (
        target.type != "cuda"
        or not torch.cuda.is_available()
    ):
        raise ValueError(
            "TensorRT engine build requires a CUDA-enabled NVIDIA environment"
        )

    with torch.cuda.device(target):
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1
            << int(
                trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
            )
        )
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(
            str(source.resolve())
        ):
            errors = "\n".join(
                str(parser.get_error(index))
                for index in range(parser.num_errors)
            )
            raise RuntimeError(
                "TensorRT could not parse dynamic relation ONNX:\n"
                + errors
            )

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(workspace_gib * (1 << 30)),
        )
        config.clear_flag(trt.BuilderFlag.TF32)

        profile = builder.create_optimization_profile()
        actual_profiles: dict[
            str,
            tuple[
                tuple[int, ...],
                tuple[int, ...],
                tuple[int, ...],
            ],
        ] = {}

        expected_inputs = {
            "image",
            "boxes",
            "box_counts",
            "W",
            "alpha",
        }
        graph_inputs = {
            network.get_input(index).name
            for index in range(network.num_inputs)
        }
        if graph_inputs != expected_inputs:
            raise ValueError(
                "TensorRT graph inputs do not match "
                "relation.open-vocabulary contract: "
                f"{sorted(graph_inputs)}"
            )

        for index in range(network.num_inputs):
            tensor = network.get_input(index)
            name = tensor.name
            shapes = _profile_tuple(profiles[name])
            graph_shape = tuple(int(v) for v in tensor.shape)
            validate_profile_against_graph(
                graph_shape,
                shapes,
                name=name,
            )

            # Specialize every fixed axis before build. Only V remains
            # dynamic, matching the Apache deployment contract.
            if shapes[0] == shapes[2]:
                tensor.shape = shapes[0]
            if not profile.set_shape(
                name,
                shapes[0],
                shapes[1],
                shapes[2],
            ):
                raise RuntimeError(
                    f"TensorRT rejected optimization profile for {name}"
                )
            actual_profiles[name] = shapes

        if not config.add_optimization_profile(profile):
            raise RuntimeError(
                "TensorRT rejected the dynamic-vocabulary optimization profile"
            )

        plan = builder.build_serialized_network(
            network,
            config,
        )
        if plan is None:
            raise RuntimeError(
                "TensorRT engine build failed"
            )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output.write_bytes(bytes(plan))

        engine_metadata = dict(metadata)
        engine_metadata["tensorrt"] = {
            "version": trt.__version__,
            "precision": "fp32",
            "tf32": False,
            "workspace_gib": workspace_gib,
            "gpu": torch.cuda.get_device_name(target),
            "compute_capability": list(
                torch.cuda.get_device_capability(target)
            ),
            "source_onnx": source.name,
            "source_onnx_sha256": _sha256(source),
            "profiles": {
                name: {
                    "min": list(shapes[0]),
                    "opt": list(shapes[1]),
                    "max": list(shapes[2]),
                }
                for name, shapes in actual_profiles.items()
            },
        }
        engine_metadata["tensorrt_engine_sha256"] = (
            _sha256(output)
        )
        sidecar = output.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                engine_metadata,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return engine_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TensorRT 10 engine for KFCore "
            "relation.open-vocabulary while keeping only V dynamic."
        )
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--out", required=True)
    parser.add_argument("--opt-vocab", type=int)
    parser.add_argument("--max-vocab", type=int)
    parser.add_argument(
        "--workspace-gib",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    args = parser.parse_args()

    metadata = build_engine(
        args.onnx,
        args.out,
        metadata_path=args.metadata,
        opt_vocab=args.opt_vocab,
        max_vocab=args.max_vocab,
        workspace_gib=args.workspace_gib,
        device=args.device,
    )
    print(
        json.dumps(
            metadata["tensorrt"],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
