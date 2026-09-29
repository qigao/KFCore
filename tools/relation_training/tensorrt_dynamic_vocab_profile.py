from __future__ import annotations

import argparse
import json
from pathlib import Path


def _shape_triplet(shape: list[int]) -> dict[str, list[int]]:
    return {
        "min": list(shape),
        "opt": list(shape),
        "max": list(shape),
    }


def dynamic_vocabulary_profiles(
    metadata: dict[str, object],
    *,
    opt_vocab: int | None = None,
    max_vocab: int | None = None,
) -> dict[str, dict[str, list[int]]]:
    if metadata.get("model_type") != "relation.open-vocabulary":
        raise ValueError(
            "TensorRT dynamic-vocabulary profile requires "
            "model_type=relation.open-vocabulary"
        )
    if metadata.get("vocabulary_dynamic") is not True:
        raise ValueError(
            "TensorRT profile requires vocabulary_dynamic=true"
        )
    if metadata.get("vocabulary_graph_input") is not True:
        raise ValueError(
            "TensorRT profile requires vocabulary_graph_input=true"
        )

    image_size = int(metadata["image_size"])
    max_boxes = int(metadata["max_boxes"])
    text_dim = int(metadata["query_dim"])
    default_vocab = int(metadata["default_predicate_count"])
    if (
        image_size <= 0
        or max_boxes <= 0
        or text_dim <= 0
        or default_vocab <= 0
    ):
        raise ValueError(
            "dynamic-vocabulary metadata dimensions must be positive"
        )

    optimum = default_vocab if opt_vocab is None else int(opt_vocab)
    maximum = max(default_vocab, optimum) if max_vocab is None else int(max_vocab)
    if optimum <= 0 or maximum <= 0:
        raise ValueError("TensorRT vocabulary profile extents must be positive")
    if optimum > maximum:
        raise ValueError("TensorRT opt vocabulary must not exceed max vocabulary")

    profiles = {
        "image": _shape_triplet([1, 3, image_size, image_size]),
        "boxes": _shape_triplet([1, max_boxes, 4]),
        "box_counts": _shape_triplet([1]),
        "W": {
            "min": [1, text_dim],
            "opt": [optimum, text_dim],
            "max": [maximum, text_dim],
        },
        "alpha": {
            "min": [1],
            "opt": [optimum],
            "max": [maximum],
        },
    }
    return profiles


def _shape_text(value: list[int]) -> str:
    return "x".join(str(int(item)) for item in value)


def trtexec_shape_flags(
    profiles: dict[str, dict[str, list[int]]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for selector in ("min", "opt", "max"):
        entries = [
            f"{name}:{_shape_text(shapes[selector])}"
            for name, shapes in profiles.items()
        ]
        result[f"{selector}Shapes"] = ",".join(entries)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the TensorRT optimization-profile contract for a "
            "KFCore relation.open-vocabulary ONNX sidecar."
        )
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opt-vocab", type=int)
    parser.add_argument("--max-vocab", type=int)
    args = parser.parse_args()

    source = Path(args.metadata)
    metadata = json.loads(source.read_text(encoding="utf-8"))
    profiles = dynamic_vocabulary_profiles(
        metadata,
        opt_vocab=args.opt_vocab,
        max_vocab=args.max_vocab,
    )
    payload = {
        "schema": "kfcore.tensorrt-dynamic-vocab-profile/1",
        "source_metadata": source.name,
        "precision": "fp32",
        "tf32": False,
        "profiles": profiles,
        "trtexec": trtexec_shape_flags(profiles),
    }
    output = Path(args.out)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
