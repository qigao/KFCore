from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


IMAGE_SIZE = 8
MAX_BOXES = 3
PAIR_BUDGET = 4
QUERY_DIM = 4
OUTPUT_NAMES = [
    "pred_logits",
    "pair_logits",
    "sub_idx",
    "obj_idx",
    "valid_mask",
]
ENCODER_OUTPUT_NAMES = [
    "semantic_query",
    "spatial_query",
    "pair_logits",
    "sub_idx",
    "obj_idx",
    "valid_mask",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cases() -> list[dict[str, object]]:
    return [
        {
            "label": "v1",
            "predicates": ["holding"],
            "embeddings": [[1.0, 0.0, 0.0, 0.0]],
            "alpha": [0.0],
        },
        {
            "label": "v3",
            "predicates": ["left of", "holding", "behind"],
            "embeddings": [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
            ],
            "alpha": [0.0, 0.0, 0.0],
        },
        {
            "label": "default",
            "predicates": ["above", "below", "beside", "holding"],
            "embeddings": [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            "alpha": [0.0, 0.0, 0.0, 0.0],
        },
        {
            "label": "v1-repeat",
            "predicates": ["holding"],
            "embeddings": [[1.0, 0.0, 0.0, 0.0]],
            "alpha": [0.0],
        },
    ]


def make_model(path: Path) -> None:
    inputs = [
        helper.make_tensor_value_info(
            "image", TensorProto.FLOAT, [1, 3, IMAGE_SIZE, IMAGE_SIZE]
        ),
        helper.make_tensor_value_info(
            "boxes", TensorProto.FLOAT, [1, MAX_BOXES, 4]
        ),
        helper.make_tensor_value_info("box_counts", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("W", TensorProto.FLOAT, ["V", QUERY_DIM]),
        helper.make_tensor_value_info("alpha", TensorProto.FLOAT, ["V"]),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "pred_logits", TensorProto.FLOAT, [1, PAIR_BUDGET, "V"]
        ),
        helper.make_tensor_value_info(
            "pair_logits", TensorProto.FLOAT, [1, PAIR_BUDGET]
        ),
        helper.make_tensor_value_info(
            "sub_idx", TensorProto.INT64, [1, PAIR_BUDGET]
        ),
        helper.make_tensor_value_info(
            "obj_idx", TensorProto.INT64, [1, PAIR_BUDGET]
        ),
        helper.make_tensor_value_info(
            "valid_mask", TensorProto.BOOL, [1, PAIR_BUDGET]
        ),
    ]

    initializers = [
        numpy_helper.from_array(np.asarray(0, dtype=np.int64), name="idx0"),
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="axis0"),
        numpy_helper.from_array(np.asarray([0, 1], dtype=np.int64), name="axes01"),
        numpy_helper.from_array(
            np.asarray([1, PAIR_BUDGET], dtype=np.int64), name="pred_prefix"
        ),
        numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), name="zero"),
        numpy_helper.from_array(
            np.asarray([[2.0, 1.0, 0.0, -1.0]], dtype=np.float32),
            name="pair_base",
        ),
        numpy_helper.from_array(
            np.asarray([[0, 1, 0, 0]], dtype=np.int64), name="sub_base"
        ),
        numpy_helper.from_array(
            np.asarray([[1, 0, 2, 0]], dtype=np.int64), name="obj_base"
        ),
        numpy_helper.from_array(
            np.asarray([[True, True, True, False]], dtype=np.bool_),
            name="valid_base",
        ),
    ]

    nodes = [
        helper.make_node("ReduceSum", ["image"], ["image_sum"], keepdims=0),
        helper.make_node("ReduceSum", ["boxes"], ["boxes_sum"], keepdims=0),
        helper.make_node("Cast", ["box_counts"], ["count_float"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["count_float"], ["count_sum"], keepdims=0),
        helper.make_node("Add", ["image_sum", "boxes_sum"], ["dep0"]),
        helper.make_node("Add", ["dep0", "count_sum"], ["dep1"]),
        helper.make_node("Mul", ["dep1", "zero"], ["zero_dep"]),
        helper.make_node("Gather", ["W", "idx0"], ["w0"], axis=1),
        helper.make_node("Add", ["w0", "alpha"], ["predicate_base0"]),
        helper.make_node("Add", ["predicate_base0", "zero_dep"], ["predicate_base"]),
        helper.make_node("Unsqueeze", ["predicate_base", "axes01"], ["pred_111v"]),
        helper.make_node("Shape", ["W"], ["w_shape"]),
        helper.make_node("Gather", ["w_shape", "idx0"], ["v_scalar"], axis=0),
        helper.make_node("Unsqueeze", ["v_scalar", "axis0"], ["v_vec"]),
        helper.make_node("Concat", ["pred_prefix", "v_vec"], ["pred_shape"], axis=0),
        helper.make_node("Expand", ["pred_111v", "pred_shape"], ["pred_logits"]),
        helper.make_node("Add", ["pair_base", "zero_dep"], ["pair_logits"]),
        helper.make_node("Identity", ["sub_base"], ["sub_idx"]),
        helper.make_node("Identity", ["obj_base"], ["obj_idx"]),
        helper.make_node("Identity", ["valid_base"], ["valid_mask"]),
    ]

    graph = helper.make_graph(
        nodes,
        "kfcore-ort-cpu-cpp-qualification",
        inputs,
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="kfcore-qualification",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_encoder_model(path: Path) -> None:
    inputs = [
        helper.make_tensor_value_info(
            "image", TensorProto.FLOAT, [1, 3, IMAGE_SIZE, IMAGE_SIZE]
        ),
        helper.make_tensor_value_info(
            "boxes", TensorProto.FLOAT, [1, MAX_BOXES, 4]
        ),
        helper.make_tensor_value_info("box_counts", TensorProto.INT64, [1]),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "semantic_query",
            TensorProto.FLOAT,
            [1, PAIR_BUDGET, QUERY_DIM],
        ),
        helper.make_tensor_value_info(
            "spatial_query",
            TensorProto.FLOAT,
            [1, PAIR_BUDGET, QUERY_DIM],
        ),
        helper.make_tensor_value_info(
            "pair_logits", TensorProto.FLOAT, [1, PAIR_BUDGET]
        ),
        helper.make_tensor_value_info(
            "sub_idx", TensorProto.INT64, [1, PAIR_BUDGET]
        ),
        helper.make_tensor_value_info(
            "obj_idx", TensorProto.INT64, [1, PAIR_BUDGET]
        ),
        helper.make_tensor_value_info(
            "valid_mask", TensorProto.BOOL, [1, PAIR_BUDGET]
        ),
    ]

    query = np.zeros((1, PAIR_BUDGET, QUERY_DIM), dtype=np.float32)
    query[:, :, 0] = 1.0
    initializers = [
        numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), name="zero"),
        numpy_helper.from_array(query, name="semantic_base"),
        numpy_helper.from_array(query, name="spatial_base"),
        numpy_helper.from_array(
            np.asarray([[2.0, 1.0, 0.0, -1.0]], dtype=np.float32),
            name="pair_base",
        ),
        numpy_helper.from_array(
            np.asarray([[0, 1, 0, 0]], dtype=np.int64), name="sub_base"
        ),
        numpy_helper.from_array(
            np.asarray([[1, 0, 2, 0]], dtype=np.int64), name="obj_base"
        ),
        numpy_helper.from_array(
            np.asarray([[True, True, True, False]], dtype=np.bool_),
            name="valid_base",
        ),
    ]
    nodes = [
        helper.make_node("ReduceSum", ["image"], ["image_sum"], keepdims=0),
        helper.make_node("ReduceSum", ["boxes"], ["boxes_sum"], keepdims=0),
        helper.make_node("Cast", ["box_counts"], ["count_float"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["count_float"], ["count_sum"], keepdims=0),
        helper.make_node("Add", ["image_sum", "boxes_sum"], ["dep0"]),
        helper.make_node("Add", ["dep0", "count_sum"], ["dep1"]),
        helper.make_node("Mul", ["dep1", "zero"], ["zero_dep"]),
        helper.make_node("Add", ["semantic_base", "zero_dep"], ["semantic_query"]),
        helper.make_node("Add", ["spatial_base", "zero_dep"], ["spatial_query"]),
        helper.make_node("Add", ["pair_base", "zero_dep"], ["pair_logits"]),
        helper.make_node("Identity", ["sub_base"], ["sub_idx"]),
        helper.make_node("Identity", ["obj_base"], ["obj_idx"]),
        helper.make_node("Identity", ["valid_base"], ["valid_mask"]),
    ]
    graph = helper.make_graph(
        nodes,
        "kfcore-ort-cpu-cpp-host-scoring-qualification",
        inputs,
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="kfcore-qualification",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("vocabulary fixture contains a zero embedding")
    return (values / norms).astype(np.float32)


def decode_reference(
    outputs: list[np.ndarray],
) -> list[tuple[int, int, int, float]]:
    pred, pair, sub, obj, valid = outputs
    pred = pred[0]
    pair = pair[0]
    sub = sub[0]
    obj = obj[0]
    valid = valid[0]
    result: list[tuple[int, int, int, float]] = []
    for row in range(PAIR_BUDGET):
        if not bool(valid[row]):
            continue
        subject = int(sub[row])
        object_ = int(obj[row])
        if subject < 0 or object_ < 0 or subject >= MAX_BOXES or object_ >= MAX_BOXES:
            continue
        if subject == object_:
            continue
        scores = []
        for predicate in range(pred.shape[1]):
            fused = float(pred[row, predicate]) + float(pair[row])
            score = 1.0 / (1.0 + math.exp(-fused))
            scores.append(score)
        predicate = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        result.append((subject, object_, predicate, scores[predicate]))
    result.sort(key=lambda row: -row[3])
    return result


def write_vocabularies(path: Path) -> None:
    lines: list[str] = []
    for case in cases():
        label = str(case["label"])
        predicates = list(case["predicates"])
        embeddings = np.asarray(case["embeddings"], dtype=np.float32)
        alpha = np.asarray(case["alpha"], dtype=np.float32)
        if embeddings.shape != (len(predicates), QUERY_DIM):
            raise ValueError("invalid vocabulary fixture embedding shape")
        if alpha.shape != (len(predicates),):
            raise ValueError("invalid vocabulary fixture alpha shape")
        for predicate, row, routing in zip(predicates, embeddings, alpha):
            values = [
                label,
                predicate,
                *(format(float(value), ".9g") for value in row),
                format(float(routing), ".9g"),
            ]
            lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reference(
    path: Path,
    model_path: Path,
) -> None:
    session = ort.InferenceSession(
        str(model_path.resolve()),
        providers=["CPUExecutionProvider"],
    )
    image = np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    boxes = np.asarray(
        [
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.75, 0.25, 0.25, 0.25],
                [0.50, 0.75, 0.25, 0.25],
            ]
        ],
        dtype=np.float32,
    )
    box_counts = np.asarray([MAX_BOXES], dtype=np.int64)

    lines: list[str] = []
    for case in cases():
        bank = normalize_rows(np.asarray(case["embeddings"], dtype=np.float32))
        alpha = np.asarray(case["alpha"], dtype=np.float32)
        values = session.run(
            OUTPUT_NAMES,
            {
                "image": image,
                "boxes": boxes,
                "box_counts": box_counts,
                "W": bank,
                "alpha": alpha,
            },
        )
        for subject, object_, predicate, score in decode_reference(values):
            lines.append(
                "\t".join(
                    [
                        str(case["label"]),
                        str(subject),
                        str(object_),
                        str(predicate),
                        format(score, ".9g"),
                    ]
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_package(
    path: Path,
    model_path: Path,
    *,
    package_id: str = "relation-ort-cpu-cpp-qualification",
    model_type: str = "relation.open-vocabulary",
) -> None:
    payload = {
        "schema": "kfcore.model/1",
        "id": package_id,
        "version": "1",
        "model_type": model_type,
        "artifacts": [
            {
                "id": "ort-cpu",
                "format": "onnx",
                "path": model_path.name,
                "sha256": sha256_file(model_path),
                "backend": "onnxruntime",
                "device": "cpu",
            }
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=False)
    model_path = root / "relation.onnx"
    package_path = root / "model.json"
    host_root = root / "host"
    host_root.mkdir()
    host_model_path = host_root / "relation-encoder.onnx"
    host_package_path = host_root / "model.json"
    vocab_path = root / "vocabularies.tsv"
    reference_path = root / "reference.tsv"

    make_model(model_path)
    make_encoder_model(host_model_path)
    write_vocabularies(vocab_path)
    write_reference(reference_path, model_path)
    write_package(package_path, model_path)
    write_package(
        host_package_path,
        host_model_path,
        package_id="relation-ort-cpu-cpp-host-qualification",
        model_type="relation.open-vocabulary-encoder",
    )

    evidence = {
        "schema": "kfcore.relation-ort-cpu-fixture/1",
        "onnxruntime_version": ort.__version__,
        "provider": "CPUExecutionProvider",
        "model_sha256": sha256_file(model_path),
        "package_sha256": sha256_file(package_path),
        "host_model_sha256": sha256_file(host_model_path),
        "host_package_sha256": sha256_file(host_package_path),
        "vocabulary_fixture_sha256": sha256_file(vocab_path),
        "reference_sha256": sha256_file(reference_path),
        "cases": [str(case["label"]) for case in cases()],
        "vocabulary_sizes": [
            len(list(case["predicates"]))
            for case in cases()
        ],
    }
    (root / "fixture-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
