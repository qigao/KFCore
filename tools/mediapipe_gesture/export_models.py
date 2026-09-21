"""Convert the pinned official Gesture Recognizer bundle, with TFLite parity gates.

Offline tooling only; TensorFlow is not a KFCore runtime dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

BUNDLE_SHA256 = "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482"
MEMBERS = {
    "palm_detector": ("hand_landmarker.task", "hand_detector.tflite"),
    "hand_landmarker": ("hand_landmarker.task", "hand_landmarks_detector.tflite"),
    "gesture_embedder": ("hand_gesture_recognizer.task", "gesture_embedder.tflite"),
    "gesture_classifier": ("hand_gesture_recognizer.task", "canned_gesture_classifier.tflite"),
}
MAX_MEMBER_BYTES = 32 * 1024 * 1024
OPSET = 13
PARITY_CASES = 3
ABS_TOLERANCE = 0.002
REL_TOLERANCE = 0.002


def read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    entry = archive.getinfo(name)
    if entry.file_size > MAX_MEMBER_BYTES:
        raise ValueError(f"oversize model member: {name}")
    return archive.read(entry)


def verify(tflite: Path, exported: Path) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(tflite), num_threads=1)
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    if {v.name for v in session.get_inputs()} != {v["name"] for v in inputs}:
        raise ValueError("converted input names differ from TFLite")
    if {v.name for v in session.get_outputs()} != {v["name"] for v in outputs}:
        raise ValueError("converted output names differ from TFLite")
    rng = np.random.default_rng(42)
    errors = {v["name"]: 0.0 for v in outputs}
    for case in range(PARITY_CASES):
        feeds = {}
        for tensor in inputs:
            if tensor["dtype"] != np.float32:
                raise ValueError(f"unexpected input dtype: {tensor['dtype']}")
            value = (np.zeros(tensor["shape"], dtype=np.float32) if case == 0 else
                     rng.uniform(0, 1, tensor["shape"]).astype(np.float32))
            feeds[tensor["name"]] = value
            interpreter.set_tensor(tensor["index"], value)
        interpreter.invoke()
        actual = session.run([v["name"] for v in outputs], feeds)
        for tensor, result in zip(outputs, actual):
            expected = interpreter.get_tensor(tensor["index"])
            np.testing.assert_allclose(result, expected, rtol=REL_TOLERANCE,
                                       atol=ABS_TOLERANCE, equal_nan=False)
            errors[tensor["name"]] = max(errors[tensor["name"]],
                float(np.max(np.abs(result - expected))))
    return {"inputs": {v["name"]: v["shape"].tolist() for v in inputs},
            "outputs": {v["name"]: v["shape"].tolist() for v in outputs},
            "maximum_absolute_errors": errors}


def package_models(root: Path) -> None:
    model = onnx.load(root / "hand_landmarker.onnx")
    old_input = model.graph.input[0].name
    model.graph.input[0].CopyFrom(onnx.helper.make_tensor_value_info(
        "image", onnx.TensorProto.FLOAT, [1, 3, 224, 224]))
    model.graph.node.insert(0, onnx.helper.make_node("Transpose", ["image"], [old_input], perm=[0, 2, 3, 1]))
    names = ["landmarks", "hand_score", "handedness", "world_landmarks"]
    for output, name in zip(model.graph.output, names):
        old_name = output.name
        model.graph.node.append(onnx.helper.make_node("Identity", [old_name], [name]))
        output.name = name
    onnx.checker.check_model(model)
    onnx.save(model, root / "hand_world.onnx")
    rng = np.random.default_rng(7)
    image = rng.uniform(0, 1, (1, 3, 224, 224)).astype(np.float32)
    raw = ort.InferenceSession(str(root / "hand_landmarker.onnx"), providers=["CPUExecutionProvider"])
    wrapped = ort.InferenceSession(str(root / "hand_world.onnx"), providers=["CPUExecutionProvider"])
    for a, b in zip(raw.run(None, {old_input: image.transpose(0, 2, 3, 1)}), wrapped.run(None, {"image": image})):
        np.testing.assert_allclose(a, b, atol=ABS_TOLERANCE, rtol=REL_TOLERANCE, equal_nan=False)
    for name, kind, flavor in [
        ("hand_world", "hand.landmarker", "mediapipe-hand-world-v1"),
        ("gesture_embedder", "hand.gesture-embedder", "mediapipe-gesture-embedder-v1"),
        ("gesture_classifier", "hand.gesture-classifier", "mediapipe-canned-gesture-v1")]:
        artifact = root / f"{name}.onnx"
        manifest = {"schema": "kfcore.model/1", "id": name.replace("_", "-"), "version": "1.0.0",
                    "model_type": kind, "artifacts": [{"id": "onnx", "format": "onnx", "path": artifact.name,
                    "flavor": flavor, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "backend": "onnxruntime", "device": "any"}]}
        (root / f"{name}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def golden_classification(root: Path) -> None:
    """Independent TFLite oracle consumed by the native ONNX integration test."""
    rng = np.random.default_rng(19)
    hand = rng.uniform(-1, 1, (1, 21, 3)).astype(np.float32)
    world = rng.uniform(-1, 1, (1, 21, 3)).astype(np.float32)
    handedness = np.array([[0.8]], dtype=np.float32)
    embedder = tf.lite.Interpreter(model_path=str(root / "gesture_embedder.tflite"), num_threads=1)
    embedder.allocate_tensors()
    values = {"hand": hand, "world_hand": world, "handedness": handedness}
    for tensor in embedder.get_input_details():
        embedder.set_tensor(tensor["index"], values[tensor["name"]])
    embedder.invoke()
    embedding = embedder.get_tensor(embedder.get_output_details()[0]["index"])
    classifier = tf.lite.Interpreter(model_path=str(root / "gesture_classifier.tflite"), num_threads=1)
    classifier.allocate_tensors()
    classifier.set_tensor(classifier.get_input_details()[0]["index"], embedding)
    classifier.invoke()
    scores = classifier.get_tensor(classifier.get_output_details()[0]["index"])
    np.concatenate([hand.ravel(), world.ravel(), handedness.ravel(), scores.ravel()]).astype("<f4").tofile(root / "classification-golden.f32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory; existing directories are never overwritten")
    args = parser.parse_args()
    if args.bundle.stat().st_size > MAX_MEMBER_BYTES:
        raise ValueError("oversize task bundle")
    payload = args.bundle.read_bytes()
    if hashlib.sha256(payload).hexdigest() != BUNDLE_SHA256:
        raise ValueError("bundle differs from the pinned official float16/1 artifact")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"bundle_sha256": BUNDLE_SHA256, "opset": OPSET,
              "parity_cases": PARITY_CASES, "atol": ABS_TOLERANCE,
              "rtol": REL_TOLERANCE, "models": {}}
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        for name, (task, member) in MEMBERS.items():
            with zipfile.ZipFile(io.BytesIO(read_member(bundle, task))) as subtask:
                model = read_member(subtask, member)
            source, target = args.output / f"{name}.tflite", args.output / f"{name}.onnx"
            source.write_bytes(model)
            tf2onnx.convert.from_tflite(str(source), opset=OPSET, output_path=str(target))
            onnx.checker.check_model(onnx.load(target))
            result = verify(source, target)
            result["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            report["models"][name] = result
            print(name, json.dumps(result), flush=True)
    # Presence of this report means every conversion passed; partial directories
    # are not accepted as a deployable model package.
    package_models(args.output)
    golden_classification(args.output)
    (args.output / "parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
