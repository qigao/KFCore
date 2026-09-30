from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--native-jsonl", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--ort-archive", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--score-tolerance", type=float, default=2.0e-5)
    args = parser.parse_args()

    fixture = Path(args.fixture)
    reference = json.loads(
        (fixture / "reference.json").read_text(encoding="utf-8")
    )
    native = [
        json.loads(line)
        for line in Path(args.native_jsonl)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    expected_cases = reference["cases"]
    if len(native) != len(expected_cases):
        raise RuntimeError(
            f"native case count {len(native)} != {len(expected_cases)}"
        )

    versions = []
    total_ms = []
    for index, (got, expected) in enumerate(zip(native, expected_cases)):
        if got["case"] != expected["case"]:
            raise RuntimeError(
                f"case order drift at {index}: {got['case']} != {expected['case']}"
            )
        if int(got["v"]) != int(expected["v"]):
            raise RuntimeError(f"V drift for {got['case']}")
        if got["backend"] != "onnxruntime" or got["device"] != "cpu":
            raise RuntimeError("native qualification did not use ORT CPU")
        if int(got["model_load_count"]) != 1:
            raise RuntimeError("relation model was reloaded during vocabulary swaps")
        if not bool(got["backend_scoring"]):
            raise RuntimeError("native qualification did not use backend scoring")
        if int(got["scene_object_count"]) != int(reference["region_count"]):
            raise RuntimeError("SceneGraph object count drifted")
        if not bool(got["all_track_ids_null"]):
            raise RuntimeError("GT-box qualification unexpectedly created track ids")
        if int(got["valid_pair_count"]) != int(expected["valid_pair_count"]):
            raise RuntimeError("selected valid-pair count drifted")

        version = int(got["vocabulary_version"])
        versions.append(version)
        if version != index + 1:
            raise RuntimeError(
                f"vocabulary version {version} is not monotonic expected {index + 1}"
            )

        got_edges = got["edges"]
        expected_edges = expected["edges"]
        if len(got_edges) != len(expected_edges):
            raise RuntimeError(
                f"edge count drift for {got['case']}: "
                f"{len(got_edges)} != {len(expected_edges)}"
            )
        for edge_index, (g, e) in enumerate(zip(got_edges, expected_edges)):
            for key in ("subject", "object", "predicate"):
                if int(g[key]) != int(e[key]):
                    raise RuntimeError(
                        f"{got['case']} edge {edge_index} {key} drift: "
                        f"{g[key]} != {e[key]}"
                    )
            if not math.isclose(
                float(g["score"]),
                float(e["score"]),
                rel_tol=0.0,
                abs_tol=args.score_tolerance,
            ):
                raise RuntimeError(
                    f"{got['case']} edge {edge_index} score drift: "
                    f"{g['score']} != {e['score']}"
                )

        timing = got["timing"]
        for key in (
            "preprocess_ms",
            "runtime_ms",
            "scoring_ms",
            "decode_ms",
            "total_ms",
        ):
            value = float(timing[key])
            if not math.isfinite(value) or value < 0.0:
                raise RuntimeError(f"invalid timing {key}={value}")
            if key != "total_ms" and float(timing["total_ms"]) < value:
                raise RuntimeError(
                    f"total latency is smaller than stage {key}"
                )
        # Backend scoring owns predicate scoring inside runtime.
        if float(timing["scoring_ms"]) != 0.0:
            raise RuntimeError(
                "backend-scoring qualification must report host scoring_ms=0"
            )
        total_ms.append(float(timing["total_ms"]))

    expected_v_sequence = [1, 3, 5, 3]
    if [int(item["v"]) for item in native] != expected_v_sequence:
        raise RuntimeError("dynamic vocabulary swap sequence drifted")

    plugin = Path(args.plugin)
    model = fixture / "relation.onnx"
    report = {
        "schema": "kfcore.gtbox-ort-native-qualification/1",
        "backend": "onnxruntime",
        "device": "cpu",
        "model_type": "relation.open-vocabulary",
        "model_sha256": sha256(model),
        "plugin_sha256": sha256(plugin),
        "ort_archive_sha256": (
            sha256(Path(args.ort_archive))
            if args.ort_archive
            else ""
        ),
        "fixture_image_sha256": reference["image_sha256"],
        "vocabulary_sequence": expected_v_sequence,
        "vocabulary_versions": versions,
        "model_load_count": 1,
        "case_count": len(native),
        "max_score_abs_tolerance": args.score_tolerance,
        "mean_total_ms": sum(total_ms) / len(total_ms),
        "cases": native,
    }
    Path(args.out).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
