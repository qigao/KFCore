from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_KEYS = (
    "sampler_recall",
    "pair_ap",
    "predicate_top1_accuracy",
    "r20",
    "r50",
    "r100",
    "mr20",
    "mr50",
    "mr100",
    "seen_mr20",
    "seen_mr50",
    "seen_mr100",
    "natural_zero_mr20",
    "natural_zero_mr50",
    "natural_zero_mr100",
)


def load_rows(arms_dir: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for arm_json in sorted(arms_dir.glob("*/arm.json")):
        root = arm_json.parent
        arm = json.loads(arm_json.read_text(encoding="utf-8"))
        training = json.loads(
            (root / "training.json").read_text(encoding="utf-8")
        )
        benchmark = json.loads(
            (root / "benchmark.json").read_text(encoding="utf-8")
        )
        prototypes = json.loads(
            (root / "prototypes.json").read_text(encoding="utf-8")
        )

        groups = benchmark["predicate_groups"]
        seen = groups["seen"]["mean_recall_at_k"]
        zero = groups["train_zero_support"]["mean_recall_at_k"]
        model_config = dict(training["model_config"])
        geometry_mode = model_config.pop("pair_geometry_evidence")

        rows[str(arm["arm"])] = {
            "arm": arm["arm"],
            "pair_geometry_evidence": geometry_mode,
            "prototype_tensor_sha256": prototypes["tensor_sha256"],
            "trainable_parameter_count": training[
                "trainable_parameter_count"
            ],
            "train_annotations_sha256": training[
                "train_annotations_sha256"
            ],
            "validation_annotations_sha256": training[
                "validation_annotations_sha256"
            ],
            "vocabulary_sha256": training["vocabulary_sha256"],
            "common_model_config": model_config,
            "baseline_config": training["baseline_config"],
            "benchmark_config": training["benchmark_config"],
            "predicate_weighting": training["predicate_weighting"],
            "predicate_supervision": training[
                "predicate_supervision"
            ],
            "predicate_adapter_contract": {
                "rank": training["predicate_adapter"]["rank"],
                "parameter_count": training[
                    "predicate_adapter"
                ]["parameter_count"],
                "source_tensor_sha256": training[
                    "predicate_adapter"
                ]["source_tensor_sha256"],
                "effective_tensor_sha256": training[
                    "predicate_adapter"
                ]["effective_tensor_sha256"],
            },
            "loss_config": training["loss_config"],
            "sampler_recall": benchmark["sampler_recall"],
            "pair_ap": benchmark["pair_ap"],
            "predicate_top1_accuracy": benchmark[
                "predicate_top1_accuracy_on_sampled_pairs"
            ],
            "r20": benchmark["recall_at_k"]["20"],
            "r50": benchmark["recall_at_k"]["50"],
            "r100": benchmark["recall_at_k"]["100"],
            "mr20": benchmark["mean_recall_at_k"]["20"],
            "mr50": benchmark["mean_recall_at_k"]["50"],
            "mr100": benchmark["mean_recall_at_k"]["100"],
            "seen_mr20": seen["20"],
            "seen_mr50": seen["50"],
            "seen_mr100": seen["100"],
            "natural_zero_mr20": zero["20"],
            "natural_zero_mr50": zero["50"],
            "natural_zero_mr100": zero["100"],
        }
    return rows


def compare(arms_dir: Path) -> dict[str, object]:
    rows = load_rows(arms_dir)
    expected = {"basic", "rich"}
    if set(rows) != expected:
        raise RuntimeError(
            f"unexpected geometry-evidence arms: {sorted(rows)}"
        )

    basic = rows["basic"]
    rich = rows["rich"]

    common_keys = (
        "prototype_tensor_sha256",
        "train_annotations_sha256",
        "validation_annotations_sha256",
        "vocabulary_sha256",
        "common_model_config",
        "baseline_config",
        "benchmark_config",
        "predicate_weighting",
        "predicate_supervision",
        "predicate_adapter_contract",
        "loss_config",
    )
    for key in common_keys:
        if basic[key] != rich[key]:
            raise RuntimeError(
                f"A/B contract mismatch for {key}"
            )

    if basic["pair_geometry_evidence"] != "basic":
        raise RuntimeError("basic geometry mode drift")
    if rich["pair_geometry_evidence"] != "rich":
        raise RuntimeError("rich geometry mode drift")

    common_model = basic["common_model_config"]
    if common_model["pair_visual_evidence"] != "contact":
        raise RuntimeError(
            "geometry A/B must keep contact visual evidence fixed"
        )

    basic_params = int(basic["trainable_parameter_count"])
    rich_params = int(rich["trainable_parameter_count"])
    parameter_delta = rich_params - basic_params
    geometry_dim = int(common_model["geometry_dim"])
    expected_delta = 10 * (geometry_dim + 1)
    if parameter_delta != expected_delta:
        raise RuntimeError(
            "rich geometry parameter delta drifted: "
            f"{parameter_delta} != {expected_delta}"
        )

    loss = basic["loss_config"]
    if (
        loss["predicate_objective"] != "batch-local-infonce"
        or int(loss["predicate_contrastive_hard_negative_count"]) != 8
        or abs(float(loss["predicate_calibration_loss_weight"]) - 0.10)
        > 1.0e-12
    ):
        raise RuntimeError(
            "fixed predicate objective contract drifted"
        )

    def delta(
        left: dict[str, object],
        right: dict[str, object],
    ) -> dict[str, float]:
        return {
            key: float(left[key]) - float(right[key])
            for key in METRIC_KEYS
        }

    return {
        "schema": "kfcore.pair-geometry-evidence/1",
        "basic": basic,
        "rich": rich,
        "rich_minus_basic": delta(rich, basic),
        "rich_parameter_count": parameter_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = compare(Path(args.arms_dir))
    output = Path(args.out)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
