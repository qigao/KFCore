from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_ARMS = {
    "alpha1": 1.0,
    "alpha05": 0.5,
    "alpha025": 0.25,
    "alpha01": 0.1,
    "alpha0": 0.0,
}


def _numeric_dict_close(
    left: dict[str, object],
    right: dict[str, object],
    *,
    subject: str,
) -> None:
    if set(left) != set(right):
        raise RuntimeError(f"{subject} fields differ")
    for key in sorted(left):
        if not math.isclose(
            float(left[key]),
            float(right[key]),
            rel_tol=1.0e-6,
            abs_tol=1.0e-8,
        ):
            raise RuntimeError(
                f"{subject}.{key} mismatch: {left[key]} != {right[key]}"
            )


def _adapter_contract_matches(
    adapter: dict[str, object],
    anchor: dict[str, object],
    *,
    arm: str,
) -> None:
    for key in (
        "rank",
        "parameter_count",
        "source_tensor_sha256",
        "effective_tensor_sha256",
    ):
        if adapter[key] != anchor[key]:
            raise RuntimeError(
                f"adapter contract mismatch for {arm} {key}"
            )

    for key in (
        "mean_row_cosine_to_source",
        "min_row_cosine_to_source",
    ):
        if not math.isclose(
            float(adapter[key]),
            float(anchor[key]),
            rel_tol=1.0e-6,
            abs_tol=1.0e-8,
        ):
            raise RuntimeError(
                f"adapter diagnostic mismatch for {arm} {key}"
            )

    for gram_name in ("source_gram", "effective_gram"):
        left = adapter[gram_name]
        right = anchor[gram_name]
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise RuntimeError(
                f"adapter Gram diagnostics malformed for {arm} {gram_name}"
            )
        _numeric_dict_close(
            left,
            right,
            subject=f"{arm}.{gram_name}",
        )


def compare_sweep(arms_dir: str | Path) -> dict[str, object]:
    root_dir = Path(arms_dir)
    rows: dict[str, dict[str, object]] = {}

    for arm_json in sorted(root_dir.glob("*/arm.json")):
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

        row = {
            "arm": arm["arm"],
            "alpha": arm["zero_support_negative_weight"],
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
            "predicate_weighting": training["predicate_weighting"],
            "predicate_supervision": training["predicate_supervision"],
            "predicate_negative_weighting": training[
                "predicate_negative_weighting"
            ],
            "predicate_adapter": training["predicate_adapter"],
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
            "seen_mr20": groups["seen"]["mean_recall_at_k"]["20"],
            "seen_mr50": groups["seen"]["mean_recall_at_k"]["50"],
            "seen_mr100": groups["seen"]["mean_recall_at_k"]["100"],
            "zero_shot_mr20": groups[
                "train_zero_support"
            ]["mean_recall_at_k"]["20"],
            "zero_shot_mr50": groups[
                "train_zero_support"
            ]["mean_recall_at_k"]["50"],
            "zero_shot_mr100": groups[
                "train_zero_support"
            ]["mean_recall_at_k"]["100"],
        }
        rows[str(arm["arm"])] = row

    if set(rows) != set(EXPECTED_ARMS):
        raise RuntimeError(
            f"unexpected sweep arms: {sorted(rows)}"
        )

    anchor = rows["alpha1"]
    sampler_values = {
        row["sampler_recall"] for row in rows.values()
    }
    if len(sampler_values) != 1:
        raise RuntimeError(
            "sampler recall drifted across loss-only arms: "
            f"{sorted(sampler_values)}"
        )

    zero_support = anchor["predicate_weighting"][
        "zero_support_predicate_indices"
    ]
    if zero_support != [2, 3, 4]:
        raise RuntimeError(
            f"zero-support predicate set drifted: {zero_support}"
        )

    for name, alpha in EXPECTED_ARMS.items():
        row = rows[name]
        if row["alpha"] != alpha:
            raise RuntimeError(
                f"arm alpha mismatch for {name}: "
                f"{row['alpha']} != {alpha}"
            )

        for key in (
            "prototype_tensor_sha256",
            "trainable_parameter_count",
            "train_annotations_sha256",
            "validation_annotations_sha256",
            "vocabulary_sha256",
            "predicate_weighting",
            "predicate_supervision",
        ):
            if row[key] != anchor[key]:
                raise RuntimeError(
                    f"A/B contract mismatch for {name} {key}"
                )

        _adapter_contract_matches(
            row["predicate_adapter"],
            anchor["predicate_adapter"],
            arm=name,
        )

        negative = row["predicate_negative_weighting"]
        if negative["zero_support_negative_weight"] != alpha:
            raise RuntimeError(
                f"training report alpha mismatch for {name}"
            )
        vector = negative["negative_weights"]
        if len(vector) != 17:
            raise RuntimeError(
                "negative-weight vector width drifted"
            )
        for index, value in enumerate(vector):
            expected_value = (
                alpha if index in zero_support else 1.0
            )
            if abs(float(value) - expected_value) > 1.0e-7:
                raise RuntimeError(
                    f"negative weight mismatch {name}[{index}]: "
                    f"{value} != {expected_value}"
                )

    metric_keys = (
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
        "zero_shot_mr20",
        "zero_shot_mr50",
        "zero_shot_mr100",
    )

    comparison = []
    for name, _ in sorted(
        EXPECTED_ARMS.items(),
        key=lambda item: -item[1],
    ):
        row = rows[name]
        item = dict(row)
        item["delta_vs_alpha1"] = {
            key: row[key] - anchor[key]
            for key in metric_keys
        }
        comparison.append(item)

    return {
        "schema": "kfcore.zero-support-negative-sweep/1",
        "anchor": "alpha1",
        "arms": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = compare_sweep(args.arms_dir)
    output = Path(args.out)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
