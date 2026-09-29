from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_predicate_objective_v2 import (
    HOLDOUT_NAMES,
    _assert_group_close,
    _metric_group,
)


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
    "retained_seen_mr20",
    "retained_seen_mr50",
    "retained_seen_mr100",
    "natural_zero_mr20",
    "natural_zero_mr50",
    "natural_zero_mr100",
    "holdout_mr20",
    "holdout_mr50",
    "holdout_mr100",
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
        vocabulary = json.loads(
            (root / "vocabulary.json").read_text(encoding="utf-8")
        )

        names = vocabulary["predicates"]
        holdout_indices = sorted(
            names.index(name) for name in HOLDOUT_NAMES
        )
        original_support = training[
            "predicate_weighting"
        ]["predicate_positive_counts"]
        per_predicate = benchmark["per_predicate"]
        retained_seen_indices = [
            index
            for index, item in enumerate(per_predicate)
            if (
                item["support"] > 0
                and original_support[index] > 0
                and index not in holdout_indices
            )
        ]
        natural_zero_indices = [
            index
            for index, item in enumerate(per_predicate)
            if item["support"] > 0 and original_support[index] == 0
        ]

        holdout_eval = _metric_group(
            benchmark, holdout_indices
        )
        retained_seen_eval = _metric_group(
            benchmark, retained_seen_indices
        )
        natural_zero_eval = _metric_group(
            benchmark, natural_zero_indices
        )
        groups = benchmark["predicate_groups"]
        _assert_group_close(
            groups["train_zero_support"],
            natural_zero_eval,
            f"{arm['arm']} train_zero_support",
        )
        _assert_group_close(
            groups["explicit_holdout"],
            holdout_eval,
            f"{arm['arm']} explicit_holdout",
        )
        _assert_group_close(
            groups["seen"],
            retained_seen_eval,
            f"{arm['arm']} seen",
        )

        history = training["history"]
        rows[str(arm["arm"])] = {
            "arm": arm["arm"],
            "calibration_weight": float(
                arm["calibration_weight"]
            ),
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
            "model_config": training["model_config"],
            "baseline_config": training["baseline_config"],
            "benchmark_config": training["benchmark_config"],
            "predicate_weighting": training["predicate_weighting"],
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
            "contrast_set_size": history[-1][
                "predicate_contrast_set_size"
            ],
            "hard_negative_count": history[-1][
                "predicate_hard_negative_count"
            ],
            "calibration_loss": history[-1][
                "predicate_calibration_loss"
            ],
            "calibration_rows": history[-1][
                "predicate_calibration_rows"
            ],
            "calibration_rows_skipped": history[-1][
                "predicate_calibration_rows_skipped"
            ],
            "calibration_column_fraction": history[-1][
                "predicate_calibration_column_fraction"
            ],
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
            "retained_seen_mr20": retained_seen_eval[
                "mean_recall_at_k"
            ]["20"],
            "retained_seen_mr50": retained_seen_eval[
                "mean_recall_at_k"
            ]["50"],
            "retained_seen_mr100": retained_seen_eval[
                "mean_recall_at_k"
            ]["100"],
            "natural_zero_mr20": natural_zero_eval[
                "mean_recall_at_k"
            ]["20"],
            "natural_zero_mr50": natural_zero_eval[
                "mean_recall_at_k"
            ]["50"],
            "natural_zero_mr100": natural_zero_eval[
                "mean_recall_at_k"
            ]["100"],
            "holdout_mr20": holdout_eval[
                "mean_recall_at_k"
            ]["20"],
            "holdout_mr50": holdout_eval[
                "mean_recall_at_k"
            ]["50"],
            "holdout_mr100": holdout_eval[
                "mean_recall_at_k"
            ]["100"],
        }
    return rows


def compare(arms_dir: Path) -> dict[str, object]:
    rows = load_rows(arms_dir)
    expected = {"cal0", "cal005", "cal010", "cal025"}
    if set(rows) != expected:
        raise RuntimeError(
            f"unexpected calibration sweep arms: {sorted(rows)}"
        )

    weights = {
        "cal0": 0.0,
        "cal005": 0.05,
        "cal010": 0.10,
        "cal025": 0.25,
    }
    control = rows["cal0"]

    common_keys = (
        "prototype_tensor_sha256",
        "trainable_parameter_count",
        "train_annotations_sha256",
        "validation_annotations_sha256",
        "vocabulary_sha256",
        "model_config",
        "baseline_config",
        "benchmark_config",
        "predicate_weighting",
        "predicate_adapter_contract",
    )
    for name, row in rows.items():
        if abs(row["calibration_weight"] - weights[name]) > 1.0e-12:
            raise RuntimeError(
                f"{name} calibration weight drift"
            )
        loss = row["loss_config"]
        if loss["predicate_objective"] != "batch-local-infonce":
            raise RuntimeError(f"{name} objective drift")
        if int(loss["predicate_contrastive_hard_negative_count"]) != 8:
            raise RuntimeError(f"{name} hard-negative count drift")
        if float(row["hard_negative_count"]) <= 0.0:
            raise RuntimeError(f"{name} selected no hard negatives")

        for key in common_keys:
            if control[key] != row[key]:
                raise RuntimeError(
                    f"A/B contract mismatch for {name} {key}"
                )

        if weights[name] == 0.0:
            if float(row["calibration_loss"]) != 0.0:
                raise RuntimeError(
                    "cal0 unexpectedly ran calibration"
                )
        else:
            if float(row["calibration_rows"]) <= 0.0:
                raise RuntimeError(
                    f"{name} calibrated no rows"
                )
            fraction = float(
                row["calibration_column_fraction"]
            )
            if not 0.0 < fraction < 1.0:
                raise RuntimeError(
                    f"{name} invalid calibration column fraction"
                )

    sampler = float(control["sampler_recall"])
    for name, row in rows.items():
        if float(row["sampler_recall"]) != sampler:
            raise RuntimeError(
                f"sampler recall drift in {name}"
            )

    def delta(left: dict[str, object], right: dict[str, object]) -> dict[str, float]:
        return {
            key: float(left[key]) - float(right[key])
            for key in METRIC_KEYS
        }

    return {
        "schema": "kfcore.predicate-calibration-sweep/1",
        "holdout_predicates": list(HOLDOUT_NAMES),
        "arms": rows,
        "delta_vs_cal0": {
            name: delta(row, control)
            for name, row in rows.items()
            if name != "cal0"
        },
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
