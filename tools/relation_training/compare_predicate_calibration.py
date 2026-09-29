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


def _load_rows(arms_dir: Path) -> dict[str, dict[str, object]]:
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
        if arm["holdout_enabled"]:
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
        elif "explicit_holdout" in groups:
            raise RuntimeError(
                f"{arm['arm']} unexpectedly reports explicit holdout"
            )

        history = training["history"]
        row = {
            "arm": arm["arm"],
            "holdout_enabled": arm["holdout_enabled"],
            "calibration_weight": float(arm["calibration_weight"]),
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
            "predicate_supervision": training[
                "predicate_supervision"
            ],
            "contrast_set_size": history[-1][
                "predicate_contrast_set_size"
            ],
            "hard_negative_count": history[-1][
                "predicate_hard_negative_count"
            ],
            "positive_cosine": history[-1][
                "predicate_positive_cosine"
            ],
            "hard_negative_margin": history[-1][
                "predicate_hard_negative_margin"
            ],
            "query_raw_norm": history[-1][
                "predicate_query_raw_norm"
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
        rows[str(arm["arm"])] = row
    return rows


def compare(arms_dir: Path) -> dict[str, object]:
    rows = _load_rows(arms_dir)
    expected = {
        "hard8-baseline",
        "hard8-holdout",
        "cal025-baseline",
        "cal025-holdout",
    }
    if set(rows) != expected:
        raise RuntimeError(
            f"unexpected calibration arms: {sorted(rows)}"
        )

    hard_base = rows["hard8-baseline"]
    hard_hold = rows["hard8-holdout"]
    cal_base = rows["cal025-baseline"]
    cal_hold = rows["cal025-holdout"]

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
    for candidate in (hard_hold, cal_base, cal_hold):
        for key in common_keys:
            if hard_base[key] != candidate[key]:
                raise RuntimeError(
                    f"A/B contract mismatch for "
                    f"{candidate['arm']} {key}"
                )

    for arm in rows.values():
        loss = arm["loss_config"]
        if loss["predicate_objective"] != "batch-local-infonce":
            raise RuntimeError(
                f"{arm['arm']} is not an InfoNCE arm"
            )
        if int(loss["predicate_contrastive_hard_negative_count"]) != 8:
            raise RuntimeError(
                f"{arm['arm']} hard-negative count drift"
            )
        if float(arm["hard_negative_count"]) <= 0.0:
            raise RuntimeError(
                f"{arm['arm']} selected no hard negatives"
            )

    for arm in (hard_base, hard_hold):
        if arm["calibration_weight"] != 0.0:
            raise RuntimeError(
                f"{arm['arm']} calibration control drift"
            )
        if float(arm["calibration_loss"]) != 0.0:
            raise RuntimeError(
                f"{arm['arm']} unexpectedly ran calibration"
            )

    for arm in (cal_base, cal_hold):
        if abs(arm["calibration_weight"] - 0.25) > 1.0e-12:
            raise RuntimeError(
                f"{arm['arm']} calibration weight drift"
            )
        if float(arm["calibration_rows"]) <= 0.0:
            raise RuntimeError(
                f"{arm['arm']} calibrated no rows"
            )
        fraction = float(arm["calibration_column_fraction"])
        if not 0.0 < fraction < 1.0:
            raise RuntimeError(
                f"{arm['arm']} invalid calibration column fraction"
            )

    if hard_base["holdout_enabled"] or cal_base["holdout_enabled"]:
        raise RuntimeError(
            "baseline arm unexpectedly enables holdout"
        )
    if not hard_hold["holdout_enabled"] or not cal_hold["holdout_enabled"]:
        raise RuntimeError(
            "holdout arm did not enable holdout"
        )

    sampler = float(hard_base["sampler_recall"])
    for candidate in (hard_hold, cal_base, cal_hold):
        if float(candidate["sampler_recall"]) != sampler:
            raise RuntimeError(
                f"sampler recall drift in {candidate['arm']}"
            )

    def delta(left: dict[str, object], right: dict[str, object]) -> dict[str, float]:
        return {
            key: float(left[key]) - float(right[key])
            for key in METRIC_KEYS
        }

    return {
        "schema": "kfcore.predicate-calibration/1",
        "holdout_predicates": list(HOLDOUT_NAMES),
        "hard8_baseline": hard_base,
        "hard8_holdout": hard_hold,
        "cal025_baseline": cal_base,
        "cal025_holdout": cal_hold,
        "cal025_baseline_minus_hard8": delta(
            cal_base, hard_base
        ),
        "cal025_holdout_minus_hard8": delta(
            cal_hold, hard_hold
        ),
        "hard8_holdout_minus_baseline": delta(
            hard_hold, hard_base
        ),
        "cal025_holdout_minus_baseline": delta(
            cal_hold, cal_base
        ),
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
