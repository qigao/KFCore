from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


HOLDOUT_NAMES = ("contain", "holds", "ride")
TOP_KS = ("20", "50", "100")
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


def _metric_group(benchmark: dict[str, object], indices: list[int]) -> dict[str, object]:
    per_predicate = benchmark["per_predicate"]
    assert isinstance(per_predicate, list)
    by_index = {
        int(item["predicate_index"]): item
        for item in per_predicate
    }
    ordered = sorted(indices)
    if not ordered:
        raise RuntimeError("metric group must not be empty")
    for index in ordered:
        if int(by_index[index]["support"]) <= 0:
            raise RuntimeError(
                f"predicate {index} has no validation support"
            )
    return {
        "predicate_indices": ordered,
        "predicate_count": len(ordered),
        "validation_triplet_support": sum(
            int(by_index[index]["support"])
            for index in ordered
        ),
        "mean_recall_at_k": {
            k: sum(
                float(by_index[index]["recall"][k])
                for index in ordered
            )
            / len(ordered)
            for k in TOP_KS
        },
    }


def _assert_group_close(
    actual: dict[str, object],
    expected: dict[str, object],
    subject: str,
) -> None:
    for key in (
        "predicate_indices",
        "predicate_count",
        "validation_triplet_support",
    ):
        if actual[key] != expected[key]:
            raise RuntimeError(
                f"{subject} {key} drift: "
                f"{actual[key]} != {expected[key]}"
            )

    actual_mean = actual["mean_recall_at_k"]
    expected_mean = expected["mean_recall_at_k"]
    assert isinstance(actual_mean, dict)
    assert isinstance(expected_mean, dict)
    for k in TOP_KS:
        left = actual_mean[k]
        right = expected_mean[k]
        if left is None or right is None:
            if left != right:
                raise RuntimeError(
                    f"{subject} mR@{k} nullability drift"
                )
            continue
        if not math.isclose(
            float(left),
            float(right),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{subject} mR@{k} drift: {left} != {right}"
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

        predicate_names = vocabulary["predicates"]
        holdout_indices = sorted(
            predicate_names.index(name)
            for name in HOLDOUT_NAMES
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
            "objective": arm["predicate_objective"],
            "holdout_enabled": arm["holdout_enabled"],
            "zero_support_negative_weight": arm[
                "zero_support_negative_weight"
            ],
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
            "predicate_adapter": training["predicate_adapter"],
            "loss_config": training["loss_config"],
            "predicate_supervision": training[
                "predicate_supervision"
            ],
            "predicate_rows_skipped": sum(
                int(epoch["predicate_rows_skipped"])
                for epoch in history
            ),
            "contrast_set_size": history[-1][
                "predicate_contrast_set_size"
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
            "unobserved_column_fraction": history[-1][
                "predicate_unobserved_column_fraction"
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
        "bce-baseline",
        "bce-holdout",
        "infonce-baseline",
        "infonce-holdout",
    }
    if set(rows) != expected:
        raise RuntimeError(
            f"unexpected objective arms: {sorted(rows)}"
        )

    bce_base = rows["bce-baseline"]
    bce_hold = rows["bce-holdout"]
    nce_base = rows["infonce-baseline"]
    nce_hold = rows["infonce-holdout"]

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
        "predicate_adapter",
    )
    for candidate in (bce_hold, nce_base, nce_hold):
        for key in common_keys:
            if bce_base[key] != candidate[key]:
                raise RuntimeError(
                    f"A/B contract mismatch for "
                    f"{candidate['arm']} {key}"
                )

    for arm in (bce_base, bce_hold):
        if arm["objective"] != "bce":
            raise RuntimeError("BCE arm objective drift")
        if abs(float(arm["zero_support_negative_weight"]) - 0.1) > 1e-12:
            raise RuntimeError("BCE alpha drift")
        if arm["loss_config"]["predicate_objective"] != "bce":
            raise RuntimeError("BCE loss-config drift")

    for arm in (nce_base, nce_hold):
        if arm["objective"] != "batch-local-infonce":
            raise RuntimeError("InfoNCE arm objective drift")
        if abs(float(arm["zero_support_negative_weight"]) - 1.0) > 1e-12:
            raise RuntimeError("InfoNCE alpha must remain neutral")
        if (
            arm["loss_config"]["predicate_objective"]
            != "batch-local-infonce"
        ):
            raise RuntimeError("InfoNCE loss-config drift")
        if float(arm["contrast_set_size"]) <= 0.0:
            raise RuntimeError("InfoNCE contrast set is empty")
        fraction = float(arm["unobserved_column_fraction"])
        if not 0.0 <= fraction <= 1.0:
            raise RuntimeError("invalid InfoNCE unobserved fraction")

    if bce_base["holdout_enabled"] or nce_base["holdout_enabled"]:
        raise RuntimeError("baseline arm unexpectedly holds out labels")
    if not bce_hold["holdout_enabled"] or not nce_hold["holdout_enabled"]:
        raise RuntimeError("holdout arm did not enable holdout")
    if int(nce_hold["predicate_rows_skipped"]) <= 0:
        raise RuntimeError(
            "InfoNCE holdout did not skip holdout-only rows"
        )

    sampler = float(bce_base["sampler_recall"])
    for candidate in (bce_hold, nce_base, nce_hold):
        if not math.isclose(
            float(candidate["sampler_recall"]),
            sampler,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                f"sampler recall drift in {candidate['arm']}"
            )

    def delta(left: dict[str, object], right: dict[str, object]) -> dict[str, float]:
        return {
            key: float(left[key]) - float(right[key])
            for key in METRIC_KEYS
        }

    return {
        "schema": "kfcore.predicate-objective-v2/1",
        "holdout_predicates": list(HOLDOUT_NAMES),
        "bce_baseline": bce_base,
        "bce_holdout": bce_hold,
        "infonce_baseline": nce_base,
        "infonce_holdout": nce_hold,
        "bce_holdout_minus_baseline": delta(
            bce_hold, bce_base
        ),
        "infonce_holdout_minus_baseline": delta(
            nce_hold, nce_base
        ),
        "infonce_baseline_minus_bce_baseline": delta(
            nce_base, bce_base
        ),
        "infonce_holdout_minus_bce_holdout": delta(
            nce_hold, bce_hold
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
