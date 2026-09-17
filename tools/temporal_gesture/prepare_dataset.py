from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from label_contract import GESTURE_LABEL_CONTRACT

MANIFEST_SCHEMA = "kfcore-gesture-source-manifest/1"
SOURCE_DATASET = "jester-v1"
DEFAULT_DATASET_ROOT = Path(r"F:\KFCoreDatasets\temporal_gesture")
SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
JESTER_LABEL_MAP = {
    "No gesture": (0, "none", "hard_negative"),
    "Doing other things": (0, "none", "hard_negative"),
    "Swiping Left": (1, "swipe_left", "positive"),
    "Swiping Right": (2, "swipe_right", "positive"),
}


@dataclass(frozen=True)
class SourceSample:
    source_dataset: str
    source_split: str
    source_sample_id: str
    source_label: str
    source_path: Path
    frame_count: int
    target_gesture_label: int
    target_gesture_name: str
    role: str
    subject_id: Optional[str]
    usage: str


@dataclass(frozen=True)
class SelectionResult:
    samples: Tuple[SourceSample, ...]
    skipped_by_label: Dict[str, int]


def _validate_sample_id(value: str, line_number: int) -> str:
    sample_id = value.strip()
    if (
        not sample_id
        or sample_id in {".", ".."}
        or sample_id != value
        or "/" in sample_id
        or "\\" in sample_id
        or ":" in sample_id
    ):
        raise ValueError(f"line {line_number}: invalid source sample id {value!r}")
    return sample_id


def _sample_sort_key(sample: SourceSample) -> tuple[int, object]:
    sample_id = sample.source_sample_id
    if sample_id.isdecimal():
        return (0, int(sample_id))
    return (1, sample_id)


def select_jester_samples(
    labels_csv: Path, frames_root: Path, source_split: str
) -> SelectionResult:
    source_split = source_split.strip()
    if not source_split:
        raise ValueError("source_split must not be empty")
    if (
        source_split in {".", ".."}
        or any(character in source_split for character in "/\\:")
        or not all(character.isalnum() or character in "-_" for character in source_split)
    ):
        raise ValueError(f"invalid source_split: {source_split!r}")

    labels_csv = labels_csv.resolve(strict=True)
    if not labels_csv.is_file():
        raise ValueError(f"labels path is not a file: {labels_csv}")
    frames_root = frames_root.resolve(strict=True)
    if not frames_root.is_dir():
        raise ValueError(f"frames root is not a directory: {frames_root}")

    samples: List[SourceSample] = []
    skipped_by_label: Dict[str, int] = {}
    seen_sample_ids: set[str] = set()
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=";")
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(
                    f"{labels_csv}:{line_number}: expected two semicolon-delimited fields"
                )
            sample_id = _validate_sample_id(row[0], line_number)
            if sample_id in seen_sample_ids:
                raise ValueError(
                    f"{labels_csv}:{line_number}: duplicate source sample id '{sample_id}'"
                )
            seen_sample_ids.add(sample_id)

            source_label = row[1].strip()
            if not source_label:
                raise ValueError(f"{labels_csv}:{line_number}: source label must not be empty")
            target = JESTER_LABEL_MAP.get(source_label)
            if target is None:
                skipped_by_label[source_label] = skipped_by_label.get(source_label, 0) + 1
                continue

            source_path = (frames_root / sample_id).resolve(strict=True)
            if not source_path.is_dir() or source_path.parent != frames_root:
                raise ValueError(
                    f"{labels_csv}:{line_number}: selected sample is not a frame directory: "
                    f"{source_path}"
                )
            frame_count = sum(
                1
                for child in source_path.iterdir()
                if child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            )
            if frame_count == 0:
                raise ValueError(
                    f"{labels_csv}:{line_number}: selected sample '{sample_id}' contains no "
                    "supported image frames"
                )

            target_label, target_name, role = target
            samples.append(
                SourceSample(
                    source_dataset=SOURCE_DATASET,
                    source_split=source_split,
                    source_sample_id=sample_id,
                    source_label=source_label,
                    source_path=source_path,
                    frame_count=frame_count,
                    target_gesture_label=target_label,
                    target_gesture_name=target_name,
                    role=role,
                    subject_id=None,
                    usage="pretrain_only",
                )
            )

    if not seen_sample_ids:
        raise ValueError(f"labels file contains no samples: {labels_csv}")
    if not samples:
        raise ValueError("labels file contains no compatible Jester samples")
    samples.sort(key=_sample_sort_key)
    return SelectionResult(tuple(samples), dict(sorted(skipped_by_label.items())))


def _manifest_record(sample: SourceSample) -> Dict[str, object]:
    return {
        "manifest_schema": MANIFEST_SCHEMA,
        "gesture_label_contract": GESTURE_LABEL_CONTRACT,
        "source_dataset": sample.source_dataset,
        "source_split": sample.source_split,
        "source_sample_id": sample.source_sample_id,
        "source_label": sample.source_label,
        "source_path": str(sample.source_path),
        "frame_count": sample.frame_count,
        "target_gesture_label": sample.target_gesture_label,
        "target_gesture_name": sample.target_gesture_name,
        "role": sample.role,
        "subject_id": sample.subject_id,
        "usage": sample.usage,
    }


def write_manifest(result: SelectionResult, output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary_created = False
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            temporary_created = True
            for sample in result.samples:
                json.dump(
                    _manifest_record(sample),
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(f"output already exists: {output}") from exc
    finally:
        if temporary_created and temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select compatible clips from a local Jester dataset into a KFCore "
            "source manifest. This does not create trainable 78-D records."
        )
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Jester semicolon-delimited split label CSV",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        required=True,
        help="directory containing one decoded-frame directory per Jester sample",
    )
    parser.add_argument(
        "--source-split",
        required=True,
        help="source split name, for example train or validation",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"KFCore dataset workspace (default: {DEFAULT_DATASET_ROOT})",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        result = select_jester_samples(
            arguments.labels, arguments.frames_root, arguments.source_split
        )
        output = (
            arguments.dataset_root
            / "manifests"
            / f"jester-{arguments.source_split.strip()}.jsonl"
        )
        write_manifest(result, output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "selected_samples": len(result.samples),
                "skipped_by_label": result.skipped_by_label,
                "usage": "pretrain_only",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
