from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import RelationVocabulary
from openimages import (
    build_vocabulary,
    convert_relationship_file,
    image_id_payload,
    load_class_descriptions,
    scan_relationship_files,
    select_subset_image_ids,
    sha256_file,
    subset_relationship_csv,
    stable_manifest_json,
    vocabulary_payload,
)


def _parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "source must be SPLIT=PATH"
        )
    split, raw_path = value.split("=", 1)
    if split not in {"train", "validation", "test"}:
        raise argparse.ArgumentTypeError(
            "source split must be train/validation/test"
        )
    path = Path(raw_path)
    if not raw_path:
        raise argparse.ArgumentTypeError("source path must not be empty")
    return split, path


def _ensure_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def scan(args: argparse.Namespace) -> None:
    sources = [_parse_source(value) for value in args.source]
    splits = [split for split, _ in sources]
    if len(set(splits)) != len(splits):
        raise ValueError("each Open Images split may appear only once")

    classes = load_class_descriptions(args.class_descriptions)
    summaries = [
        scan_relationship_files([path])
        for _, path in sources
    ]
    vocabulary = build_vocabulary(summaries, classes)

    output_dir = Path(args.output_dir)
    _ensure_new_directory(output_dir)

    (output_dir / "vocabulary.json").write_text(
        vocabulary_payload(vocabulary),
        encoding="utf-8",
    )

    source_reports = []
    for (split, path), summary in zip(sources, summaries):
        (output_dir / f"{split}.image_ids.txt").write_text(
            image_id_payload(split, summary.images),
            encoding="utf-8",
        )
        source_reports.append(
            {
                "split": split,
                "path": path.name,
                "sha256": sha256_file(path),
                "rows": summary.relationships,
                "object_relationships": summary.object_relationships,
                "skipped_attribute_rows": summary.skipped_attributes,
                "images": len(summary.images),
                "predicates": len(summary.predicates),
                "object_mids": len(summary.object_mids),
            }
        )

    manifest = {
        "schema": "kfcore.openimages-v7-scan/1",
        "class_descriptions": Path(args.class_descriptions).name,
        "class_descriptions_sha256": sha256_file(args.class_descriptions),
        "vocabulary_sha256": vocabulary.sha256(),
        "predicate_count": len(vocabulary.predicates),
        "object_label_count": len(vocabulary.object_labels),
        "sources": source_reports,
    }
    (output_dir / "scan.manifest.json").write_text(
        stable_manifest_json(manifest),
        encoding="utf-8",
    )


def convert(args: argparse.Namespace) -> None:
    vocabulary = RelationVocabulary.load(args.vocabulary)
    classes = load_class_descriptions(args.class_descriptions)

    output_dir = Path(args.output_dir)
    _ensure_new_directory(output_dir)

    output, manifest = convert_relationship_file(
        args.relationships,
        split=args.split,
        image_root=args.image_root,
        class_descriptions=classes,
        vocabulary=vocabulary,
    )
    (output_dir / f"{args.split}.jsonl").write_text(
        output, encoding="utf-8"
    )
    (output_dir / f"{args.split}.manifest.json").write_text(
        stable_manifest_json(manifest),
        encoding="utf-8",
    )

    summary = {
        "split": args.split,
        "images": manifest["images"],
        "relations": manifest["relations"],
        "skipped_attribute_rows": manifest["skipped_attribute_rows"],
        "output_sha256": manifest["output_sha256"],
    }
    print(json.dumps(summary, sort_keys=True))


def subset(args: argparse.Namespace) -> None:
    selected = select_subset_image_ids(
        args.relationships,
        max_images=args.max_images,
        max_boxes=args.max_boxes,
    )
    output_path = Path(args.output_csv)
    if output_path.exists():
        raise FileExistsError(f"output CSV already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        subset_relationship_csv(args.relationships, selected),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "images": len(selected),
                "max_boxes": args.max_boxes,
                "output_csv": str(output_path),
                "output_sha256": sha256_file(output_path),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Open Images V7 visual relationships into the "
            "KFCore canonical GT-box relation format."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser(
        "scan",
        help=(
            "Build one shared vocabulary and per-split image-ID lists "
            "before downloading images."
        ),
    )
    scan_parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Repeat as SPLIT=RELATIONSHIPS.csv",
    )
    scan_parser.add_argument("--class-descriptions", required=True)
    scan_parser.add_argument("--output-dir", required=True)
    scan_parser.set_defaults(func=scan)

    subset_parser = subparsers.add_parser(
        "subset",
        help=(
            "Select a deterministic lexicographic image slice that fits "
            "the requested max_boxes budget."
        ),
    )
    subset_parser.add_argument(
        "--split",
        required=True,
        choices=("train", "validation", "test"),
    )
    subset_parser.add_argument("--relationships", required=True)
    subset_parser.add_argument("--max-images", type=int, required=True)
    subset_parser.add_argument("--max-boxes", type=int, required=True)
    subset_parser.add_argument("--output-csv", required=True)
    subset_parser.set_defaults(func=subset)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert one downloaded split into canonical JSONL.",
    )
    convert_parser.add_argument(
        "--split",
        required=True,
        choices=("train", "validation", "test"),
    )
    convert_parser.add_argument("--relationships", required=True)
    convert_parser.add_argument("--class-descriptions", required=True)
    convert_parser.add_argument("--vocabulary", required=True)
    convert_parser.add_argument("--image-root", required=True)
    convert_parser.add_argument("--output-dir", required=True)
    convert_parser.set_defaults(func=convert)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
