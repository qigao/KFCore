from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from benchmark import (
    BenchmarkConfig,
    DatasetManifest,
    RelationBenchmark,
    RelationExample,
)
from losses import RelationLossConfig, supervised_relation_loss
from model import KFRelationModel


@dataclass(frozen=True)
class FrozenBaselineConfig:
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    seed: int = 20260929

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.learning_rate > 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_disjoint_splits(
    train_manifest: DatasetManifest,
    validation_manifest: DatasetManifest,
) -> None:
    train_images = {example.image for example in train_manifest.examples}
    validation_images = {
        example.image for example in validation_manifest.examples
    }
    overlap = sorted(train_images & validation_images)
    if overlap:
        raise ValueError(
            f"train/validation image leakage: {overlap[0]}"
        )


def _resolve_image(image_root: Path, relative: str) -> Path:
    root = image_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("image path escapes image root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_square_rgb(
    example: RelationExample,
    image_root: Path,
    image_size: int,
) -> Tensor:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    path = _resolve_image(image_root, example.image)
    with Image.open(path) as image:
        if image.size != (example.width, example.height):
            raise ValueError(
                f"decoded image size for {example.image} does not match manifest"
            )
        rgb = image.convert("RGB")
        resized = rgb.resize(
            (image_size, image_size),
            resample=Image.Resampling.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.float32).copy()
    array /= 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def prepare_example(
    example: RelationExample,
    *,
    image_root: str | Path,
    image_size: int,
    max_boxes: int,
    predicate_count: int,
    object_label_to_index: dict[str, int] | None = None,
) -> dict[str, Tensor]:
    if max_boxes < 2:
        raise ValueError("max_boxes must be at least 2")
    if predicate_count <= 0:
        raise ValueError("predicate_count must be positive")
    if len(example.boxes_xyxy) > max_boxes:
        raise ValueError(
            f"{example.image} has {len(example.boxes_xyxy)} boxes; "
            f"max_boxes is {max_boxes}"
        )

    image = _load_square_rgb(
        example,
        Path(image_root),
        image_size,
    )
    boxes = torch.zeros((max_boxes, 4), dtype=torch.float32)
    for index, (left, top, right, bottom) in enumerate(
        example.boxes_xyxy
    ):
        cx = ((left + right) * 0.5) / example.width
        cy = ((top + bottom) * 0.5) / example.height
        width = (right - left) / example.width
        height = (bottom - top) / example.height
        boxes[index] = torch.tensor(
            (cx, cy, width, height),
            dtype=torch.float32,
        )

    object_label_indices = torch.full(
        (max_boxes,),
        -1,
        dtype=torch.int64,
    )
    if object_label_to_index is not None:
        for index, label in enumerate(example.object_labels):
            if label not in object_label_to_index:
                raise ValueError(
                    f"unknown object label for training: {label}"
                )
            object_label_indices[index] = int(
                object_label_to_index[label]
            )

    pair_targets = torch.zeros(
        (max_boxes, max_boxes), dtype=torch.float32
    )
    predicate_targets = torch.zeros(
        (max_boxes, max_boxes, predicate_count),
        dtype=torch.float32,
    )
    for subject, predicate, object_ in example.relations:
        if predicate >= predicate_count:
            raise ValueError("relation predicate exceeds model vocabulary")
        pair_targets[subject, object_] = 1.0
        predicate_targets[subject, object_, predicate] = 1.0

    return {
        "image": image,
        "boxes": boxes,
        "box_count": torch.tensor(
            len(example.boxes_xyxy), dtype=torch.int64
        ),
        "pair_targets": pair_targets,
        "predicate_targets": predicate_targets,
        "object_label_indices": object_label_indices,
        "source_id": torch.tensor(
            example.source_id,
            dtype=torch.int64,
        ),
    }


class RelationTrainingDataset(Dataset):
    def __init__(
        self,
        manifest: DatasetManifest,
        *,
        image_root: str | Path,
        image_size: int,
        max_boxes: int,
        predicate_count: int,
        object_labels: tuple[str, ...] = (),
    ) -> None:
        self.manifest = manifest
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.max_boxes = max_boxes
        self.predicate_count = predicate_count
        self.object_label_to_index = (
            {
                name: index
                for index, name in enumerate(object_labels)
            }
            if object_labels
            else None
        )

    def __len__(self) -> int:
        return len(self.manifest.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return prepare_example(
            self.manifest.examples[index],
            image_root=self.image_root,
            image_size=self.image_size,
            max_boxes=self.max_boxes,
            predicate_count=self.predicate_count,
            object_label_to_index=self.object_label_to_index,
        )


@dataclass(frozen=True)
class PredicateWeighting:
    mode: str
    positive_pair_count: int
    predicate_positive_counts: tuple[int, ...]
    zero_support_predicate_indices: tuple[int, ...]
    positive_weights: tuple[float, ...]
    cap: float

    def tensor(self) -> Tensor:
        return torch.tensor(
            self.positive_weights,
            dtype=torch.float32,
        )


def build_predicate_weighting(
    manifest: DatasetManifest,
    *,
    predicate_count: int,
    mode: str = "none",
    cap: float = 20.0,
) -> PredicateWeighting:
    if predicate_count <= 0:
        raise ValueError("predicate_count must be positive")
    if mode not in {"none", "sqrt-balanced", "balanced"}:
        raise ValueError(
            "predicate weighting mode must be none/sqrt-balanced/balanced"
        )
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("predicate weight cap must be finite and positive")

    positive_pair_count = 0
    counts = [0 for _ in range(predicate_count)]
    for example in manifest.examples:
        pairs: set[tuple[int, int]] = set()
        for subject, predicate, object_ in example.relations:
            if predicate < 0 or predicate >= predicate_count:
                raise ValueError(
                    "relation predicate exceeds weighting vocabulary"
                )
            pairs.add((subject, object_))
            counts[predicate] += 1
        positive_pair_count += len(pairs)

    if positive_pair_count <= 0:
        raise ValueError(
            "predicate weighting requires positive relation pairs"
        )
    zero_support = tuple(
        index for index, count in enumerate(counts)
        if count == 0
    )

    weights: list[float] = []
    for count in counts:
        if mode == "none" or count == 0:
            # A zero-support predicate has no positive BCE terms in this
            # training split, so pos_weight cannot create supervision.
            # Keep it neutral and report the unsupported index explicitly.
            weight = 1.0
        else:
            ratio = (positive_pair_count - count) / count
            if not np.isfinite(ratio) or ratio <= 0.0:
                raise ValueError(
                    "predicate balance ratio must be finite and positive"
                )
            if mode == "sqrt-balanced":
                weight = float(np.sqrt(ratio))
            else:
                weight = float(ratio)
            weight = min(weight, cap)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                "predicate positive weight must be finite and positive"
            )
        weights.append(weight)

    return PredicateWeighting(
        mode=mode,
        positive_pair_count=positive_pair_count,
        predicate_positive_counts=tuple(counts),
        zero_support_predicate_indices=zero_support,
        positive_weights=tuple(weights),
        cap=float(cap),
    )


def build_zero_support_negative_weights(
    *,
    predicate_count: int,
    zero_support_predicate_indices: tuple[int, ...],
    zero_support_negative_weight: float = 1.0,
) -> Tensor:
    if predicate_count <= 0:
        raise ValueError("predicate_count must be positive")
    if (
        not np.isfinite(zero_support_negative_weight)
        or zero_support_negative_weight < 0.0
        or zero_support_negative_weight > 1.0
    ):
        raise ValueError(
            "zero_support_negative_weight must be finite within [0,1]"
        )

    result = torch.ones(predicate_count, dtype=torch.float32)
    seen: set[int] = set()
    for index in zero_support_predicate_indices:
        if index in seen:
            raise ValueError("zero-support predicate indices must be unique")
        seen.add(index)
        if index < 0 or index >= predicate_count:
            raise ValueError(
                "zero-support predicate index is outside vocabulary"
            )
        result[index] = float(zero_support_negative_weight)
    return result


def freeze_backbone(model: KFRelationModel) -> None:
    model.backbone.requires_grad_(False)
    model.backbone.eval()


def trainable_parameters(model: KFRelationModel) -> list[Tensor]:
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


def make_training_loader(
    dataset: RelationTrainingDataset,
    config: FrozenBaselineConfig,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )


def train_epoch(
    model: KFRelationModel,
    loader: Iterable[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    loss_config: RelationLossConfig = RelationLossConfig(),
    predicate_positive_weights: Tensor | None = None,
    predicate_supervision_mask: Tensor | None = None,
    predicate_negative_weights: Tensor | None = None,
    explicit_holdout_mask: Tensor | None = None,
    explicit_holdout_row_policy: str = "dimension-only",
    predicate_contrastive_negative_mask: Tensor | None = None,
    apache_objective: torch.nn.Module | None = None,
) -> dict[str, float]:
    model.train()
    model.backbone.eval()

    sums = {
        "loss": 0.0,
        "sampler_loss": 0.0,
        "pair_loss": 0.0,
        "predicate_loss": 0.0,
    }
    diagnostic_sums = {
        "predicate_contrast_set_size": 0.0,
        "predicate_positive_cosine": 0.0,
        "predicate_hard_negative_margin": 0.0,
        "predicate_query_raw_norm": 0.0,
        "predicate_unobserved_column_fraction": 0.0,
        "predicate_hard_negative_count": 0.0,
        "predicate_contrastive_loss": 0.0,
        "predicate_calibration_loss": 0.0,
        "predicate_calibration_rows": 0.0,
        "predicate_calibration_rows_skipped": 0.0,
        "predicate_calibration_column_fraction": 0.0,
    }
    examples = 0
    predicate_rows = 0
    predicate_rows_skipped = 0
    apache_sums: dict[str, float] = {}

    for batch in loader:
        image = batch["image"].to(device)
        boxes = batch["boxes"].to(device)
        box_counts = batch["box_count"].to(device)
        pair_targets = batch["pair_targets"].to(device)
        predicate_targets = batch["predicate_targets"].to(device)
        object_label_indices = batch.get(
            "object_label_indices"
        )
        if object_label_indices is not None:
            object_label_indices = object_label_indices.to(device)
        batch_size = int(image.shape[0])

        optimizer.zero_grad(set_to_none=True)
        outputs = model.forward_training(
            image,
            boxes,
            box_counts,
            pair_targets=pair_targets,
            entity_labels=(
                object_label_indices
                if apache_objective is not None
                else None
            ),
        )
        if apache_objective is not None:
            source_ids = batch["source_id"].to(device)
            losses = apache_objective(
                outputs,
                predicate_targets,
                source_ids=source_ids,
                object_label_indices=object_label_indices,
            )
        else:
            losses = supervised_relation_loss(
                outputs,
                pair_targets,
                predicate_targets,
                loss_config,
                predicate_positive_weights=predicate_positive_weights,
                predicate_supervision_mask=predicate_supervision_mask,
                predicate_negative_weights=predicate_negative_weights,
                explicit_holdout_mask=explicit_holdout_mask,
                explicit_holdout_row_policy=explicit_holdout_row_policy,
                predicate_contrastive_negative_mask=(
                    predicate_contrastive_negative_mask
                ),
            )
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError("training loss became non-finite")
        losses["loss"].backward()

        for parameter in trainable_parameters(model):
            if parameter.grad is not None and not torch.isfinite(
                parameter.grad
            ).all():
                raise RuntimeError("training gradient became non-finite")

        optimizer.step()
        examples += batch_size
        if apache_objective is not None:
            for key, value in losses.items():
                if not isinstance(value, Tensor) or value.ndim != 0:
                    raise ValueError(
                        "Apache objective diagnostics must be scalar tensors"
                    )
                apache_sums[key] = (
                    apache_sums.get(key, 0.0)
                    + float(value.detach().cpu()) * batch_size
                )
        else:
            predicate_rows += int(
                losses["predicate_rows"].detach().cpu().item()
            )
            predicate_rows_skipped += int(
                losses["predicate_rows_skipped"].detach().cpu().item()
            )
            for key in sums:
                sums[key] += (
                    float(losses[key].detach().cpu()) * batch_size
                )
            for key in diagnostic_sums:
                diagnostic_sums[key] += (
                    float(losses[key].detach().cpu()) * batch_size
                )

    if examples == 0:
        raise ValueError("training loader produced no examples")
    if apache_objective is not None:
        return {
            key: value / examples
            for key, value in apache_sums.items()
        }

    report = {
        key: value / examples
        for key, value in sums.items()
    }
    report["predicate_rows"] = float(predicate_rows)
    report["predicate_rows_skipped"] = float(
        predicate_rows_skipped
    )
    for key, value in diagnostic_sums.items():
        report[key] = value / examples
    return report


def evaluate_gt_boxes(
    model: KFRelationModel,
    manifest: DatasetManifest,
    *,
    image_root: str | Path,
    device: torch.device,
    benchmark_config: BenchmarkConfig = BenchmarkConfig(),
    train_predicate_support: tuple[int, ...] | None = None,
    explicit_holdout_predicate_indices: tuple[int, ...] | None = None,
) -> dict[str, object]:
    benchmark = RelationBenchmark(
        predicate_count=int(model.predicate_bank.shape[0]),
        config=benchmark_config,
    )
    model.eval()
    with torch.inference_mode():
        for example in manifest.examples:
            prepared = prepare_example(
                example,
                image_root=image_root,
                image_size=model.config.image_size,
                max_boxes=model.config.max_boxes,
                predicate_count=int(model.predicate_bank.shape[0]),
            )
            runtime = model(
                prepared["image"].unsqueeze(0).to(device),
                prepared["boxes"].unsqueeze(0).to(device),
                prepared["box_count"].reshape(1).to(device),
            )
            benchmark.add(runtime, example.relations)

    return benchmark.report(
        annotations_sha256=manifest.annotations_sha256,
        vocabulary_sha256=manifest.vocabulary_sha256,
        train_predicate_support=train_predicate_support,
        explicit_holdout_predicate_indices=(
            explicit_holdout_predicate_indices
        ),
    )


def config_payload(config: FrozenBaselineConfig) -> dict[str, object]:
    return asdict(config)
