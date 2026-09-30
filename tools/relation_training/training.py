from __future__ import annotations

from dataclasses import asdict, dataclass
from bisect import bisect_right
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from apache_mixture import LoadedRelationMixture
from apache_multiscale import (
    EpochRandomSampler,
    MultiScaleBatchSampler,
)
from benchmark import (
    BenchmarkConfig,
    DatasetManifest,
    RelationBenchmark,
    RelationExample,
)
from detector_ceiling import (
    DetectorPredictionExample,
    DetectorPredictionManifest,
    DetectorRecoverabilityBenchmark,
    DetectorRecoverabilityConfig,
)
from losses import RelationLossConfig, supervised_relation_loss
from model import KFRelationModel


_LUMA = torch.tensor(
    [0.299, 0.587, 0.114]
).view(3, 1, 1)


def photometric_jitter(
    image: Tensor,
    strength: float,
) -> Tensor:
    """Apache released brightness/contrast/saturation jitter."""
    if strength <= 0.0:
        return image

    def factor() -> float:
        return float(
            1.0
            + (
                torch.rand(()) * 2.0
                - 1.0
            )
            * strength
        )

    image = image * factor()
    mean = image.mean(
        dim=(1, 2),
        keepdim=True,
    )
    image = (
        image - mean
    ) * factor() + mean
    grey = (
        image
        * _LUMA.to(image.dtype)
    ).sum(
        dim=0,
        keepdim=True,
    )
    image = (
        image - grey
    ) * factor() + grey
    return image.clamp_(0.0, 1.0)


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


def prepare_detector_boxes(
    example: DetectorPredictionExample,
    *,
    max_boxes: int,
) -> tuple[Tensor, Tensor]:
    if max_boxes <= 0:
        raise ValueError("max_boxes must be positive")
    limited = example.limit(max_boxes)
    boxes = torch.zeros(
        (max_boxes, 4),
        dtype=torch.float32,
    )
    for index, (left, top, right, bottom) in enumerate(
        limited.boxes_xyxy
    ):
        cx = ((left + right) * 0.5) / example.width
        cy = ((top + bottom) * 0.5) / example.height
        width = (right - left) / example.width
        height = (bottom - top) / example.height
        boxes[index] = torch.tensor(
            (cx, cy, width, height),
            dtype=torch.float32,
        )
    return (
        boxes,
        torch.tensor(
            len(limited.boxes_xyxy),
            dtype=torch.int64,
        ),
    )


def prepare_example(
    example: RelationExample,
    *,
    image_root: str | Path,
    image_size: int,
    max_boxes: int,
    predicate_count: int,
    object_label_to_index: dict[str, int] | None = None,
    augment: float = 0.0,
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
    if augment > 0.0:
        image = photometric_jitter(
            image,
            augment,
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
    cfa_predicate_labels = torch.full(
        (max_boxes, max_boxes),
        -1,
        dtype=torch.int64,
    )
    for subject, predicate, object_ in example.relations:
        if predicate >= predicate_count:
            raise ValueError("relation predicate exceeds model vocabulary")
        pair_targets[subject, object_] = 1.0
        predicate_targets[subject, object_, predicate] = 1.0
        # Apache sampler/CFA contract: last predicate annotation wins for
        # the per-pair feature-mixing label while the loss remains multi-hot.
        cfa_predicate_labels[subject, object_] = int(predicate)

    return {
        "image": image,
        "boxes": boxes,
        "box_count": torch.tensor(
            len(example.boxes_xyxy), dtype=torch.int64
        ),
        "pair_targets": pair_targets,
        "predicate_targets": predicate_targets,
        "cfa_predicate_labels": cfa_predicate_labels,
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
        augment: float = 0.0,
    ) -> None:
        self.manifest = manifest
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.max_boxes = max_boxes
        self.predicate_count = predicate_count
        if (
            not np.isfinite(augment)
            or augment < 0.0
        ):
            raise ValueError(
                "augmentation strength must be finite and non-negative"
            )
        self.augment = float(augment)
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

    def __getitem__(
        self,
        index: int | tuple[int, int],
    ) -> dict[str, Tensor]:
        if isinstance(index, tuple):
            example_index = int(index[0])
            image_size = int(index[1])
        else:
            example_index = int(index)
            image_size = self.image_size
        if image_size <= 0:
            raise ValueError(
                "training image resolution must be positive"
            )
        result = prepare_example(
            self.manifest.examples[example_index],
            image_root=self.image_root,
            image_size=image_size,
            max_boxes=self.max_boxes,
            predicate_count=self.predicate_count,
            object_label_to_index=self.object_label_to_index,
            augment=self.augment,
        )
        result["training_resolution"] = torch.tensor(
            image_size,
            dtype=torch.int64,
        )
        return result


class RelationMixtureTrainingDataset(Dataset):
    """Concatenated relation datasets with source-specific image roots.

    Supports the same tuple index contract used by MultiScaleBatchSampler.
    """

    def __init__(
        self,
        mixture: LoadedRelationMixture,
        *,
        image_size: int,
        max_boxes: int,
        predicate_count: int,
        object_labels: tuple[str, ...] = (),
        augment: float = 0.0,
    ) -> None:
        self.mixture = mixture
        self.datasets = tuple(
            RelationTrainingDataset(
                manifest,
                image_root=source.image_root,
                image_size=image_size,
                max_boxes=max_boxes,
                predicate_count=predicate_count,
                object_labels=object_labels,
                augment=augment,
            )
            for source, manifest in zip(
                mixture.config.sources,
                mixture.manifests,
            )
        )
        cumulative: list[int] = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            cumulative.append(total)
        if total <= 0:
            raise ValueError(
                "relation mixture training dataset must not be empty"
            )
        self.cumulative_sizes = tuple(cumulative)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source = bisect_right(
            self.cumulative_sizes,
            index,
        )
        previous = (
            0
            if source == 0
            else self.cumulative_sizes[source - 1]
        )
        return source, index - previous

    def __getitem__(
        self,
        index: int | tuple[int, int],
    ) -> dict[str, Tensor]:
        if isinstance(index, tuple):
            global_index = int(index[0])
            resolution = int(index[1])
            source, local = self._locate(global_index)
            return self.datasets[source][
                (local, resolution)
            ]
        source, local = self._locate(int(index))
        return self.datasets[source][local]

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
    dataset: Dataset,
    config: FrozenBaselineConfig,
    *,
    resolutions: list[int] | None = None,
    drop_last: bool = False,
    sampler: Sampler[int] | None = None,
    batch_sampler: Sampler[list[int | tuple[int, int]]] | None = None,
) -> DataLoader:
    if batch_sampler is not None:
        if sampler is not None or resolutions:
            raise ValueError(
                "explicit batch_sampler cannot be combined with sampler/resolutions"
            )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
        )

    active_sampler = sampler
    if resolutions:
        if active_sampler is None:
            active_sampler = EpochRandomSampler(
                len(dataset),
                seed=config.seed,
            )
        batch_sampler = MultiScaleBatchSampler(
            active_sampler,
            config.batch_size,
            resolutions,
            drop_last=drop_last,
            seed=config.seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
        )

    if active_sampler is not None:
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=active_sampler,
            shuffle=False,
            num_workers=0,
            drop_last=drop_last,
        )

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=drop_last,
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
    backbone_training: bool = False,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    clip_grad: float | None = None,
    record_gradient_health: bool = False,
    grad_accum: int = 1,
    ema: object | None = None,
) -> dict[str, float]:
    model.train()
    if backbone_training:
        model.backbone.train()
    else:
        model.backbone.eval()
    if clip_grad is not None and (
        not np.isfinite(clip_grad) or clip_grad <= 0.0
    ):
        raise ValueError("clip_grad must be finite and positive")
    if (
        isinstance(grad_accum, bool)
        or not isinstance(grad_accum, int)
        or grad_accum <= 0
    ):
        raise ValueError(
            "grad_accum must be a positive integer"
        )

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
    micro_batches = 0
    optimizer_steps = 0
    resolution_counts: dict[int, int] = {}
    predicate_rows = 0
    predicate_rows_skipped = 0
    apache_sums: dict[str, float] = {}
    gradient_sums = {
        "backbone_gradient_norm": 0.0,
        "head_gradient_norm": 0.0,
        "backbone_gradient_tensors": 0.0,
        "head_gradient_tensors": 0.0,
    }

    try:
        total_batches = len(loader)  # type: ignore[arg-type]
    except TypeError:
        total_batches = None
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        image = batch["image"].to(device)
        boxes = batch["boxes"].to(device)
        box_counts = batch["box_count"].to(device)
        pair_targets = batch["pair_targets"].to(device)
        predicate_targets = batch["predicate_targets"].to(device)
        cfa_predicate_labels = batch.get(
            "cfa_predicate_labels"
        )
        if cfa_predicate_labels is not None:
            cfa_predicate_labels = cfa_predicate_labels.to(device)
        object_label_indices = batch.get(
            "object_label_indices"
        )
        if object_label_indices is not None:
            object_label_indices = object_label_indices.to(device)
        coverage = batch.get("coverage")
        if coverage is not None:
            coverage = coverage.to(device)
        fill = batch.get("fill")
        if fill is not None:
            fill = fill.to(device)
        batch_size = int(image.shape[0])
        micro_batches += 1
        resolution = batch.get("training_resolution")
        if resolution is not None:
            unique_resolution = torch.unique(
                resolution.to(torch.int64)
            )
            if unique_resolution.numel() != 1:
                raise RuntimeError(
                    "multi-scale batch contains mixed image resolutions"
                )
            value = int(unique_resolution.item())
            resolution_counts[value] = (
                resolution_counts.get(value, 0)
                + batch_size
            )

        outputs = model.forward_training(
            image,
            boxes,
            box_counts,
            pair_targets=pair_targets,
            cfa_predicate_labels=cfa_predicate_labels,
            entity_labels=(
                object_label_indices
                if apache_objective is not None
                else None
            ),
            coverage=coverage,
            fill=fill,
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
        backward_loss = (
            losses["loss"] / float(grad_accum)
            if grad_accum > 1
            else losses["loss"]
        )
        backward_loss.backward()

        boundary = (
            (step + 1) % grad_accum == 0
            or (
                total_batches is not None
                and step + 1 == total_batches
            )
        )

        backbone_norm2 = 0.0
        head_norm2 = 0.0
        backbone_tensors = 0
        head_tensors = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError("training gradient became non-finite")
            if record_gradient_health:
                norm = float(
                    parameter.grad.detach().float().norm().item()
                )
                if name.startswith("backbone."):
                    backbone_norm2 += norm * norm
                    backbone_tensors += 1
                else:
                    head_norm2 += norm * norm
                    head_tensors += 1

        if record_gradient_health:
            gradient_sums["backbone_gradient_norm"] += (
                float(np.sqrt(backbone_norm2)) * batch_size
            )
            gradient_sums["head_gradient_norm"] += (
                float(np.sqrt(head_norm2)) * batch_size
            )
            gradient_sums["backbone_gradient_tensors"] += (
                float(backbone_tensors) * batch_size
            )
            gradient_sums["head_gradient_tensors"] += (
                float(head_tensors) * batch_size
            )

        if boundary:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_grad,
                )
            optimizer.step()
            optimizer_steps += 1
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                update = getattr(
                    ema,
                    "update",
                    None,
                )
                if not callable(update):
                    raise TypeError(
                        "ema must provide an update(model) method"
                    )
                update(model)
            optimizer.zero_grad(set_to_none=True)
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
        report = {
            key: value / examples
            for key, value in apache_sums.items()
        }
        if record_gradient_health:
            report.update(
                {
                    key: value / examples
                    for key, value in gradient_sums.items()
                }
            )
        report["micro_batches"] = float(micro_batches)
        report["optimizer_steps"] = float(optimizer_steps)
        report["grad_accum"] = float(grad_accum)
        if resolution_counts:
            report["training_resolution_min"] = float(
                min(resolution_counts)
            )
            report["training_resolution_max"] = float(
                max(resolution_counts)
            )
            report["training_resolution_distinct"] = float(
                len(resolution_counts)
            )
        if scheduler is not None:
            learning_rates = scheduler.get_last_lr()
            report["learning_rate_min"] = float(
                min(learning_rates)
            )
            report["learning_rate_max"] = float(
                max(learning_rates)
            )
        return report

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
    if record_gradient_health:
        report.update(
            {
                key: value / examples
                for key, value in gradient_sums.items()
            }
        )
    report["micro_batches"] = float(micro_batches)
    report["optimizer_steps"] = float(optimizer_steps)
    report["grad_accum"] = float(grad_accum)
    if resolution_counts:
        report["training_resolution_min"] = float(
            min(resolution_counts)
        )
        report["training_resolution_max"] = float(
            max(resolution_counts)
        )
        report["training_resolution_distinct"] = float(
            len(resolution_counts)
        )
    if scheduler is not None:
        learning_rates = scheduler.get_last_lr()
        report["learning_rate_min"] = float(
            min(learning_rates)
        )
        report["learning_rate_max"] = float(
            max(learning_rates)
        )
    return report


def evaluate_detector_boxes(
    model: KFRelationModel,
    manifest: DatasetManifest,
    detector_manifest: DetectorPredictionManifest,
    *,
    image_root: str | Path,
    device: torch.device,
    benchmark_config: DetectorRecoverabilityConfig = (
        DetectorRecoverabilityConfig()
    ),
    detector_id: str = "",
    detector_model_sha256: str = "",
    detector_config_sha256: str = "",
) -> dict[str, object]:
    benchmark = DetectorRecoverabilityBenchmark(
        predicate_count=int(model.predicate_bank.shape[0]),
        config=benchmark_config,
    )
    detector_by_image = detector_manifest.by_image()
    expected_images = {
        example.image
        for example in manifest.examples
    }
    actual_images = set(detector_by_image)
    missing = sorted(expected_images - actual_images)
    extra = sorted(actual_images - expected_images)
    if missing:
        raise ValueError(
            f"missing detector prediction for image: {missing[0]}"
        )
    if extra:
        raise ValueError(
            f"detector prediction has unknown image: {extra[0]}"
        )

    model.eval()
    with torch.inference_mode():
        for example in manifest.examples:
            raw_detector = detector_by_image[example.image]
            detector = raw_detector.limit(
                model.config.max_boxes
            )
            if (
                detector.width != example.width
                or detector.height != example.height
            ):
                raise ValueError(
                    f"detector dimensions for {example.image} "
                    "do not match GT manifest"
                )

            image = _load_square_rgb(
                example,
                Path(image_root),
                model.config.image_size,
            ).unsqueeze(0).to(device)
            boxes, box_count = prepare_detector_boxes(
                detector,
                max_boxes=model.config.max_boxes,
            )
            runtime = model(
                image,
                boxes.unsqueeze(0).to(device),
                box_count.unsqueeze(0).to(device),
            )
            benchmark.add(
                tuple(
                    value.detach().cpu()
                    for value in runtime
                ),
                example,
                detector,
                detector_boxes_before_cap=len(
                    raw_detector.boxes_xyxy
                ),
            )

    return benchmark.report(
        annotations_sha256=manifest.annotations_sha256,
        detector_predictions_sha256=(
            detector_manifest.predictions_sha256
        ),
        detector_id=detector_id,
        detector_model_sha256=detector_model_sha256,
        detector_config_sha256=detector_config_sha256,
    )


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
