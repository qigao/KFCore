from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn

from model import KFRelationModel


LEGACY_DEFAULT_MAX_BOXES = 32
RELEASED_MAX_BOXES = 40
LEGACY_DEFAULT_HIDDEN_DIM = 256
RELEASED_D_MODEL = 512
RELEASED_TEXT_DIM = 512
RELEASED_EMA_DECAY = 0.9998
LEGACY_DEFAULT_AUGMENT = 0.0
RELEASED_PHOTOMETRIC_AUGMENT = 0.3


def resolve_training_hidden_dim(
    requested: int | None,
    *,
    recipe: str,
) -> int:
    if recipe not in {"legacy", "apache-reference"}:
        raise ValueError("training recipe must be legacy/apache-reference")
    value = (
        int(requested)
        if requested is not None
        else (
            RELEASED_D_MODEL
            if recipe == "apache-reference"
            else LEGACY_DEFAULT_HIDDEN_DIM
        )
    )
    if value <= 0:
        raise ValueError("hidden dimension must be positive")
    if (
        recipe == "apache-reference"
        and value != RELEASED_D_MODEL
    ):
        raise ValueError(
            "apache-reference requires released d_model=512"
        )
    return value


def resolve_training_max_boxes(
    requested: int | None,
    *,
    recipe: str,
) -> int:
    if recipe not in {"legacy", "apache-reference"}:
        raise ValueError("training recipe must be legacy/apache-reference")
    value = (
        int(requested)
        if requested is not None
        else (
            RELEASED_MAX_BOXES
            if recipe == "apache-reference"
            else LEGACY_DEFAULT_MAX_BOXES
        )
    )
    if value < 2:
        raise ValueError("max boxes must be at least 2")
    if (
        recipe == "apache-reference"
        and value != RELEASED_MAX_BOXES
    ):
        raise ValueError(
            "apache-reference requires released max_objects=40"
        )
    return value



def validate_training_text_dim(
    dimension: int,
    *,
    recipe: str,
) -> int:
    if recipe not in {"legacy", "apache-reference"}:
        raise ValueError("training recipe must be legacy/apache-reference")
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise ValueError(
            "predicate text dimension must be a positive integer"
        )
    if (
        recipe == "apache-reference"
        and dimension != RELEASED_TEXT_DIM
    ):
        raise ValueError(
            "apache-reference requires released text_dim=512"
        )
    return dimension


def resolve_training_augment(
    requested: float | None,
    *,
    recipe: str,
) -> float:
    if recipe not in {"legacy", "apache-reference"}:
        raise ValueError("training recipe must be legacy/apache-reference")
    value = (
        float(requested)
        if requested is not None
        else (
            RELEASED_PHOTOMETRIC_AUGMENT
            if recipe == "apache-reference"
            else LEGACY_DEFAULT_AUGMENT
        )
    )
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "photometric augmentation strength must be finite and non-negative"
        )
    if (
        recipe == "apache-reference"
        and not math.isclose(
            value,
            RELEASED_PHOTOMETRIC_AUGMENT,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "apache-reference requires released augment=0.3"
        )
    return value


def state_dict_sha256(
    state_dict: dict[str, torch.Tensor],
) -> str:
    """Deterministic SHA-256 over names/shapes/dtypes/tensor bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            ",".join(str(int(dim)) for dim in tensor.shape).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    return state_dict_sha256(
        dict(module.state_dict())
    )


class ModelEMA:
    """Exact released RelateAnything EMA update semantics."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = RELEASED_EMA_DECAY,
    ) -> None:
        if (
            not math.isfinite(decay)
            or decay < 0.0
            or decay >= 1.0
        ):
            raise ValueError(
                "EMA decay must be finite within [0,1)"
            )
        self.decay = float(decay)
        self.updates = 0
        self.ema_model = copy.deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    def effective_decay(self) -> float:
        return self.decay * (
            1.0
            - math.exp(
                -float(self.updates) / 2000.0
            )
        )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.effective_decay()
        source = (
            model.module
            if hasattr(model, "module")
            else model
        )
        for ema_parameter, source_parameter in zip(
            self.ema_model.parameters(),
            source.parameters(),
        ):
            ema_parameter.mul_(decay).add_(
                source_parameter.data,
                alpha=1.0 - decay,
            )
        for ema_buffer, source_buffer in zip(
            self.ema_model.buffers(),
            source.buffers(),
        ):
            ema_buffer.copy_(source_buffer)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.ema_model.state_dict()

    def report(self) -> dict[str, object]:
        return {
            "enabled": True,
            "decay": self.decay,
            "updates": self.updates,
            "effective_decay": self.effective_decay(),
            "state_sha256": module_state_sha256(
                self.ema_model
            ),
            "weights_source": "ema",
        }


def select_artifact_model(
    raw_model: KFRelationModel,
    ema: ModelEMA | None,
    *,
    recipe: str,
) -> tuple[KFRelationModel, str]:
    if recipe == "apache-reference":
        if ema is None:
            raise ValueError(
                "apache-reference artifact selection requires EMA"
            )
        return ema.ema_model, "ema"
    if recipe == "legacy":
        return raw_model, "raw"
    raise ValueError(
        "training recipe must be legacy/apache-reference"
    )


def tap_fusion_weights(model: KFRelationModel) -> list[float]:
    return [
        float(value)
        for value in torch.softmax(
            model.tap_logits.detach().float().cpu(),
            dim=0,
        ).tolist()
    ]


def backbone_provenance(
    model: KFRelationModel,
    *,
    model_name: str,
    mode: str,
) -> dict[str, object]:
    mean = getattr(model.backbone, "_mean", None)
    std = getattr(model.backbone, "_std", None)
    return {
        "model": model_name,
        "mode": mode,
        "state_sha256": module_state_sha256(model.backbone),
        "hidden_size": int(model.backbone.hidden_size),
        "patch_size": int(model.backbone.patch_size),
        "depth": int(getattr(model.backbone, "depth", 0)),
        "tap_indices": list(model.config.tap_indices),
        "tap_weights": tap_fusion_weights(model),
        "image_mean": (
            [float(v) for v in mean.reshape(-1).tolist()]
            if isinstance(mean, torch.Tensor)
            else None
        ),
        "image_std": (
            [float(v) for v in std.reshape(-1).tolist()]
            if isinstance(std, torch.Tensor)
            else None
        ),
    }


def resolve_training_epochs(
    requested: int | None,
    *,
    recipe: str,
) -> int:
    if recipe not in {"legacy", "apache-reference"}:
        raise ValueError("training recipe must be legacy/apache-reference")
    value = (
        int(requested)
        if requested is not None
        else (12 if recipe == "apache-reference" else 5)
    )
    if value <= 0:
        raise ValueError("training epochs must be positive")
    return value


@dataclass(frozen=True)
class ApacheTrainingRecipeConfig:
    head_lr: float = 4.0e-4
    backbone_lr: float = 5.0e-5
    weight_decay: float = 1.0e-4
    epochs: int = 12
    warmup_steps: int = 500
    min_lr_factor: float = 0.01
    clip_grad: float = 1.0
    backbone_mode: str = "full"
    micro_batch_size: int = 32
    grad_accum: int = 4
    multi_scale: str = "0.5,1.5"
    multi_scale_n: int = 7
    cfa_prob: float = 0.5
    cfa_alpha: float = 1.0
    ema_decay: float = RELEASED_EMA_DECAY
    augment: float = RELEASED_PHOTOMETRIC_AUGMENT
    text_dim: int = RELEASED_TEXT_DIM

    def __post_init__(self) -> None:
        for name in (
            "head_lr",
            "backbone_lr",
            "weight_decay",
            "min_lr_factor",
            "clip_grad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative"
                )
        if self.head_lr <= 0.0 or self.backbone_lr <= 0.0:
            raise ValueError(
                "head/backbone learning rates must be positive"
            )
        if self.min_lr_factor <= 0.0 or self.min_lr_factor > 1.0:
            raise ValueError(
                "min_lr_factor must be within (0,1]"
            )
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, int)
            or self.epochs <= 0
        ):
            raise ValueError("epochs must be a positive integer")
        if (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or self.warmup_steps < 0
        ):
            raise ValueError(
                "warmup_steps must be a non-negative integer"
            )
        for name in (
            "micro_batch_size",
            "grad_accum",
            "multi_scale_n",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer"
                )
        if (
            not math.isfinite(self.cfa_prob)
            or self.cfa_prob < 0.0
            or self.cfa_prob > 1.0
        ):
            raise ValueError(
                "cfa_prob must be within [0,1]"
            )
        if (
            not math.isfinite(self.cfa_alpha)
            or self.cfa_alpha <= 0.0
        ):
            raise ValueError(
                "cfa_alpha must be finite and positive"
            )
        if (
            not math.isfinite(self.ema_decay)
            or self.ema_decay < 0.0
            or self.ema_decay >= 1.0
        ):
            raise ValueError(
                "ema_decay must be finite within [0,1)"
            )
        if (
            isinstance(self.text_dim, bool)
            or not isinstance(self.text_dim, int)
            or self.text_dim <= 0
        ):
            raise ValueError(
                "text_dim must be a positive integer"
            )
        if self.text_dim != RELEASED_TEXT_DIM:
            raise ValueError(
                "Apache released recipe requires text_dim=512"
            )
        if (
            not math.isfinite(self.augment)
            or self.augment < 0.0
        ):
            raise ValueError(
                "augment must be finite and non-negative"
            )
        if self.multi_scale:
            parts = self.multi_scale.split(",")
            if len(parts) != 2:
                raise ValueError(
                    "multi_scale must be 'lo,hi' or empty"
                )
            lo, hi = (float(value) for value in parts)
            if not (0.0 < lo <= hi):
                raise ValueError(
                    "multi_scale range must satisfy 0 < lo <= hi"
                )
        if self.backbone_mode not in {"frozen", "full"}:
            raise ValueError(
                "backbone_mode must be frozen/full"
            )

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum


def _freeze_unused_backbone_outputs(
    model: KFRelationModel,
) -> tuple[str, ...]:
    """Freeze final norm/mask-token parameters unused by intermediate taps."""
    frozen: list[str] = []
    backbone_model = getattr(
        model.backbone,
        "model",
        None,
    )
    if backbone_model is None:
        return ()

    for attr in ("norm", "layernorm", "fc_norm"):
        module = getattr(
            backbone_model,
            attr,
            None,
        )
        if isinstance(module, nn.Module):
            module.requires_grad_(False)
            frozen.append(
                f"backbone.model.{attr}"
            )

    embeddings = getattr(
        backbone_model,
        "embeddings",
        None,
    )
    mask_token = (
        getattr(embeddings, "mask_token", None)
        if embeddings is not None
        else None
    )
    if isinstance(mask_token, nn.Parameter):
        mask_token.requires_grad_(False)
        frozen.append(
            "backbone.model.embeddings.mask_token"
        )

    direct_mask_token = getattr(
        backbone_model,
        "mask_token",
        None,
    )
    if isinstance(direct_mask_token, nn.Parameter):
        direct_mask_token.requires_grad_(False)
        frozen.append(
            "backbone.model.mask_token"
        )

    return tuple(sorted(set(frozen)))


def configure_backbone_trainability(
    model: KFRelationModel,
    *,
    mode: str,
) -> dict[str, object]:
    if mode not in {"frozen", "full"}:
        raise ValueError(
            "backbone trainability mode must be frozen/full"
        )

    if mode == "frozen":
        model.backbone.requires_grad_(False)
        frozen_unused: tuple[str, ...] = ()
    else:
        model.backbone.requires_grad_(True)
        frozen_unused = _freeze_unused_backbone_outputs(
            model
        )

    total = 0
    trainable = 0
    for parameter in model.backbone.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()

    if total <= 0:
        raise ValueError(
            "relation backbone exposes no parameters"
        )
    if mode == "full" and trainable <= 0:
        raise ValueError(
            "full backbone mode left no trainable backbone parameters"
        )
    if mode == "frozen" and trainable != 0:
        raise RuntimeError(
            "frozen backbone mode left trainable parameters"
        )

    return {
        "mode": mode,
        "parameter_count": total,
        "trainable_parameter_count": trainable,
        "frozen_unused_modules": list(
            frozen_unused
        ),
    }


def build_reference_optimizer(
    model: KFRelationModel,
    config: ApacheTrainingRecipeConfig,
) -> tuple[torch.optim.AdamW, dict[str, object]]:
    backbone_report = configure_backbone_trainability(
        model,
        mode=config.backbone_mode,
    )

    backbone_decay: list[nn.Parameter] = []
    backbone_no_decay: list[nn.Parameter] = []
    head_decay: list[nn.Parameter] = []
    head_no_decay: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        no_decay = parameter.ndim <= 1
        if is_backbone:
            (
                backbone_no_decay
                if no_decay
                else backbone_decay
            ).append(parameter)
        else:
            (
                head_no_decay
                if no_decay
                else head_decay
            ).append(parameter)

    groups: list[dict[str, object]] = []

    def emit(
        parameters: list[nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        name: str,
    ) -> None:
        if parameters:
            groups.append(
                {
                    "params": parameters,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "group_name": name,
                }
            )

    emit(
        head_decay,
        lr=config.head_lr,
        weight_decay=config.weight_decay,
        name="head_decay",
    )
    emit(
        head_no_decay,
        lr=config.head_lr,
        weight_decay=0.0,
        name="head_no_decay",
    )
    emit(
        backbone_decay,
        lr=config.backbone_lr,
        weight_decay=config.weight_decay,
        name="backbone_decay",
    )
    emit(
        backbone_no_decay,
        lr=config.backbone_lr,
        weight_decay=0.0,
        name="backbone_no_decay",
    )

    if not groups:
        raise ValueError(
            "Apache reference optimizer has no trainable parameters"
        )

    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=config.weight_decay,
        fused=torch.cuda.is_available(),
    )
    report = {
        "head_lr": config.head_lr,
        "backbone": backbone_report,
        "backbone_lr": config.backbone_lr,
        "weight_decay": config.weight_decay,
        "groups": [
            {
                "name": str(group["group_name"]),
                "lr": float(group["lr"]),
                "weight_decay": float(
                    group["weight_decay"]
                ),
                "parameter_tensors": len(
                    group["params"]
                ),
                "parameter_count": sum(
                    parameter.numel()
                    for parameter in group["params"]
                ),
            }
            for group in groups
        ],
    }
    return optimizer, report


def build_reference_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ApacheTrainingRecipeConfig,
    *,
    steps_per_epoch: int,
) -> tuple[
    torch.optim.lr_scheduler.LambdaLR,
    dict[str, object],
]:
    if steps_per_epoch <= 0:
        raise ValueError(
            "steps_per_epoch must be positive"
        )
    total_steps = max(
        config.epochs * steps_per_epoch,
        1,
    )
    warmup_steps = min(
        config.warmup_steps,
        total_steps // 10 + 1,
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(
                1.0e-3,
                step / max(warmup_steps, 1),
            )
        progress = (
            step - warmup_steps
        ) / max(
            total_steps - warmup_steps,
            1,
        )
        cosine = 0.5 * (
            1.0
            + math.cos(
                math.pi
                * min(progress, 1.0)
            )
        )
        return max(
            config.min_lr_factor,
            cosine,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )
    return scheduler, {
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "min_lr_factor": config.min_lr_factor,
    }


def gradient_health(
    model: KFRelationModel,
) -> dict[str, float | int]:
    backbone_norm2 = 0.0
    head_norm2 = 0.0
    backbone_tensors = 0
    head_tensors = 0

    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if (
            not parameter.requires_grad
            or gradient is None
        ):
            continue
        if not torch.isfinite(gradient).all():
            raise RuntimeError(
                f"non-finite gradient: {name}"
            )
        value = float(
            gradient.detach().float().norm().item()
        )
        if name.startswith("backbone."):
            backbone_norm2 += value * value
            backbone_tensors += 1
        else:
            head_norm2 += value * value
            head_tensors += 1

    return {
        "backbone_gradient_norm": math.sqrt(
            backbone_norm2
        ),
        "head_gradient_norm": math.sqrt(
            head_norm2
        ),
        "backbone_gradient_tensors": backbone_tensors,
        "head_gradient_tensors": head_tensors,
    }
