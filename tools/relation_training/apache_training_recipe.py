from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from model import KFRelationModel


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
        if self.backbone_mode not in {"frozen", "full"}:
            raise ValueError(
                "backbone_mode must be frozen/full"
            )


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
    configure_backbone_trainability(
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
