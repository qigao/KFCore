from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


CLIP_VOCAB_SIZE = 49_408


@dataclass(frozen=True)
class PredicateTextStudentConfig:
    vocab_size: int = CLIP_VOCAB_SIZE
    token_dim: int = 128
    model_dim: int = 256
    depth: int = 6
    heads: int = 4
    ffn_dim: int = 1024
    output_dim: int = 512
    max_length: int = 32

    def __post_init__(self) -> None:
        values = (
            self.vocab_size,
            self.token_dim,
            self.model_dim,
            self.depth,
            self.heads,
            self.ffn_dim,
            self.output_dim,
            self.max_length,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in values
        ):
            raise ValueError(
                "predicate text-student dimensions must be positive integers"
            )
        if self.model_dim % self.heads != 0:
            raise ValueError(
                "predicate text-student model_dim must be divisible by heads"
            )
        if self.token_dim > self.model_dim:
            raise ValueError(
                "predicate text-student token_dim must not exceed model_dim"
            )


class _PredicateTextBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        heads: int,
        ffn_dim: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim,
            heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, model_dim),
        )

    def forward(
        self,
        value: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        normalized = self.norm1(value)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        value = value + attended
        return value + self.ffn(self.norm2(value))


class PredicateTextStudent(nn.Module):
    """Apache-reference predicate text student.

    Contract:
      input_ids    INT64 [N,L]
      padding_mask BOOL  [N,L], True for padding
      output       FP32  [N,D], L2-normalized
    """

    PAD_ID = 0

    def __init__(
        self,
        config: PredicateTextStudentConfig = PredicateTextStudentConfig(),
    ) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.token_dim,
        )
        if config.token_dim < config.model_dim:
            self.token_projection: nn.Module = nn.Linear(
                config.token_dim,
                config.model_dim,
                bias=False,
            )
        else:
            self.token_projection = nn.Identity()

        self.positional = nn.Parameter(
            torch.zeros(
                config.max_length,
                config.model_dim,
            )
        )
        self.blocks = nn.ModuleList(
            _PredicateTextBlock(
                config.model_dim,
                config.heads,
                config.ffn_dim,
            )
            for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.output_projection = nn.Linear(
            config.model_dim,
            config.output_dim,
            bias=False,
        )

        nn.init.normal_(
            self.token_embedding.weight,
            std=0.02,
        )
        nn.init.normal_(
            self.positional,
            std=0.02,
        )

    def forward(
        self,
        input_ids: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [N,L]")
        if input_ids.dtype != torch.int64:
            raise ValueError("input_ids must be int64")
        if input_ids.shape[1] > self.config.max_length:
            raise ValueError(
                "input length exceeds predicate text-student max_length"
            )
        if padding_mask is None:
            padding_mask = input_ids == self.PAD_ID
        if (
            padding_mask.shape != input_ids.shape
            or padding_mask.dtype != torch.bool
        ):
            raise ValueError(
                "padding_mask must be bool with the same shape as input_ids"
            )

        length = input_ids.shape[1]
        value = self.token_projection(
            self.token_embedding(input_ids)
        )
        value = (
            value
            + self.positional[:length].unsqueeze(0)
        )
        for block in self.blocks:
            value = block(
                value,
                padding_mask,
            )
        value = self.final_norm(value)

        keep = (
            ~padding_mask
        ).to(value.dtype).unsqueeze(-1)
        pooled = (
            (value * keep).sum(dim=1)
            / keep.sum(dim=1).clamp_min(1.0)
        )
        return F.normalize(
            self.output_projection(pooled),
            dim=-1,
        )

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )


def config_from_checkpoint(
    payload: dict,
) -> PredicateTextStudentConfig:
    raw = payload.get("cfg")
    if not isinstance(raw, dict):
        raise ValueError(
            "predicate text-student checkpoint is missing cfg"
        )

    # The Apache checkpoint names the factorized token width d_tok.
    token_dim = raw.get("d_tok")
    if token_dim is None:
        token_dim = raw.get("dim")

    mapping = {
        "vocab_size": raw.get("vocab_size"),
        "token_dim": token_dim,
        "model_dim": raw.get("dim"),
        "depth": raw.get("depth"),
        "heads": raw.get("heads"),
        "ffn_dim": raw.get("ffn_dim"),
        "output_dim": raw.get("out_dim"),
        "max_length": raw.get("max_len"),
    }
    if any(value is None for value in mapping.values()):
        missing = [
            key
            for key, value in mapping.items()
            if value is None
        ]
        raise ValueError(
            "predicate text-student checkpoint config is missing "
            + ", ".join(missing)
        )
    return PredicateTextStudentConfig(
        **{
            key: int(value)
            for key, value in mapping.items()
        }
    )


def load_apache_checkpoint(
    path: str,
    *,
    device: str | torch.device = "cpu",
) -> PredicateTextStudent:
    payload = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError(
            "predicate text-student checkpoint must be a dictionary"
        )
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(
            "predicate text-student checkpoint is missing state_dict"
        )
    model = PredicateTextStudent(
        config_from_checkpoint(payload)
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model.eval()
