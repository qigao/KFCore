from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def fourier_pe(
    coordinates: Tensor,
    num_freqs: int = 16,
    max_octave: float = 7.0,
) -> Tensor:
    """Apache-reference sinusoidal coordinate encoding."""
    if num_freqs <= 0:
        raise ValueError("num_freqs must be positive")
    frequencies = 2.0 ** torch.linspace(
        0.0,
        float(max_octave),
        num_freqs,
        device=coordinates.device,
        dtype=coordinates.dtype,
    )
    angles = coordinates.unsqueeze(-1) * frequencies * math.pi
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ScenePosEnc(nn.Module):
    """Zero-gated absolute Fourier encoding of the patch grid."""

    def __init__(
        self,
        d_model: int,
        *,
        num_freqs: int = 16,
        max_octave: float = 7.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.num_freqs = int(num_freqs)
        self.max_octave = float(max_octave)
        self.proj = nn.Linear(4 * self.num_freqs, d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gamma = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if height <= 0 or width <= 0:
            raise ValueError("scene grid dimensions must be positive")
        ys = (
            torch.arange(height, device=device, dtype=dtype) + 0.5
        ) / float(height)
        xs = (
            torch.arange(width, device=device, dtype=dtype) + 0.5
        ) / float(width)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        encoded = torch.cat(
            (
                fourier_pe(
                    xx.reshape(-1),
                    self.num_freqs,
                    self.max_octave,
                ),
                fourier_pe(
                    yy.reshape(-1),
                    self.num_freqs,
                    self.max_octave,
                ),
            ),
            dim=-1,
        )
        return (
            self.gamma.to(dtype=dtype)
            * self.proj(encoded)
        ).unsqueeze(0)


class BoxPromptEncoder(nn.Module):
    """Normalized xyxy box -> top-left and bottom-right prompt tokens."""

    def __init__(
        self,
        d_model: int,
        *,
        num_freqs: int = 16,
        max_octave: float = 7.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.num_freqs = int(num_freqs)
        self.max_octave = float(max_octave)
        self.proj = nn.Linear(4 * self.num_freqs, d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.corner_bias = nn.Embedding(2, d_model)
        nn.init.normal_(self.corner_bias.weight, std=0.02)

    def _point(self, x: Tensor, y: Tensor) -> Tensor:
        return self.proj(
            torch.cat(
                (
                    fourier_pe(x, self.num_freqs, self.max_octave),
                    fourier_pe(y, self.num_freqs, self.max_octave),
                ),
                dim=-1,
            )
        )

    def forward(self, boxes_xyxy: Tensor) -> Tensor:
        if boxes_xyxy.ndim < 2 or boxes_xyxy.shape[-1] != 4:
            raise ValueError("boxes_xyxy must end with four coordinates")
        x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
        top_left = self._point(x1, y1)
        bottom_right = self._point(x2, y2)
        device = boxes_xyxy.device
        top_left = top_left + self.corner_bias(
            torch.zeros(1, dtype=torch.long, device=device)
        )
        bottom_right = bottom_right + self.corner_bias(
            torch.ones(1, dtype=torch.long, device=device)
        )
        return torch.stack((top_left, bottom_right), dim=-2)

    def encode_pairs(
        self,
        subject_xyxy: Tensor,
        object_xyxy: Tensor,
    ) -> Tensor:
        return torch.cat(
            (self(subject_xyxy), self(object_xyxy)),
            dim=-2,
        )


class SoftSpatialPool(nn.Module):
    """Box-conditioned global cross-attention over the full patch grid."""

    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int = 8,
        num_freqs: int = 16,
        max_octave: float = 7.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.base_query = nn.Parameter(
            torch.empty(1, 1, self.d_model)
        )
        nn.init.normal_(self.base_query, std=0.02)
        self.scene_pe = ScenePosEnc(
            self.d_model,
            num_freqs=num_freqs,
            max_octave=max_octave,
        )
        self.box_pe = BoxPromptEncoder(
            self.d_model,
            num_freqs=num_freqs,
            max_octave=max_octave,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            batch_first=True,
            bias=True,
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.cov_lambda = nn.Parameter(
            torch.zeros(self.n_heads)
        )

    @staticmethod
    def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
        if boxes.ndim < 2 or boxes.shape[-1] != 4:
            raise ValueError("boxes must end with four coordinates")
        cx, cy, width, height = boxes.unbind(dim=-1)
        return torch.stack(
            (
                (cx - width * 0.5).clamp(0.0, 1.0),
                (cy - height * 0.5).clamp(0.0, 1.0),
                (cx + width * 0.5).clamp(0.0, 1.0),
                (cy + height * 0.5).clamp(0.0, 1.0),
            ),
            dim=-1,
        )

    def forward(
        self,
        feature_map: Tensor,
        boxes: Tensor,
        *,
        coverage: Tensor | None = None,
    ) -> Tensor:
        """feature_map [B,C,H,W], boxes [B,N,4], coverage [B,N,H*W]."""
        if feature_map.ndim != 4:
            raise ValueError("feature_map must be BCHW")
        if feature_map.shape[1] != self.d_model:
            raise ValueError("feature_map channel width does not match d_model")
        if boxes.ndim != 3 or boxes.shape[0] != feature_map.shape[0]:
            raise ValueError("boxes must be [B,N,4]")
        if boxes.shape[-1] != 4:
            raise ValueError("boxes must use normalized cxcywh")

        batch, _, height, width = feature_map.shape
        count = boxes.shape[1]
        patches = feature_map.flatten(2).transpose(1, 2)
        patches = patches + self.scene_pe(
            height,
            width,
            device=patches.device,
            dtype=patches.dtype,
        )

        corners = self.box_pe(
            self.cxcywh_to_xyxy(boxes)
        )
        query = self.base_query.expand(
            batch,
            count,
            self.d_model,
        ) + corners.mean(dim=-2)

        attention_mask = None
        if coverage is not None:
            if coverage.shape != (
                batch,
                count,
                height * width,
            ):
                raise ValueError(
                    "coverage must be [B,N,H*W] matching the feature grid"
                )
            log_coverage = torch.log(
                coverage.to(
                    device=patches.device,
                    dtype=patches.dtype,
                ).clamp(0.0, 1.0)
                + 1.0e-4
            )
            lam = self.cov_lambda.to(
                dtype=patches.dtype
            ).view(1, -1, 1, 1)
            attention_mask = (
                lam * log_coverage.unsqueeze(1)
            ).expand(
                batch,
                self.n_heads,
                count,
                height * width,
            )
            attention_mask = attention_mask.reshape(
                batch * self.n_heads,
                count,
                height * width,
            )

        attended, _ = self.cross_attn(
            query=query,
            key=patches,
            value=patches,
            attn_mask=attention_mask,
            need_weights=False,
        )
        return self.norm(attended)


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    return SoftSpatialPool.cxcywh_to_xyxy(boxes)


def union_box(subject: Tensor, object_: Tensor) -> Tensor:
    """Tight cxcywh box covering two normalized cxcywh boxes."""
    subject_xyxy = cxcywh_to_xyxy(subject)
    object_xyxy = cxcywh_to_xyxy(object_)
    x1 = torch.minimum(subject_xyxy[..., 0], object_xyxy[..., 0])
    y1 = torch.minimum(subject_xyxy[..., 1], object_xyxy[..., 1])
    x2 = torch.maximum(subject_xyxy[..., 2], object_xyxy[..., 2])
    y2 = torch.maximum(subject_xyxy[..., 3], object_xyxy[..., 3])
    return torch.stack(
        (
            (x1 + x2) * 0.5,
            (y1 + y2) * 0.5,
            x2 - x1,
            y2 - y1,
        ),
        dim=-1,
    )


def contact_box(subject: Tensor, object_: Tensor) -> Tensor:
    """Apache reference intersection/gap contact-zone box in cxcywh."""
    subject_xyxy = cxcywh_to_xyxy(subject)
    object_xyxy = cxcywh_to_xyxy(object_)
    inner_x1 = torch.maximum(
        subject_xyxy[..., 0],
        object_xyxy[..., 0],
    )
    inner_y1 = torch.maximum(
        subject_xyxy[..., 1],
        object_xyxy[..., 1],
    )
    inner_x2 = torch.minimum(
        subject_xyxy[..., 2],
        object_xyxy[..., 2],
    )
    inner_y2 = torch.minimum(
        subject_xyxy[..., 3],
        object_xyxy[..., 3],
    )
    return torch.stack(
        (
            (inner_x1 + inner_x2) * 0.5,
            (inner_y1 + inner_y2) * 0.5,
            (inner_x2 - inner_x1).abs(),
            (inner_y2 - inner_y1).abs(),
        ),
        dim=-1,
    )


def coverage_pair_metrics(
    coverage: Tensor,
) -> tuple[Tensor, Tensor]:
    """Coverage [B,N,C] -> pairwise region IoU/contact [B,N,N]."""
    if coverage.ndim != 3:
        raise ValueError("coverage must be [B,N,C]")
    coverage = coverage.float()
    area = coverage.sum(dim=-1)
    intersection = torch.einsum(
        "bnc,bmc->bnm",
        coverage,
        coverage,
    )
    subject_area = area.unsqueeze(2)
    object_area = area.unsqueeze(1)
    region_iou = intersection / (
        subject_area
        + object_area
        - intersection
    ).clamp_min(1.0e-6)
    region_contact = intersection / torch.minimum(
        subject_area,
        object_area,
    ).clamp_min(1.0e-6)
    return region_iou, region_contact


class RelGeomEncoder(nn.Module):
    """Apache-reference 19-D ordered-pair geometry encoder."""

    NUM_GEO = 19
    NUM_BOX_GEO = 15

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        hidden = max(1, d_model // 2)
        self.mlp = nn.Sequential(
            nn.Linear(self.NUM_GEO, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )
        with torch.no_grad():
            self.mlp[0].weight[
                :,
                self.NUM_BOX_GEO :,
            ].zero_()

    @staticmethod
    def features(
        subject: Tensor,
        object_: Tensor,
        region: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    ) -> Tensor:
        if subject.shape != object_.shape or subject.shape[-1] != 4:
            raise ValueError(
                "subject/object boxes must have matching [...,4] shape"
            )
        eps = 1.0e-6
        s_cx, s_cy, s_w, s_h = subject.unbind(dim=-1)
        o_cx, o_cy, o_w, o_h = object_.unbind(dim=-1)

        dx = (o_cx - s_cx) / (s_w + eps)
        dy = (o_cy - s_cy) / (s_h + eps)
        log_width_ratio = torch.log(
            (o_w + eps) / (s_w + eps)
        )
        log_height_ratio = torch.log(
            (o_h + eps) / (s_h + eps)
        )
        log_area_ratio = torch.log(
            (o_w * o_h + eps) / (s_w * s_h + eps)
        )
        log_subject_area = torch.log(
            s_w * s_h + eps
        )
        log_object_area = torch.log(
            o_w * o_h + eps
        )

        s_x1 = s_cx - s_w * 0.5
        s_y1 = s_cy - s_h * 0.5
        s_x2 = s_cx + s_w * 0.5
        s_y2 = s_cy + s_h * 0.5
        o_x1 = o_cx - o_w * 0.5
        o_y1 = o_cy - o_h * 0.5
        o_x2 = o_cx + o_w * 0.5
        o_y2 = o_cy + o_h * 0.5

        intersection_width = (
            torch.minimum(s_x2, o_x2)
            - torch.maximum(s_x1, o_x1)
        ).clamp_min(0.0)
        intersection_height = (
            torch.minimum(s_y2, o_y2)
            - torch.maximum(s_y1, o_y1)
        ).clamp_min(0.0)
        intersection = (
            intersection_width * intersection_height
        )

        subject_area = (s_w * s_h).clamp_min(eps)
        object_area = (o_w * o_h).clamp_min(eps)
        iou = intersection / (
            subject_area
            + object_area
            - intersection
            + eps
        )
        subject_inside = intersection / subject_area
        object_inside = intersection / object_area
        subject_aspect = torch.log(
            (s_w / (s_h + eps)).clamp_min(eps)
        )
        object_aspect = torch.log(
            (o_w / (o_h + eps)).clamp_min(eps)
        )

        center_dx = o_cx - s_cx
        center_dy = o_cy - s_cy
        distance = (
            center_dx.square()
            + center_dy.square()
        ).clamp_min(eps).sqrt()
        direction_cos = center_dx / distance
        direction_sin = center_dy / distance
        delta_center_y = center_dy

        if region is None:
            ones = torch.ones_like(iou)
            fill_subject = ones
            fill_object = ones
            region_iou = iou
            region_contact = intersection / torch.minimum(
                subject_area,
                object_area,
            ).clamp_min(eps)
        else:
            (
                fill_subject,
                fill_object,
                region_iou,
                region_contact,
            ) = region

        raw = torch.stack(
            (
                dx,
                dy,
                log_width_ratio,
                log_height_ratio,
                log_area_ratio,
                log_subject_area,
                log_object_area,
                iou,
                subject_inside,
                object_inside,
                subject_aspect,
                object_aspect,
                direction_cos,
                direction_sin,
                delta_center_y,
                fill_subject.expand_as(iou),
                fill_object.expand_as(iou),
                region_iou.expand_as(iou),
                region_contact.expand_as(iou),
            ),
            dim=-1,
        )
        return 10.0 * torch.tanh(raw / 10.0)

    def forward(
        self,
        subject: Tensor,
        object_: Tensor,
        region: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    ) -> Tensor:
        return self.mlp(
            self.features(subject, object_, region)
        )
