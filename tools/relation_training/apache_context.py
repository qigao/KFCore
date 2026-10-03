from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from apache_pair_evidence import ScenePosEnc


class CrossAttentionLayer(nn.Module):
    """Pre-norm self-attention, scene cross-attention and feed-forward."""

    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int,
        ffn_ratio: float = 2.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        ffn_dim = int(d_model * ffn_ratio)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
            bias=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
            bias=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        *,
        target_padding_mask: Tensor | None = None,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        normalized = self.norm1(target)
        self_attended, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=target_padding_mask,
            need_weights=False,
        )
        target = target + self_attended

        normalized = self.norm2(target)
        scene_attended, _ = self.cross_attn(
            normalized,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        target = target + scene_attended
        return target + self.ffn(self.norm3(target))


class ApacheRelationTransformer(nn.Module):
    """Apache-reference pair self-attention + scene/box-token cross-attention."""

    def __init__(
        self,
        *,
        d_model: int,
        scene_dim: int,
        n_self_layers: int = 2,
        n_cross_layers: int = 2,
        n_heads: int = 8,
        ffn_ratio: float = 2.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if n_self_layers < 0 or n_cross_layers <= 0:
            raise ValueError(
                "reference relation transformer requires >=0 self and >0 cross layers"
            )
        if d_model <= 0 or scene_dim <= 0:
            raise ValueError("context dimensions must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.scene_proj = nn.Linear(scene_dim, d_model)
        nn.init.xavier_uniform_(self.scene_proj.weight)
        nn.init.zeros_(self.scene_proj.bias)
        self.scene_pe = ScenePosEnc(d_model)

        ffn_dim = int(d_model * ffn_ratio)
        self.self_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_self_layers)
        )
        self.cross_layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(max(n_cross_layers - 1, 0))
        )
        self.last_cross = CrossAttentionLayer(
            d_model,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
        )

    def project_scene(self, scene_feature: Tensor) -> tuple[Tensor, int, int]:
        """BCHW scene -> projected [B,HW,D] with reference positional encoding."""
        if scene_feature.ndim != 4:
            raise ValueError("scene_feature must be BCHW")
        batch, channels, height, width = scene_feature.shape
        del channels
        scene = scene_feature.flatten(2).transpose(1, 2)
        scene = self.scene_proj(scene)
        scene = scene + self.scene_pe(
            height,
            width,
            device=scene.device,
            dtype=scene.dtype,
        )
        if scene.shape[0] != batch:
            raise RuntimeError("scene projection changed batch dimension")
        return scene, height, width

    def projected_scene_map(self, scene_feature: Tensor) -> Tensor:
        """BCHW scene -> [B,D,H,W] without positional encoding."""
        if scene_feature.ndim != 4:
            raise ValueError("scene_feature must be BCHW")
        projected = self.scene_proj(
            scene_feature.permute(0, 2, 3, 1)
        )
        return projected.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        pair_feature: Tensor,
        scene_feature: Tensor,
        *,
        box_tokens: Tensor | None = None,
        pair_padding_mask: Tensor | None = None,
        box_token_drop: Tensor | None = None,
    ) -> Tensor:
        if pair_feature.ndim != 3:
            raise ValueError("pair_feature must be [B,K,D]")
        scene, _, _ = self.project_scene(scene_feature)
        batch, scene_count, _ = scene.shape

        memory_padding_mask = None
        if box_tokens is not None:
            if (
                box_tokens.ndim != 4
                or box_tokens.shape[0] != batch
                or box_tokens.shape[1] != pair_feature.shape[1]
                or box_tokens.shape[3] != pair_feature.shape[2]
            ):
                raise ValueError(
                    "box_tokens must be [B,K,T,D] matching pair features"
                )
            pair_count = box_tokens.shape[1]
            token_count = box_tokens.shape[2]
            memory = torch.cat(
                (
                    scene,
                    box_tokens.reshape(
                        batch,
                        pair_count * token_count,
                        -1,
                    ),
                ),
                dim=1,
            )
            if (
                pair_padding_mask is not None
                or box_token_drop is not None
            ):
                scene_mask = torch.zeros(
                    batch,
                    scene_count,
                    dtype=torch.bool,
                    device=scene.device,
                )
                if pair_padding_mask is not None:
                    box_mask = pair_padding_mask.repeat_interleave(
                        token_count,
                        dim=1,
                    )
                else:
                    box_mask = torch.zeros(
                        batch,
                        pair_count * token_count,
                        dtype=torch.bool,
                        device=scene.device,
                    )
                if box_token_drop is not None:
                    if box_token_drop.shape != (batch,):
                        raise ValueError(
                            "box_token_drop must be [B]"
                        )
                    box_mask = box_mask | box_token_drop.view(-1, 1)
                memory_padding_mask = torch.cat(
                    (scene_mask, box_mask),
                    dim=1,
                )
        else:
            memory = scene

        result = pair_feature
        for layer in self.self_layers:
            result = layer(
                result,
                src_key_padding_mask=pair_padding_mask,
            )
        for layer in self.cross_layers:
            result = layer(
                tgt=result,
                memory=memory,
                tgt_key_padding_mask=pair_padding_mask,
                memory_key_padding_mask=memory_padding_mask,
            )
        return self.last_cross(
            result,
            memory,
            target_padding_mask=pair_padding_mask,
            memory_padding_mask=memory_padding_mask,
        )


class ApacheDeformableRelRead(nn.Module):
    """Apache-reference sparse scene read around four relation anchors."""

    ANCHOR_COUNT = 4

    def __init__(
        self,
        *,
        d_model: int,
        n_points: int = 4,
        n_heads: int = 8,
        null_slots: int = 2,
        null_logit_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by deformable heads")
        if n_points <= 0 or n_heads <= 0 or null_slots < 0:
            raise ValueError("invalid deformable read configuration")

        self.n_points = int(n_points)
        self.n_heads = int(n_heads)
        self.null_slots = int(null_slots)
        anchors = self.ANCHOR_COUNT
        points = self.n_points
        heads = self.n_heads
        nulls = self.null_slots
        head_dim = d_model // heads

        self.norm = nn.LayerNorm(d_model)
        self.offset_mlp = nn.Linear(
            d_model,
            heads * anchors * points * 2,
        )
        self.weight_mlp = nn.Linear(
            d_model,
            heads * anchors * (points + nulls),
        )
        if nulls:
            self.null_vectors = nn.Parameter(
                torch.zeros(
                    heads,
                    anchors,
                    nulls,
                    head_dim,
                )
            )
        else:
            self.register_parameter("null_vectors", None)

        self.out_proj = nn.Linear(d_model, d_model)
        self.gamma = nn.Parameter(torch.zeros(d_model))

        nn.init.zeros_(self.offset_mlp.weight)
        nn.init.zeros_(self.weight_mlp.weight)
        nn.init.zeros_(self.weight_mlp.bias)
        if nulls:
            with torch.no_grad():
                self.weight_mlp.bias.view(
                    heads,
                    anchors,
                    points + nulls,
                )[..., points:] = null_logit_bias

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        ring = torch.zeros(
            heads,
            anchors,
            points,
            2,
        )
        for head in range(heads):
            for point in range(points):
                theta = (
                    2.0
                    * math.pi
                    * (head + point / max(points, 1))
                    / heads
                )
                radius = 0.35 * (point + 1)
                ring[head, :, point, 0] = (
                    math.cos(theta) * radius
                )
                ring[head, :, point, 1] = (
                    math.sin(theta) * radius
                )
        with torch.no_grad():
            self.offset_mlp.bias.copy_(
                ring.reshape(-1)
            )

    def forward(
        self,
        queries: Tensor,
        scene: Tensor,
        anchors_cxcywh: Tensor,
    ) -> Tensor:
        if queries.ndim != 3:
            raise ValueError("queries must be [B,K,D]")
        if scene.ndim != 4:
            raise ValueError("scene must be [B,D,H,W]")
        if (
            anchors_cxcywh.ndim != 4
            or anchors_cxcywh.shape[:2] != queries.shape[:2]
            or anchors_cxcywh.shape[2:] != (
                self.ANCHOR_COUNT,
                4,
            )
        ):
            raise ValueError(
                "anchors must be [B,K,4,4]"
            )

        batch, pair_count, d_model = queries.shape
        heads = self.n_heads
        anchors = self.ANCHOR_COUNT
        points = self.n_points
        nulls = self.null_slots
        head_dim = d_model // heads

        normalized = self.norm(queries)
        offsets = self.offset_mlp(normalized).view(
            batch,
            pair_count,
            heads,
            anchors,
            1,
            points,
            2,
        )
        centers = anchors_cxcywh[..., :2].view(
            batch,
            pair_count,
            1,
            anchors,
            1,
            1,
            2,
        )
        half_extent = (
            anchors_cxcywh[..., 2:]
            .clamp_min(0.05)
            .mul(0.5)
            .view(
                batch,
                pair_count,
                1,
                anchors,
                1,
                1,
                2,
            )
        )
        positions = (
            centers + offsets * half_extent
        ).clamp(0.0, 1.0)

        height, width = scene.shape[-2:]
        scene_by_head = scene.view(
            batch,
            heads,
            head_dim,
            height,
            width,
        ).reshape(
            batch * heads,
            head_dim,
            height,
            width,
        )
        grid = (
            positions[..., 0:1, :, :]
            .squeeze(4)
            .permute(0, 2, 1, 3, 4, 5)
            .reshape(
                batch * heads,
                pair_count,
                anchors * points,
                2,
            )
            * 2.0
            - 1.0
        )
        values = F.grid_sample(
            scene_by_head,
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="zeros",
        ).view(
            batch,
            heads,
            head_dim,
            pair_count,
            anchors,
            points,
        )

        weights = F.softmax(
            self.weight_mlp(normalized).view(
                batch,
                pair_count,
                heads,
                anchors * (points + nulls),
            ),
            dim=-1,
        )

        if nulls:
            assert self.null_vectors is not None
            null_values = (
                self.null_vectors.view(
                    1,
                    heads,
                    anchors,
                    nulls,
                    head_dim,
                )
                .permute(0, 1, 4, 2, 3)
                .unsqueeze(3)
                .expand(
                    batch,
                    heads,
                    head_dim,
                    pair_count,
                    anchors,
                    nulls,
                )
            )
            values = torch.cat(
                (values, null_values),
                dim=-1,
            )

        values = values.reshape(
            batch * heads,
            head_dim,
            pair_count,
            anchors * (points + nulls),
        )
        weights_by_head = (
            weights.permute(0, 2, 1, 3)
            .reshape(
                batch * heads,
                pair_count,
                anchors * (points + nulls),
            )
        )
        read = torch.einsum(
            "bdkp,bkp->bkd",
            values,
            weights_by_head,
        )
        read = (
            read.view(
                batch,
                heads,
                pair_count,
                head_dim,
            )
            .permute(0, 2, 1, 3)
            .reshape(batch, pair_count, d_model)
        )
        return (
            queries
            + self.gamma
            * self.out_proj(read)
        )


class ApacheRelationInteractionBlock(nn.Module):
    """Apache-reference dependency + joint scene grounding refinement."""

    def __init__(
        self,
        *,
        d_model: int,
        scene_dim: int,
        n_dependency_layers: int = 2,
        n_grounding_layers: int = 1,
        n_heads: int = 8,
        ffn_ratio: float = 2.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        ffn_dim = int(d_model * ffn_ratio)

        if scene_dim != d_model:
            self.scene_proj: nn.Module = nn.Linear(
                scene_dim,
                d_model,
                bias=False,
            )
            nn.init.xavier_uniform_(
                self.scene_proj.weight
            )
        else:
            self.scene_proj = nn.Identity()
        self.scene_pe = ScenePosEnc(d_model)

        def layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

        self.dependency_layers = nn.ModuleList(
            layer() for _ in range(n_dependency_layers)
        )
        self.grounding_layers = nn.ModuleList(
            layer() for _ in range(n_grounding_layers)
        )

    def forward(
        self,
        queries: Tensor,
        scene_feature: Tensor,
        *,
        query_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if queries.ndim != 3 or scene_feature.ndim != 4:
            raise ValueError(
                "queries must be [B,K,D] and scene_feature BCHW"
            )

        result = queries
        scene = scene_feature.permute(0, 2, 3, 1)
        scene = self.scene_proj(scene)
        height, width = scene.shape[1:3]
        scene = scene.reshape(
            scene.shape[0],
            height * width,
            scene.shape[-1],
        )
        scene = scene + self.scene_pe(
            height,
            width,
            device=scene.device,
            dtype=scene.dtype,
        )

        for layer in self.dependency_layers:
            result = layer(
                result,
                src_key_padding_mask=query_padding_mask,
            )

        pair_count = result.shape[1]
        for layer in self.grounding_layers:
            joint = torch.cat(
                (result, scene),
                dim=1,
            )
            joint_mask = None
            if query_padding_mask is not None:
                scene_mask = torch.zeros(
                    result.shape[0],
                    scene.shape[1],
                    dtype=torch.bool,
                    device=result.device,
                )
                joint_mask = torch.cat(
                    (query_padding_mask, scene_mask),
                    dim=1,
                )
            result = layer(
                joint,
                src_key_padding_mask=joint_mask,
            )[:, :pair_count]
        return result
