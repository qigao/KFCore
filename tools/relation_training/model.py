from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RelationModelConfig:
    image_size: int = 448
    max_boxes: int = 32
    pair_budget: int = 128
    hidden_dim: int = 256
    geometry_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.0
    tap_indices: tuple[int, ...] = (-6, -3, -1)

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.max_boxes < 2:
            raise ValueError("max_boxes must be at least 2")
        if self.pair_budget <= 0 or self.pair_budget > self.max_boxes * self.max_boxes:
            raise ValueError("pair_budget must be within [1,max_boxes^2]")
        if self.hidden_dim <= 0 or self.geometry_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        if self.num_heads <= 0 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0,1)")
        if not self.tap_indices:
            raise ValueError("at least one backbone tap is required")


class BackboneAdapter(nn.Module):
    hidden_size: int
    patch_size: int

    def forward_taps(self, image: Tensor, taps: Sequence[int]) -> list[Tensor]:
        raise NotImplementedError


class HFDinoV3Backbone(BackboneAdapter):
    """Thin Transformers adapter. No DINOv3 source or weights are bundled here."""

    def __init__(
        self,
        model: nn.Module,
        *,
        hidden_size: int,
        patch_size: int,
        image_mean: Sequence[float],
        image_std: Sequence[float],
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or patch_size <= 0:
            raise ValueError("invalid DINOv3 backbone dimensions")
        if len(image_mean) != 3 or len(image_std) != 3:
            raise ValueError("DINOv3 image normalization must have three channels")
        if any(float(value) <= 0.0 for value in image_std):
            raise ValueError("DINOv3 image stddev must be positive")
        self.model = model
        self.hidden_size = int(hidden_size)
        self.patch_size = int(patch_size)
        self.register_buffer(
            "_mean",
            torch.tensor(image_mean, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "_std",
            torch.tensor(image_std, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=True,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        *,
        train_backbone: bool = True,
    ) -> "HFDinoV3Backbone":
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = int(model.config.hidden_size)
        patch_size_value = model.config.patch_size
        patch_size = (
            int(patch_size_value[0])
            if isinstance(patch_size_value, (tuple, list))
            else int(patch_size_value)
        )
        backbone = cls(
            model,
            hidden_size=hidden_size,
            patch_size=patch_size,
            image_mean=processor.image_mean,
            image_std=processor.image_std,
        )
        backbone.model.requires_grad_(train_backbone)
        return backbone

    def forward_taps(self, image: Tensor, taps: Sequence[int]) -> list[Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("DINOv3 image input must be NCHW with three channels")
        height = int(image.shape[2])
        width = int(image.shape[3])
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("image dimensions must be divisible by DINOv3 patch size")

        mean = self._mean.to(dtype=image.dtype, device=image.device)
        std = self._std.to(dtype=image.dtype, device=image.device)
        normalized = (image - mean) / std
        output = self.model(
            pixel_values=normalized,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("DINOv3 backbone did not return hidden states")

        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        patch_count = patch_height * patch_width
        result: list[Tensor] = []
        for tap in taps:
            hidden = hidden_states[tap]
            if hidden.ndim != 3 or hidden.shape[2] != self.hidden_size:
                raise RuntimeError("DINOv3 hidden-state contract mismatch")
            if hidden.shape[1] < patch_count:
                raise RuntimeError("DINOv3 hidden state has too few patch tokens")
            # DINOv3 may prepend CLS/storage tokens. Patch tokens remain the tail.
            patches = hidden[:, -patch_count:, :]
            result.append(
                patches.transpose(1, 2).reshape(
                    image.shape[0],
                    self.hidden_size,
                    patch_height,
                    patch_width,
                )
            )
        return result


class MetaDinoV3Backbone(BackboneAdapter):
    """Adapter for Meta's official DINOv3 torch.hub implementation.

    The caller supplies the official repository checkout and weight URL/path.
    KFCore does not vendor either the upstream source or model weights.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        hidden_size = getattr(model, "embed_dim", None)
        patch_size = getattr(model, "patch_size", None)
        depth = getattr(model, "n_blocks", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("official DINOv3 model is missing a valid embed_dim")
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("official DINOv3 model is missing a valid patch_size")
        if not isinstance(depth, int) or depth <= 0:
            raise ValueError("official DINOv3 model is missing a valid n_blocks")
        if not callable(getattr(model, "get_intermediate_layers", None)):
            raise ValueError("official DINOv3 model lacks get_intermediate_layers")

        self.model = model
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.depth = depth
        self.register_buffer(
            "_mean",
            torch.tensor(self.IMAGENET_MEAN, dtype=torch.float32).reshape(
                1, 3, 1, 1
            ),
            persistent=True,
        )
        self.register_buffer(
            "_std",
            torch.tensor(self.IMAGENET_STD, dtype=torch.float32).reshape(
                1, 3, 1, 1
            ),
            persistent=True,
        )

    @classmethod
    def from_official_repo(
        cls,
        repo_dir: str,
        *,
        model_name: str = "dinov3_vits16",
        weights: str,
        train_backbone: bool = True,
        check_hash: bool = True,
    ) -> "MetaDinoV3Backbone":
        if not repo_dir:
            raise ValueError("repo_dir must not be empty")
        if not weights:
            raise ValueError("weights URL/path must not be empty")

        root = Path(repo_dir).expanduser().resolve()
        if not (root / "dinov3" / "hub" / "backbones.py").is_file():
            raise ValueError(
                "repo_dir must point to an official DINOv3 source checkout"
            )

        # Import only the official backbone module, not hubconf.py. hubconf also
        # imports unrelated classifier/depth/text helpers with extra optional
        # dependencies. The official builder still downloads weights through
        # torch.hub.load_state_dict_from_url().
        root_text = str(root)
        sys.path.insert(0, root_text)
        try:
            module = importlib.import_module("dinov3.hub.backbones")
            builder = getattr(module, model_name, None)
            if not callable(builder):
                raise ValueError(
                    f"official DINOv3 repo has no backbone builder {model_name}"
                )
            model = builder(
                pretrained=True,
                weights=weights,
                check_hash=check_hash,
            )
        finally:
            if sys.path and sys.path[0] == root_text:
                sys.path.pop(0)

        backbone = cls(model)
        backbone.model.requires_grad_(train_backbone)
        return backbone

    def forward_taps(self, image: Tensor, taps: Sequence[int]) -> list[Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "DINOv3 image input must be NCHW with three channels"
            )
        height = int(image.shape[2])
        width = int(image.shape[3])
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "image dimensions must be divisible by DINOv3 patch size"
            )

        absolute: list[int] = []
        for tap in taps:
            index = self.depth + int(tap) if int(tap) < 0 else int(tap)
            if index < 0 or index >= self.depth:
                raise ValueError(
                    f"DINOv3 tap {tap} resolves outside [0,{self.depth})"
                )
            absolute.append(index)

        mean = self._mean.to(dtype=image.dtype, device=image.device)
        std = self._std.to(dtype=image.dtype, device=image.device)
        normalized = (image - mean) / std
        outputs = self.model.get_intermediate_layers(
            normalized,
            n=absolute,
            reshape=True,
            norm=False,
        )
        if len(outputs) != len(absolute):
            raise RuntimeError(
                "official DINOv3 returned the wrong number of intermediate taps"
            )

        result: list[Tensor] = []
        for feature in outputs:
            if (
                feature.ndim != 4
                or int(feature.shape[1]) != self.hidden_size
                or int(feature.shape[2]) != height // self.patch_size
                or int(feature.shape[3]) != width // self.patch_size
            ):
                raise RuntimeError(
                    "official DINOv3 intermediate feature contract mismatch"
                )
            result.append(feature)
        return result


class TimmDinoV3Backbone(BackboneAdapter):
    """Adapter for the public timm DINOv3 ViT implementation."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model: nn.Module,
        *,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        hidden_size = getattr(model, "num_features", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            hidden_size = getattr(model, "embed_dim", None)
        patch_embed = getattr(model, "patch_embed", None)
        patch_size_value = getattr(patch_embed, "patch_size", None)
        if isinstance(patch_size_value, (tuple, list)):
            if len(patch_size_value) != 2 or patch_size_value[0] != patch_size_value[1]:
                raise ValueError("timm DINOv3 requires square patch size")
            patch_size = int(patch_size_value[0])
        elif patch_size_value is not None:
            patch_size = int(patch_size_value)
        else:
            patch_size = 0
        blocks = getattr(model, "blocks", None)
        depth = len(blocks) if blocks is not None else 0

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("timm DINOv3 model is missing a valid hidden size")
        if patch_size <= 0:
            raise ValueError("timm DINOv3 model is missing a valid patch size")
        if depth <= 0:
            raise ValueError("timm DINOv3 model is missing transformer blocks")
        if not callable(getattr(model, "get_intermediate_layers", None)):
            raise ValueError("timm DINOv3 model lacks get_intermediate_layers")
        if len(image_mean) != 3 or len(image_std) != 3:
            raise ValueError("timm DINOv3 normalization must have three channels")

        self.model = model
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.depth = depth
        self.register_buffer(
            "_mean",
            torch.tensor(image_mean, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "_std",
            torch.tensor(image_std, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=True,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "hf_hub:timm/vit_small_patch16_dinov3.lvd1689m",
        *,
        train_backbone: bool = True,
    ) -> "TimmDinoV3Backbone":
        import timm

        model = timm.create_model(model_name, pretrained=True)
        cfg = getattr(model, "pretrained_cfg", {}) or {}
        image_mean = tuple(cfg.get("mean", cls.IMAGENET_MEAN))
        image_std = tuple(cfg.get("std", cls.IMAGENET_STD))
        backbone = cls(
            model,
            image_mean=image_mean,
            image_std=image_std,
        )
        backbone.model.requires_grad_(train_backbone)
        return backbone

    def forward_taps(self, image: Tensor, taps: Sequence[int]) -> list[Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "DINOv3 image input must be NCHW with three channels"
            )
        height = int(image.shape[2])
        width = int(image.shape[3])
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "image dimensions must be divisible by DINOv3 patch size"
            )

        absolute: list[int] = []
        for tap in taps:
            index = self.depth + int(tap) if int(tap) < 0 else int(tap)
            if index < 0 or index >= self.depth:
                raise ValueError(
                    f"DINOv3 tap {tap} resolves outside [0,{self.depth})"
                )
            absolute.append(index)

        mean = self._mean.to(dtype=image.dtype, device=image.device)
        std = self._std.to(dtype=image.dtype, device=image.device)
        normalized = (image - mean) / std
        outputs = self.model.get_intermediate_layers(
            normalized,
            n=absolute,
            reshape=True,
            return_prefix_tokens=False,
            norm=False,
        )
        if len(outputs) != len(absolute):
            raise RuntimeError(
                "timm DINOv3 returned the wrong number of intermediate taps"
            )
        result: list[Tensor] = []
        for feature in outputs:
            if (
                feature.ndim != 4
                or int(feature.shape[1]) != self.hidden_size
                or int(feature.shape[2]) != height // self.patch_size
                or int(feature.shape[3]) != width // self.patch_size
            ):
                raise RuntimeError(
                    "timm DINOv3 intermediate feature contract mismatch"
                )
            result.append(feature)
        return result


def _batch_gather(values: Tensor, indices: Tensor) -> Tensor:
    if values.ndim != 3 or indices.ndim != 2:
        raise ValueError("batch gather expects [B,N,C] values and [B,K] indices")
    expanded = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    return torch.gather(values, 1, expanded)


class KFRelationModel(nn.Module):
    """KFCore-owned relation-v1 model.

    Runtime contract:
      image       FP32 [B,3,S,S], RGB in [0,1]
      boxes       FP32 [B,N,4], normalized cx/cy/w/h
      box_counts  INT64 [B]

    Outputs:
      pred_logits [B,K,V]
      pair_logits [B,K]
      sub_idx     [B,K] int64
      obj_idx     [B,K] int64
      valid_mask  [B,K] bool
    """

    geometry_feature_count = 8

    def __init__(
        self,
        backbone: BackboneAdapter,
        predicate_embeddings: Tensor,
        config: RelationModelConfig = RelationModelConfig(),
    ) -> None:
        super().__init__()
        if not isinstance(backbone, BackboneAdapter):
            raise TypeError("backbone must implement BackboneAdapter")
        if predicate_embeddings.ndim != 2 or predicate_embeddings.shape[0] < 1:
            raise ValueError("predicate embeddings must be [V,D]")
        if predicate_embeddings.shape[1] < 1:
            raise ValueError("predicate embedding dimension must be positive")
        if config.image_size % backbone.patch_size != 0:
            raise ValueError("image_size must be divisible by backbone patch size")

        self.backbone = backbone
        self.config = config
        self.predicate_dim = int(predicate_embeddings.shape[1])
        normalized_bank = F.normalize(predicate_embeddings.float(), dim=-1)
        self.register_buffer("predicate_bank", normalized_bank, persistent=True)

        self.tap_norms = nn.ModuleList(
            nn.LayerNorm(backbone.hidden_size) for _ in config.tap_indices
        )
        self.tap_logits = nn.Parameter(torch.zeros(len(config.tap_indices)))

        self.geometry_encoder = nn.Sequential(
            nn.Linear(self.geometry_feature_count, config.geometry_dim),
            nn.GELU(),
            nn.Linear(config.geometry_dim, config.geometry_dim),
            nn.GELU(),
        )
        self.geometry_sampler = nn.Sequential(
            nn.Linear(self.geometry_feature_count, config.geometry_dim),
            nn.GELU(),
            nn.Linear(config.geometry_dim, 1),
        )

        pair_input_dim = backbone.hidden_size * 4 + config.geometry_dim
        self.pair_projection = nn.Sequential(
            nn.Linear(pair_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation_transformer = nn.TransformerEncoder(
            layer, num_layers=config.num_layers
        )
        self.pair_head = nn.Linear(config.hidden_dim, 1)
        self.predicate_projection = nn.Linear(
            config.hidden_dim, self.predicate_dim, bias=False
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.6592600369))

    def _fused_patch_features(self, image: Tensor) -> Tensor:
        taps = self.backbone.forward_taps(image, self.config.tap_indices)
        if len(taps) != len(self.tap_norms):
            raise RuntimeError("backbone returned the wrong number of taps")

        normalized: list[Tensor] = []
        spatial_shape: tuple[int, int] | None = None
        for feature, norm in zip(taps, self.tap_norms):
            if feature.ndim != 4 or feature.shape[1] != self.backbone.hidden_size:
                raise RuntimeError("backbone tap must be BCHW")
            current_shape = (int(feature.shape[2]), int(feature.shape[3]))
            if spatial_shape is None:
                spatial_shape = current_shape
            elif current_shape != spatial_shape:
                raise RuntimeError("all backbone taps must share one spatial grid")

            tokens = feature.flatten(2).transpose(1, 2)
            tokens = norm(tokens)
            normalized.append(tokens)

        weights = torch.softmax(self.tap_logits, dim=0)
        fused = normalized[0] * weights[0]
        for index in range(1, len(normalized)):
            fused = fused + normalized[index] * weights[index]

        assert spatial_shape is not None
        return fused.transpose(1, 2).reshape(
            image.shape[0],
            self.backbone.hidden_size,
            spatial_shape[0],
            spatial_shape[1],
        )

    def _pool_regions(
        self, features: Tensor, boxes: Tensor, valid_boxes: Tensor
    ) -> Tensor:
        batch, channels, height, width = features.shape
        dtype = features.dtype
        device = features.device

        grid_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        grid_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        grid = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

        cx, cy, bw, bh = boxes.unbind(dim=-1)
        half_w = (bw * 0.5).clamp_min(1.0e-4)
        half_h = (bh * 0.5).clamp_min(1.0e-4)
        dx = (grid[:, 0].view(1, 1, -1) - cx.unsqueeze(-1)).abs() / half_w.unsqueeze(-1)
        dy = (grid[:, 1].view(1, 1, -1) - cy.unsqueeze(-1)).abs() / half_h.unsqueeze(-1)

        sharpness = 8.0
        weights = torch.sigmoid((1.0 - dx) * sharpness)
        weights = weights * torch.sigmoid((1.0 - dy) * sharpness)
        weights = weights * valid_boxes.to(dtype=dtype).unsqueeze(-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

        flat_features = features.flatten(2).transpose(1, 2)
        return torch.bmm(weights, flat_features)

    def _pair_geometry(self, boxes: Tensor) -> Tensor:
        subject = boxes.unsqueeze(2)
        object_ = boxes.unsqueeze(1)
        scx, scy, sw, sh = subject.unbind(dim=-1)
        ocx, ocy, ow, oh = object_.unbind(dim=-1)

        dx = ocx - scx
        dy = ocy - scy
        distance = torch.sqrt(dx.square() + dy.square() + 1.0e-8)
        log_w_ratio = torch.log(ow.clamp_min(1.0e-4) / sw.clamp_min(1.0e-4))
        log_h_ratio = torch.log(oh.clamp_min(1.0e-4) / sh.clamp_min(1.0e-4))
        subject_area = sw * sh
        object_area = ow * oh

        s_left = scx - sw * 0.5
        s_top = scy - sh * 0.5
        s_right = scx + sw * 0.5
        s_bottom = scy + sh * 0.5
        o_left = ocx - ow * 0.5
        o_top = ocy - oh * 0.5
        o_right = ocx + ow * 0.5
        o_bottom = ocy + oh * 0.5

        inter_w = (torch.minimum(s_right, o_right) - torch.maximum(s_left, o_left)).clamp_min(0.0)
        inter_h = (torch.minimum(s_bottom, o_bottom) - torch.maximum(s_top, o_top)).clamp_min(0.0)
        intersection = inter_w * inter_h
        union = (subject_area + object_area - intersection).clamp_min(1.0e-6)
        iou = intersection / union
        subject_area_feature = subject_area.expand_as(intersection)
        object_area_feature = object_area.expand_as(intersection)

        return torch.stack(
            (
                dx,
                dy,
                distance,
                log_w_ratio,
                log_h_ratio,
                subject_area_feature,
                object_area_feature,
                iou,
            ),
            dim=-1,
        )

    def forward(
        self, image: Tensor, boxes: Tensor, box_counts: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must be [B,3,H,W]")
        if image.shape[2] != self.config.image_size or image.shape[3] != self.config.image_size:
            raise ValueError("image spatial size does not match RelationModelConfig")
        if boxes.ndim != 3 or boxes.shape[1:] != (self.config.max_boxes, 4):
            raise ValueError("boxes must be [B,max_boxes,4]")
        if box_counts.ndim != 1 or box_counts.shape[0] != image.shape[0]:
            raise ValueError("box_counts must be [B]")
        if boxes.shape[0] != image.shape[0]:
            raise ValueError("image and boxes batch sizes must match")

        batch = image.shape[0]
        box_index = torch.arange(
            self.config.max_boxes, device=boxes.device, dtype=torch.int64
        )
        valid_boxes = box_index.view(1, -1) < box_counts.to(torch.int64).view(-1, 1)
        subject_valid = valid_boxes.unsqueeze(2)
        object_valid = valid_boxes.unsqueeze(1)
        not_self = box_index.view(1, -1, 1) != box_index.view(1, 1, -1)
        valid_pairs = subject_valid & object_valid & not_self

        patch_features = self._fused_patch_features(image)
        region_features = self._pool_regions(patch_features, boxes, valid_boxes)
        geometry = self._pair_geometry(boxes)

        flat_geometry = geometry.reshape(
            batch, self.config.max_boxes * self.config.max_boxes, self.geometry_feature_count
        )
        flat_valid = valid_pairs.reshape(batch, -1)
        sampler_scores = self.geometry_sampler(flat_geometry).squeeze(-1)
        sampler_scores = sampler_scores.masked_fill(
            ~flat_valid, torch.finfo(sampler_scores.dtype).min
        )
        _, pair_slot = torch.topk(
            sampler_scores, k=self.config.pair_budget, dim=1, largest=True, sorted=True
        )

        all_subject = (
            box_index.view(-1, 1)
            .expand(self.config.max_boxes, self.config.max_boxes)
            .reshape(-1)
        )
        all_object = (
            box_index.view(1, -1)
            .expand(self.config.max_boxes, self.config.max_boxes)
            .reshape(-1)
        )
        subject_index = all_subject[pair_slot]
        object_index = all_object[pair_slot]
        selected_valid = torch.gather(flat_valid, 1, pair_slot)

        subject_features = _batch_gather(region_features, subject_index)
        object_features = _batch_gather(region_features, object_index)
        selected_geometry = torch.gather(
            flat_geometry,
            1,
            pair_slot.unsqueeze(-1).expand(
                -1, -1, self.geometry_feature_count
            ),
        )
        geometry_features = self.geometry_encoder(selected_geometry)

        pair_features = torch.cat(
            (
                subject_features,
                object_features,
                subject_features - object_features,
                subject_features * object_features,
                geometry_features,
            ),
            dim=-1,
        )
        tokens = self.pair_projection(pair_features)
        tokens = tokens * selected_valid.to(tokens.dtype).unsqueeze(-1)
        tokens = self.relation_transformer(tokens)
        tokens = tokens * selected_valid.to(tokens.dtype).unsqueeze(-1)

        pair_logits = self.pair_head(tokens).squeeze(-1)
        predicate_query = F.normalize(self.predicate_projection(tokens), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        pred_logits = scale * torch.matmul(
            predicate_query, self.predicate_bank.transpose(0, 1)
        )
        return (
            pred_logits,
            pair_logits,
            subject_index.to(torch.int64),
            object_index.to(torch.int64),
            selected_valid.to(torch.bool),
        )
