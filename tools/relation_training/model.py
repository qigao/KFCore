from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from apache_vocab_head import ApacheVocabHead
from apache_context import (
    ApacheDeformableRelRead,
    ApacheRelationInteractionBlock,
    ApacheRelationTransformer,
)
from apache_pair_sampler import ApacheRelatednessPairSampler
from apache_pair_evidence import (
    BoxPromptEncoder as ApacheBoxPromptEncoder,
    RelGeomEncoder as ApacheRelGeomEncoder,
    SoftSpatialPool as ApacheSoftSpatialPool,
    contact_box as apache_contact_box,
    coverage_pair_metrics as apache_coverage_pair_metrics,
    cxcywh_to_xyxy as apache_cxcywh_to_xyxy,
    union_box as apache_union_box,
)


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
    predicate_adapter_rank: int = 0
    pair_visual_evidence: str = "endpoint"
    pair_geometry_evidence: str = "basic"
    pair_evidence_contract: str = "legacy"
    pair_sampler_contract: str = "legacy"
    relation_context_contract: str = "legacy"
    apache_context_dropout: float = 0.2
    apache_box_token_dropout: float = 0.3
    apache_pair_negative_floor: float = 0.3
    apache_cfa_prob: float = 0.0
    apache_cfa_alpha: float = 1.0
    allow_training_multiscale: bool = False
    predicate_head_contract: str = "legacy"

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
        if (
            isinstance(self.predicate_adapter_rank, bool)
            or not isinstance(self.predicate_adapter_rank, int)
            or self.predicate_adapter_rank < 0
        ):
            raise ValueError(
                "predicate_adapter_rank must be a non-negative integer"
            )
        if self.pair_visual_evidence not in {
            "endpoint",
            "union",
            "contact",
            "union-contact",
        }:
            raise ValueError(
                "pair_visual_evidence must be "
                "endpoint/union/contact/union-contact"
            )
        if self.pair_geometry_evidence not in {
            "basic",
            "rich",
        }:
            raise ValueError(
                "pair_geometry_evidence must be basic/rich"
            )
        if self.pair_evidence_contract not in {
            "legacy",
            "apache",
        }:
            raise ValueError(
                "pair_evidence_contract must be legacy/apache"
            )
        if (
            self.pair_evidence_contract == "apache"
            and (
                self.pair_visual_evidence != "endpoint"
                or self.pair_geometry_evidence != "basic"
            )
        ):
            raise ValueError(
                "apache pair evidence supersedes legacy visual/geometry knobs"
            )
        if self.pair_sampler_contract not in {
            "legacy",
            "apache",
        }:
            raise ValueError(
                "pair_sampler_contract must be legacy/apache"
            )
        if (
            self.pair_sampler_contract == "apache"
            and self.pair_evidence_contract != "apache"
        ):
            raise ValueError(
                "apache pair sampler requires apache pair evidence"
            )
        if (
            self.pair_sampler_contract == "apache"
            and self.pair_budget > 400
        ):
            raise ValueError(
                "apache pair sampler final budget must not exceed 400"
            )
        if self.relation_context_contract not in {
            "legacy",
            "apache",
        }:
            raise ValueError(
                "relation_context_contract must be legacy/apache"
            )
        if (
            self.relation_context_contract == "apache"
            and (
                self.pair_evidence_contract != "apache"
                or self.pair_sampler_contract != "apache"
            )
        ):
            raise ValueError(
                "apache relation context requires apache pair evidence "
                "and apache pair sampler"
            )
        if not 0.0 <= self.apache_context_dropout < 1.0:
            raise ValueError(
                "apache_context_dropout must be within [0,1)"
            )
        if not 0.0 <= self.apache_box_token_dropout <= 1.0:
            raise ValueError(
                "apache_box_token_dropout must be within [0,1]"
            )
        if (
            not torch.isfinite(
                torch.tensor(self.apache_pair_negative_floor)
            )
            or self.apache_pair_negative_floor < 0.0
            or self.apache_pair_negative_floor > 1.0
        ):
            raise ValueError(
                "apache_pair_negative_floor must be finite within [0,1]"
            )
        if (
            not torch.isfinite(
                torch.tensor(self.apache_cfa_prob)
            )
            or self.apache_cfa_prob < 0.0
            or self.apache_cfa_prob > 1.0
        ):
            raise ValueError(
                "apache_cfa_prob must be finite within [0,1]"
            )
        if (
            not torch.isfinite(
                torch.tensor(self.apache_cfa_alpha)
            )
            or self.apache_cfa_alpha <= 0.0
        ):
            raise ValueError(
                "apache_cfa_alpha must be finite and positive"
            )
        if (
            self.apache_cfa_prob > 0.0
            and self.pair_evidence_contract != "apache"
        ):
            raise ValueError(
                "Apache CFA requires Apache pair evidence"
            )
        if self.predicate_head_contract not in {
            "legacy",
            "apache",
        }:
            raise ValueError(
                "predicate_head_contract must be legacy/apache"
            )
        if (
            self.predicate_head_contract == "apache"
            and self.relation_context_contract != "apache"
        ):
            raise ValueError(
                "apache predicate head requires apache relation context"
            )
        if (
            self.predicate_head_contract == "apache"
            and self.predicate_adapter_rank != 0
        ):
            raise ValueError(
                "apache predicate head keeps W fixed and does not allow "
                "predicate_adapter_rank"
            )


@dataclass(frozen=True)
class ApachePairEvidenceOutputs:
    pair_tokens: Tensor
    box_tokens: Tensor
    anchors: Tensor
    geometry_features: Tensor
    subject_features: Tensor
    object_features: Tensor


@dataclass(frozen=True)
class RelationTrainingOutputs:
    runtime: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    sampler_logits: Tensor
    sampler_valid: Tensor
    predicate_query: Tensor
    predicate_query_raw: Tensor
    predicate_bank: Tensor
    predicate_spatial_query: Tensor | None = None
    predicate_alpha: Tensor | None = None
    object_subject_query: Tensor | None = None
    object_object_query: Tensor | None = None
    sampler_geo_loss: Tensor | None = None
    sampler_relatedness_loss: Tensor | None = None
    sampler_pair_negative_weights: Tensor | None = None


class BackboneAdapter(nn.Module):
    hidden_size: int
    patch_size: int

    def forward_taps(self, image: Tensor, taps: Sequence[int]) -> list[Tensor]:
        raise NotImplementedError


class OfficialDinoV3Backbone(BackboneAdapter):
    """Adapter for Meta's official DINOv3 PyTorch implementation."""

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if not hasattr(model, "get_intermediate_layers"):
            raise TypeError("official DINOv3 model must expose get_intermediate_layers")
        hidden_size = int(getattr(model, "embed_dim", 0))
        patch_size_value = getattr(model, "patch_size", 0)
        patch_size = (
            int(patch_size_value[0])
            if isinstance(patch_size_value, (tuple, list))
            else int(patch_size_value)
        )
        depth = int(getattr(model, "n_blocks", 0))
        if hidden_size <= 0 or patch_size <= 0 or depth <= 0:
            raise ValueError("invalid official DINOv3 model dimensions")
        self.model = model
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.depth = depth
        self.register_buffer(
            "_mean",
            torch.tensor(self._IMAGENET_MEAN, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "_std",
            torch.tensor(self._IMAGENET_STD, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=True,
        )

    @classmethod
    def from_torch_hub(
        cls,
        repo_dir: str,
        *,
        weights: str,
        model_name: str = "dinov3_vits16",
        train_backbone: bool = True,
    ) -> "OfficialDinoV3Backbone":
        model = torch.hub.load(
            repo_dir,
            model_name,
            source="local",
            weights=weights,
            check_hash=True,
        )
        model.requires_grad_(train_backbone)
        return cls(model)

    def forward_taps(self, image: Tensor, taps: Sequence[int]) -> list[Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("DINOv3 image input must be NCHW with three channels")
        height = int(image.shape[2])
        width = int(image.shape[3])
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("image dimensions must be divisible by DINOv3 patch size")

        indices: list[int] = []
        for tap in taps:
            index = self.depth + int(tap) if int(tap) < 0 else int(tap)
            if index < 0 or index >= self.depth:
                raise ValueError("DINOv3 tap index is outside backbone depth")
            indices.append(index)

        normalized = (
            image - self._mean.to(dtype=image.dtype, device=image.device)
        ) / self._std.to(dtype=image.dtype, device=image.device)
        outputs = self.model.get_intermediate_layers(
            normalized,
            n=indices,
            reshape=True,
            norm=False,
        )
        if len(outputs) != len(indices):
            raise RuntimeError("official DINOv3 returned the wrong number of taps")
        result: list[Tensor] = []
        for feature in outputs:
            if feature.ndim != 4 or feature.shape[1] != self.hidden_size:
                raise RuntimeError("official DINOv3 tap must be BCHW")
            result.append(feature)
        return result


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
        if not (
            callable(getattr(model, "forward_intermediates", None))
            or callable(getattr(model, "get_intermediate_layers", None))
        ):
            raise ValueError(
                "timm DINOv3 model lacks an intermediate-feature API"
            )
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
        forward_intermediates = getattr(
            self.model, "forward_intermediates", None
        )
        if callable(forward_intermediates):
            outputs = forward_intermediates(
                normalized,
                indices=absolute,
                return_prefix_tokens=False,
                norm=False,
                output_fmt="NCHW",
                intermediates_only=True,
            )
        else:
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


def _pair_union_contact_boxes(
    subject_boxes: Tensor,
    object_boxes: Tensor,
    valid_pairs: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if (
        subject_boxes.ndim != 3
        or subject_boxes.shape != object_boxes.shape
        or subject_boxes.shape[-1] != 4
    ):
        raise ValueError(
            "subject/object pair boxes must both be [B,K,4]"
        )
    if (
        valid_pairs.ndim != 2
        or valid_pairs.shape != subject_boxes.shape[:2]
    ):
        raise ValueError("valid_pairs must be [B,K]")

    scx, scy, sw, sh = subject_boxes.unbind(dim=-1)
    ocx, ocy, ow, oh = object_boxes.unbind(dim=-1)

    s_left = scx - sw * 0.5
    s_top = scy - sh * 0.5
    s_right = scx + sw * 0.5
    s_bottom = scy + sh * 0.5
    o_left = ocx - ow * 0.5
    o_top = ocy - oh * 0.5
    o_right = ocx + ow * 0.5
    o_bottom = ocy + oh * 0.5

    union_left = torch.minimum(s_left, o_left)
    union_top = torch.minimum(s_top, o_top)
    union_right = torch.maximum(s_right, o_right)
    union_bottom = torch.maximum(s_bottom, o_bottom)
    union_width = (union_right - union_left).clamp_min(0.0)
    union_height = (union_bottom - union_top).clamp_min(0.0)
    union_boxes = torch.stack(
        (
            (union_left + union_right) * 0.5,
            (union_top + union_bottom) * 0.5,
            union_width,
            union_height,
        ),
        dim=-1,
    )
    union_boxes = torch.where(
        valid_pairs.unsqueeze(-1),
        union_boxes,
        torch.zeros_like(union_boxes),
    )

    contact_left = torch.maximum(s_left, o_left)
    contact_top = torch.maximum(s_top, o_top)
    contact_right = torch.minimum(s_right, o_right)
    contact_bottom = torch.minimum(s_bottom, o_bottom)
    contact_width = (contact_right - contact_left).clamp_min(0.0)
    contact_height = (contact_bottom - contact_top).clamp_min(0.0)
    contact_valid = (
        valid_pairs
        & (contact_width > 0.0)
        & (contact_height > 0.0)
    )
    contact_boxes = torch.stack(
        (
            (contact_left + contact_right) * 0.5,
            (contact_top + contact_bottom) * 0.5,
            contact_width,
            contact_height,
        ),
        dim=-1,
    )
    contact_boxes = torch.where(
        contact_valid.unsqueeze(-1),
        contact_boxes,
        torch.zeros_like(contact_boxes),
    )
    return union_boxes, contact_boxes, contact_valid


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
    rich_geometry_feature_count = 10

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

        # Optional modules are initialized only after every common stochastic
        # module. This keeps rank=0 and rank>0 common parameter initialization
        # bitwise-identical under the same global RNG seed.
        if config.predicate_adapter_rank > self.predicate_dim:
            raise ValueError(
                "predicate_adapter_rank must not exceed predicate embedding dimension"
            )
        if config.predicate_adapter_rank > 0:
            self.predicate_adapter_down: nn.Linear | None = nn.Linear(
                self.predicate_dim,
                config.predicate_adapter_rank,
                bias=False,
            )
            self.predicate_adapter_up: nn.Linear | None = nn.Linear(
                config.predicate_adapter_rank,
                self.predicate_dim,
                bias=False,
            )
            nn.init.normal_(
                self.predicate_adapter_down.weight,
                mean=0.0,
                std=0.02,
            )
            nn.init.zeros_(self.predicate_adapter_up.weight)
        else:
            self.predicate_adapter_down = None
            self.predicate_adapter_up = None

        # Visual-evidence residuals are initialized after every common
        # stochastic module so endpoint/union/union-contact share identical
        # common parameters under one seed. Zero initialization makes all
        # modes start from the exact endpoint behavior.
        if config.pair_visual_evidence in {"union", "union-contact"}:
            self.union_projection: nn.Linear | None = nn.Linear(
                backbone.hidden_size,
                config.hidden_dim,
                bias=False,
            )
            nn.init.zeros_(self.union_projection.weight)
        else:
            self.union_projection = None

        if config.pair_visual_evidence in {"contact", "union-contact"}:
            self.contact_projection: nn.Linear | None = nn.Linear(
                backbone.hidden_size,
                config.hidden_dim,
                bias=False,
            )
            nn.init.zeros_(self.contact_projection.weight)
        else:
            self.contact_projection = None

        # Rich-geometry evidence is an additive residual created after every
        # common stochastic module. Zero initialization keeps basic/rich
        # common parameters and initial outputs exactly aligned under one seed.
        if config.pair_geometry_evidence == "rich":
            self.rich_geometry_projection: nn.Linear | None = nn.Linear(
                self.rich_geometry_feature_count,
                config.geometry_dim,
                bias=False,
            )
            self.rich_geometry_sampler: nn.Linear | None = nn.Linear(
                self.rich_geometry_feature_count,
                1,
                bias=False,
            )
            nn.init.zeros_(self.rich_geometry_projection.weight)
            nn.init.zeros_(self.rich_geometry_sampler.weight)
        else:
            self.rich_geometry_projection = None
            self.rich_geometry_sampler = None

        # Apache-reference pair evidence is initialized last so enabling the
        # alternative pair construction cannot perturb any existing common
        # parameter initialization under the same global RNG seed.
        self.apache_spatial_pool: ApacheSoftSpatialPool | None = None
        self.apache_box_prompt_encoder: ApacheBoxPromptEncoder | None = None
        self.apache_geometry_encoder: ApacheRelGeomEncoder | None = None
        self.apache_pair_projection: nn.Linear | None = None
        if config.pair_evidence_contract == "apache":
            self.apache_spatial_pool = ApacheSoftSpatialPool(
                backbone.hidden_size,
                n_heads=8,
                num_freqs=16,
                max_octave=7.0,
            )
            self.apache_box_prompt_encoder = ApacheBoxPromptEncoder(
                config.hidden_dim,
                num_freqs=16,
                max_octave=7.0,
            )
            self.apache_geometry_encoder = ApacheRelGeomEncoder(
                config.hidden_dim
            )
            self.apache_pair_projection = nn.Linear(
                backbone.hidden_size * 4 + config.hidden_dim,
                config.hidden_dim,
            )
            nn.init.xavier_uniform_(
                self.apache_pair_projection.weight
            )
            nn.init.zeros_(
                self.apache_pair_projection.bias
            )

            # #98 replaces pair evidence while #118 still owns the legacy
            # sampler. Freeze legacy representation modules that are no longer
            # on the Apache forward path. Box prompts are produced as the
            # #119 context contract but remain frozen until that context stack
            # consumes them.
            self.geometry_encoder.requires_grad_(False)
            self.pair_projection.requires_grad_(False)
            self.apache_box_prompt_encoder.requires_grad_(False)

        self.apache_pair_sampler: ApacheRelatednessPairSampler | None = None
        if config.pair_sampler_contract == "apache":
            self.apache_pair_sampler = ApacheRelatednessPairSampler(
                feature_dim=backbone.hidden_size,
                geo_budget=400,
                final_budget=config.pair_budget,
                rel_dim=256,
                negative_weight=config.apache_pair_negative_floor,
                swap_include=True,
            )
            # Apache stage-1 geometry scoring and stage-2 relatedness replace
            # the historical single-stage sampler and post-transformer pair
            # head. Keep those tensors for legacy checkpoint compatibility,
            # but remove them from the active optimizer.
            self.geometry_sampler.requires_grad_(False)
            self.pair_head.requires_grad_(False)

        self.apache_relation_transformer: ApacheRelationTransformer | None = None
        self.apache_deformable_read: ApacheDeformableRelRead | None = None
        self.apache_relation_interaction: ApacheRelationInteractionBlock | None = None
        if config.relation_context_contract == "apache":
            self.apache_relation_transformer = ApacheRelationTransformer(
                d_model=config.hidden_dim,
                scene_dim=backbone.hidden_size,
                n_self_layers=2,
                n_cross_layers=2,
                n_heads=8,
                ffn_ratio=2.0,
                dropout=config.apache_context_dropout,
            )
            self.apache_deformable_read = ApacheDeformableRelRead(
                d_model=config.hidden_dim,
                n_points=4,
                n_heads=8,
                null_slots=2,
            )
            self.apache_relation_interaction = ApacheRelationInteractionBlock(
                d_model=config.hidden_dim,
                scene_dim=backbone.hidden_size,
                n_dependency_layers=2,
                n_grounding_layers=1,
                n_heads=8,
                ffn_ratio=2.0,
                dropout=config.apache_context_dropout,
            )
            # The Apache context stack replaces the historical pair-only
            # Transformer. Box prompt tokens created by #98 become active here.
            self.relation_transformer.requires_grad_(False)
            assert self.apache_box_prompt_encoder is not None
            self.apache_box_prompt_encoder.requires_grad_(True)

        self.apache_vocab_head: ApacheVocabHead | None = None
        self.apache_sub_text_proj: nn.Linear | None = None
        self.apache_obj_text_proj: nn.Linear | None = None
        self.apache_compose_norm: nn.LayerNorm | None = None
        self.apache_compose_gate: nn.Parameter | None = None
        self.apache_spatial_proj: nn.Module | None = None
        if config.predicate_head_contract == "apache":
            self.apache_vocab_head = ApacheVocabHead(
                d_model=config.hidden_dim,
                text_dim=self.predicate_dim,
                logit_scale_init=5.0,
                projection_layers=2,
                gate_hidden=128,
            )
            self.apache_sub_text_proj = nn.Linear(
                backbone.hidden_size,
                self.predicate_dim,
                bias=False,
            )
            self.apache_obj_text_proj = nn.Linear(
                backbone.hidden_size,
                self.predicate_dim,
                bias=False,
            )
            nn.init.xavier_uniform_(
                self.apache_sub_text_proj.weight
            )
            nn.init.xavier_uniform_(
                self.apache_obj_text_proj.weight
            )
            self.apache_compose_norm = nn.LayerNorm(
                self.predicate_dim
            )
            self.apache_compose_gate = nn.Parameter(
                torch.tensor([0.1, 0.1])
            )

            hidden = max(
                config.hidden_dim * 2,
                self.predicate_dim // 2,
            )
            spatial_first = nn.Linear(
                config.hidden_dim * 2,
                hidden,
            )
            spatial_final = nn.Linear(
                hidden,
                self.predicate_dim,
                bias=False,
            )
            nn.init.xavier_uniform_(spatial_first.weight)
            nn.init.zeros_(spatial_first.bias)
            nn.init.xavier_uniform_(spatial_final.weight)
            self.apache_spatial_proj = nn.Sequential(
                spatial_first,
                nn.GELU(),
                nn.LayerNorm(hidden),
                spatial_final,
            )

            # The Apache head replaces the historical single-query projection
            # and learned prototype adapter. W stays as the text direction.
            self.predicate_projection.requires_grad_(False)
            self.logit_scale.requires_grad_(False)

    def effective_predicate_bank(self) -> Tensor:
        bank = self.predicate_bank
        if (
            self.predicate_adapter_down is None
            or self.predicate_adapter_up is None
        ):
            return bank
        residual = self.predicate_adapter_up(
            F.gelu(self.predicate_adapter_down(bank))
        )
        return F.normalize(bank + residual, dim=-1)

    def predicate_adapter_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith("predicate_adapter_")
        )

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

    def _rich_pair_geometry(self, boxes: Tensor) -> Tensor:
        subject = boxes.unsqueeze(2)
        object_ = boxes.unsqueeze(1)
        scx, scy, sw, sh = subject.unbind(dim=-1)
        ocx, ocy, ow, oh = object_.unbind(dim=-1)

        eps = 1.0e-4
        dx = ocx - scx
        dy = ocy - scy
        distance = torch.sqrt(
            dx.square() + dy.square() + 1.0e-8
        ).clamp_min(eps)

        dx_subject = (
            dx / sw.clamp_min(eps)
        ).clamp(-8.0, 8.0)
        dy_subject = (
            dy / sh.clamp_min(eps)
        ).clamp(-8.0, 8.0)
        dx_object = (
            dx / ow.clamp_min(eps)
        ).clamp(-8.0, 8.0)
        dy_object = (
            dy / oh.clamp_min(eps)
        ).clamp(-8.0, 8.0)

        s_left = scx - sw * 0.5
        s_top = scy - sh * 0.5
        s_right = scx + sw * 0.5
        s_bottom = scy + sh * 0.5
        o_left = ocx - ow * 0.5
        o_top = ocy - oh * 0.5
        o_right = ocx + ow * 0.5
        o_bottom = ocy + oh * 0.5

        inter_w = (
            torch.minimum(s_right, o_right)
            - torch.maximum(s_left, o_left)
        ).clamp_min(0.0)
        inter_h = (
            torch.minimum(s_bottom, o_bottom)
            - torch.maximum(s_top, o_top)
        ).clamp_min(0.0)
        intersection = inter_w * inter_h
        subject_area = (sw * sh).clamp_min(1.0e-6)
        object_area = (ow * oh).clamp_min(1.0e-6)
        intersection_over_subject = (
            intersection / subject_area
        ).clamp(0.0, 1.0)
        intersection_over_object = (
            intersection / object_area
        ).clamp(0.0, 1.0)

        horizontal_gap = torch.maximum(
            torch.maximum(
                o_left - s_right,
                s_left - o_right,
            ),
            torch.zeros_like(dx),
        )
        vertical_gap = torch.maximum(
            torch.maximum(
                o_top - s_bottom,
                s_top - o_bottom,
            ),
            torch.zeros_like(dy),
        )

        direction_cos = dx / distance
        direction_sin = dy / distance

        return torch.stack(
            (
                dx_subject,
                dy_subject,
                dx_object,
                dy_object,
                intersection_over_subject,
                intersection_over_object,
                horizontal_gap,
                vertical_gap,
                direction_cos,
                direction_sin,
            ),
            dim=-1,
        )

    def _apache_cfa_partners(
        self,
        labels: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor] | None:
        index = valid.reshape(-1).nonzero(
            as_tuple=True
        )[0]
        if index.numel() < 2:
            return None
        group = labels.reshape(-1)[index]
        order = torch.argsort(group)
        _, counts = torch.unique_consecutive(
            group[order],
            return_counts=True,
        )
        starts = torch.cumsum(counts, 0) - counts
        start_per = torch.repeat_interleave(
            starts,
            counts,
        )
        count_per = torch.repeat_interleave(
            counts,
            counts,
        )
        position = (
            torch.arange(
                group.shape[0],
                device=group.device,
            )
            - start_per
        )
        random_partner = (
            torch.rand(
                group.shape[0],
                device=group.device,
            )
            * (count_per - 1).clamp_min(1)
        ).long()
        random_partner = torch.minimum(
            random_partner,
            (count_per - 2).clamp_min(0),
        )
        random_partner = (
            random_partner
            + (random_partner >= position).long()
        )
        keep = count_per > 1
        if not bool(keep.any()):
            return None
        source = index[order[keep]]
        destination = index[
            order[
                (start_per + random_partner)[keep]
            ]
        ]
        return (
            (source, destination)
            if source.numel()
            else None
        )

    def _apache_mix_entities(
        self,
        subject_features: Tensor,
        object_features: Tensor,
        labels: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        partners = self._apache_cfa_partners(
            labels,
            valid,
        )
        if partners is None:
            return subject_features, object_features

        source, destination = partners
        count = source.shape[0]
        concentration = torch.full(
            (1,),
            float(self.config.apache_cfa_alpha),
            device=source.device,
        )
        mixing = torch.distributions.Beta(
            concentration,
            concentration,
        ).sample(
            (count,)
        ).reshape(count)
        mixing = torch.where(
            torch.rand(
                count,
                device=source.device,
            ) < self.config.apache_cfa_prob,
            mixing,
            torch.ones_like(mixing),
        ).unsqueeze(-1)

        result: list[Tensor] = []
        for value in (
            subject_features,
            object_features,
        ):
            flat = value.reshape(
                -1,
                value.shape[-1],
            )
            mixed = flat.clone()
            mixed[source] = (
                mixing * flat[source]
                + (1.0 - mixing)
                * flat[destination]
            )
            result.append(
                mixed.reshape(value.shape)
            )
        return result[0], result[1]

    def _apache_pair_evidence(
        self,
        patch_features: Tensor,
        boxes: Tensor,
        valid_boxes: Tensor,
        subject_index: Tensor,
        object_index: Tensor,
        selected_valid: Tensor,
        region_features: Tensor | None = None,
        coverage_grid: Tensor | None = None,
        fill: Tensor | None = None,
        region_iou: Tensor | None = None,
        region_contact: Tensor | None = None,
        cfa_predicate_labels: Tensor | None = None,
    ) -> ApachePairEvidenceOutputs:
        if (
            self.apache_spatial_pool is None
            or self.apache_box_prompt_encoder is None
            or self.apache_geometry_encoder is None
            or self.apache_pair_projection is None
        ):
            raise RuntimeError(
                "Apache pair evidence modules are not configured"
            )

        if region_features is None:
            region_features = self.apache_spatial_pool(
                patch_features,
                boxes,
                coverage=coverage_grid,
            )
            region_features = (
                region_features
                * valid_boxes.to(region_features.dtype).unsqueeze(-1)
            )

        subject_features = _batch_gather(
            region_features,
            subject_index,
        )
        object_features = _batch_gather(
            region_features,
            object_index,
        )
        subject_boxes = _batch_gather(
            boxes,
            subject_index,
        )
        object_boxes = _batch_gather(
            boxes,
            object_index,
        )
        union_boxes = apache_union_box(
            subject_boxes,
            object_boxes,
        )
        contact_boxes = apache_contact_box(
            subject_boxes,
            object_boxes,
        )

        pool_coverage = None
        if coverage_grid is not None:
            subject_coverage = _batch_gather(
                coverage_grid,
                subject_index,
            )
            object_coverage = _batch_gather(
                coverage_grid,
                object_index,
            )
            union_coverage = torch.maximum(
                subject_coverage,
                object_coverage,
            )
            contact_coverage = union_coverage.new_full(
                union_coverage.shape,
                1.0 - 1.0e-4,
            )
            pool_coverage = torch.cat(
                (union_coverage, contact_coverage),
                dim=1,
            )

        pooled = self.apache_spatial_pool(
            patch_features,
            torch.cat(
                (union_boxes, contact_boxes),
                dim=1,
            ),
            coverage=pool_coverage,
        )
        pair_count = subject_index.shape[1]
        union_features = pooled[:, :pair_count]
        contact_features = pooled[:, pair_count:]

        region_metrics = None
        if coverage_grid is not None:
            if (
                fill is None
                or region_iou is None
                or region_contact is None
            ):
                raise ValueError(
                    "mask-aware Apache pair evidence requires fill/iou/contact"
                )
            batch_index = torch.arange(
                boxes.shape[0],
                device=boxes.device,
                dtype=torch.int64,
            ).unsqueeze(1).expand_as(subject_index)
            region_metrics = (
                torch.gather(
                    fill,
                    1,
                    subject_index,
                ),
                torch.gather(
                    fill,
                    1,
                    object_index,
                ),
                region_iou[
                    batch_index,
                    subject_index,
                    object_index,
                ],
                region_contact[
                    batch_index,
                    subject_index,
                    object_index,
                ],
            )

        geometry_features = self.apache_geometry_encoder(
            subject_boxes,
            object_boxes,
            region_metrics,
        )

        valid_float = selected_valid.to(
            subject_features.dtype
        ).unsqueeze(-1)
        subject_features = subject_features * valid_float
        object_features = object_features * valid_float
        union_features = union_features * valid_float
        contact_features = contact_features * valid_float
        geometry_features = geometry_features * valid_float

        if (
            self.training
            and self.config.apache_cfa_prob > 0.0
            and cfa_predicate_labels is not None
        ):
            if cfa_predicate_labels.shape != selected_valid.shape:
                raise ValueError(
                    "CFA predicate labels must match selected pair shape"
                )
            cfa_valid = (
                selected_valid
                & (cfa_predicate_labels >= 0)
            )
            (
                subject_features,
                object_features,
            ) = self._apache_mix_entities(
                subject_features,
                object_features,
                cfa_predicate_labels,
                cfa_valid,
            )

        pair_input = torch.cat(
            (
                subject_features,
                object_features,
                union_features,
                contact_features,
                geometry_features,
            ),
            dim=-1,
        )
        pair_tokens = self.apache_pair_projection(
            pair_input
        ) * valid_float

        subject_xyxy = apache_cxcywh_to_xyxy(
            subject_boxes
        )
        object_xyxy = apache_cxcywh_to_xyxy(
            object_boxes
        )
        box_tokens = self.apache_box_prompt_encoder.encode_pairs(
            subject_xyxy,
            object_xyxy,
        )
        box_tokens = (
            box_tokens
            * selected_valid.to(box_tokens.dtype)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        anchors = torch.stack(
            (
                subject_boxes,
                object_boxes,
                union_boxes,
                contact_boxes,
            ),
            dim=2,
        )
        anchors = (
            anchors
            * selected_valid.to(anchors.dtype)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        return ApachePairEvidenceOutputs(
            pair_tokens=pair_tokens,
            box_tokens=box_tokens,
            anchors=anchors,
            geometry_features=geometry_features,
            subject_features=subject_features,
            object_features=object_features,
        )

    def _forward_impl(
        self,
        image: Tensor,
        boxes: Tensor,
        box_counts: Tensor,
        *,
        encoder_only: bool = False,
        pair_targets: Tensor | None = None,
        cfa_predicate_labels: Tensor | None = None,
        entity_labels: Tensor | None = None,
        coverage: Tensor | None = None,
        fill: Tensor | None = None,
    ) -> tuple[
        tuple[Tensor, ...],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
    ]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must be [B,3,H,W]")
        height = int(image.shape[2])
        width = int(image.shape[3])
        fixed_shape = (
            height == self.config.image_size
            and width == self.config.image_size
        )
        if not fixed_shape:
            if not (
                self.training
                and self.config.allow_training_multiscale
            ):
                raise ValueError(
                    "image spatial size does not match RelationModelConfig"
                )
            if height != width:
                raise ValueError(
                    "Apache multi-scale training requires square images"
                )
            if (
                height % int(self.backbone.patch_size) != 0
                or width % int(self.backbone.patch_size) != 0
            ):
                raise ValueError(
                    "multi-scale image size must be divisible by backbone patch size"
                )
        if boxes.ndim != 3 or boxes.shape[1:] != (self.config.max_boxes, 4):
            raise ValueError("boxes must be [B,max_boxes,4]")
        if box_counts.ndim != 1 or box_counts.shape[0] != image.shape[0]:
            raise ValueError("box_counts must be [B]")
        if boxes.shape[0] != image.shape[0]:
            raise ValueError("image and boxes batch sizes must match")
        if coverage is not None and self.config.pair_evidence_contract != "apache":
            raise ValueError(
                "region coverage is supported only by the Apache pair-evidence contract"
            )
        if fill is not None and coverage is None:
            raise ValueError(
                "region fill requires region coverage"
            )

        batch = image.shape[0]
        box_index = torch.arange(
            self.config.max_boxes, device=boxes.device, dtype=torch.int64
        )
        valid_boxes = box_index.view(1, -1) < box_counts.to(torch.int64).view(-1, 1)
        subject_valid = valid_boxes.unsqueeze(2)
        object_valid = valid_boxes.unsqueeze(1)
        not_self = box_index.view(1, -1, 1) != box_index.view(1, 1, -1)
        valid_pairs = subject_valid & object_valid & not_self

        region_iou = None
        region_contact = None
        coverage_grid = None
        normalized_fill = None
        if coverage is not None:
            if (
                coverage.ndim != 4
                or coverage.shape[0] != batch
                or coverage.shape[1] != self.config.max_boxes
                or coverage.shape[2] <= 0
                or coverage.shape[3] <= 0
            ):
                raise ValueError(
                    "coverage must be [B,max_boxes,g,g]"
                )
            if coverage.dtype == torch.uint8:
                normalized_coverage = coverage.float() / 255.0
            else:
                normalized_coverage = coverage.to(
                    device=boxes.device,
                    dtype=boxes.dtype,
                )
            normalized_coverage = normalized_coverage.to(
                device=boxes.device,
                dtype=boxes.dtype,
            )
            if (
                not torch.isfinite(normalized_coverage).all()
                or (normalized_coverage < 0).any()
                or (normalized_coverage > 1).any()
            ):
                raise ValueError(
                    "coverage values must be finite within [0,1]"
                )

            if fill is None:
                normalized_fill = torch.ones(
                    (batch, self.config.max_boxes),
                    dtype=boxes.dtype,
                    device=boxes.device,
                )
            else:
                if fill.shape != (batch, self.config.max_boxes):
                    raise ValueError(
                        "fill must be [B,max_boxes]"
                    )
                normalized_fill = fill.to(
                    device=boxes.device,
                    dtype=boxes.dtype,
                )
                if (
                    not torch.isfinite(normalized_fill).all()
                    or (normalized_fill < 0).any()
                    or (normalized_fill > 1).any()
                ):
                    raise ValueError(
                        "fill values must be finite within [0,1]"
                    )

            region_iou, region_contact = apache_coverage_pair_metrics(
                normalized_coverage.flatten(2)
            )

        patch_features = self._fused_patch_features(image)
        if normalized_coverage is not None:
            feature_height = int(patch_features.shape[2])
            feature_width = int(patch_features.shape[3])
            coverage_grid = F.adaptive_avg_pool2d(
                normalized_coverage.reshape(
                    batch * self.config.max_boxes,
                    1,
                    normalized_coverage.shape[2],
                    normalized_coverage.shape[3],
                ),
                (feature_height, feature_width),
            ).reshape(
                batch,
                self.config.max_boxes,
                feature_height * feature_width,
            )

        if self.config.pair_evidence_contract == "legacy":
            region_features = self._pool_regions(
                patch_features,
                boxes,
                valid_boxes,
            )
        else:
            if self.apache_spatial_pool is None:
                raise RuntimeError(
                    "Apache pair evidence is not configured"
                )
            region_features = self.apache_spatial_pool(
                patch_features,
                boxes,
                coverage=coverage_grid,
            )
            region_features = (
                region_features
                * valid_boxes.to(region_features.dtype).unsqueeze(-1)
            )

        sampler_geo_loss: Tensor | None = None
        sampler_relatedness_loss: Tensor | None = None
        sampler_pair_negative_weights: Tensor | None = None
        apache_pair_logits: Tensor | None = None
        flat_rich_geometry: Tensor | None = None
        flat_geometry: Tensor | None = None
        pair_slot: Tensor | None = None

        if self.config.pair_sampler_contract == "apache":
            if self.apache_pair_sampler is None:
                raise RuntimeError(
                    "Apache pair sampler is not configured"
                )
            sampler_out = self.apache_pair_sampler(
                boxes,
                region_features,
                box_counts,
                pair_targets=pair_targets,
                entity_labels=entity_labels,
            )
            subject_index = sampler_out.sub_idx
            object_index = sampler_out.obj_idx
            selected_valid = sampler_out.valid_mask
            sampler_logits = sampler_out.geo_logits
            flat_valid = sampler_out.pair_valid
            apache_pair_logits = sampler_out.pair_logits
            if pair_targets is not None:
                sampler_geo_loss = sampler_out.geo_loss
                sampler_relatedness_loss = (
                    sampler_out.relatedness_loss
                )
                sampler_pair_negative_weights = (
                    sampler_out.pair_negative_weights
                )
        else:
            geometry = self._pair_geometry(boxes)
            rich_geometry = (
                self._rich_pair_geometry(boxes)
                if self.rich_geometry_projection is not None
                else None
            )
            flat_geometry = geometry.reshape(
                batch,
                self.config.max_boxes * self.config.max_boxes,
                self.geometry_feature_count,
            )
            flat_valid = valid_pairs.reshape(batch, -1)
            sampler_logits = self.geometry_sampler(
                flat_geometry
            ).squeeze(-1)
            if rich_geometry is not None:
                flat_rich_geometry = rich_geometry.reshape(
                    batch,
                    self.config.max_boxes * self.config.max_boxes,
                    self.rich_geometry_feature_count,
                )
                assert self.rich_geometry_sampler is not None
                sampler_logits = (
                    sampler_logits
                    + self.rich_geometry_sampler(
                        flat_rich_geometry
                    ).squeeze(-1)
                )
            sampler_scores = sampler_logits.masked_fill(
                ~flat_valid,
                torch.finfo(sampler_logits.dtype).min,
            )
            _, pair_slot = torch.topk(
                sampler_scores,
                k=self.config.pair_budget,
                dim=1,
                largest=True,
                sorted=True,
            )

            all_subject = (
                box_index.view(-1, 1)
                .expand(
                    self.config.max_boxes,
                    self.config.max_boxes,
                )
                .reshape(-1)
            )
            all_object = (
                box_index.view(1, -1)
                .expand(
                    self.config.max_boxes,
                    self.config.max_boxes,
                )
                .reshape(-1)
            )
            subject_index = all_subject[pair_slot]
            object_index = all_object[pair_slot]
            selected_valid = torch.gather(
                flat_valid,
                1,
                pair_slot,
            )

        if self.config.pair_evidence_contract == "apache":
            selected_cfa_labels = None
            if cfa_predicate_labels is not None:
                if cfa_predicate_labels.shape != (
                    batch,
                    self.config.max_boxes,
                    self.config.max_boxes,
                ):
                    raise ValueError(
                        "cfa_predicate_labels must be [B,N,N]"
                    )
                if cfa_predicate_labels.dtype != torch.int64:
                    raise ValueError(
                        "cfa_predicate_labels must be int64"
                    )
                batch_index = torch.arange(
                    batch,
                    device=boxes.device,
                    dtype=torch.int64,
                ).unsqueeze(1)
                selected_cfa_labels = cfa_predicate_labels[
                    batch_index,
                    subject_index,
                    object_index,
                ]
                selected_cfa_labels = torch.where(
                    selected_valid,
                    selected_cfa_labels,
                    torch.full_like(
                        selected_cfa_labels,
                        -1,
                    ),
                )

            reference_evidence = self._apache_pair_evidence(
                patch_features,
                boxes,
                valid_boxes,
                subject_index,
                object_index,
                selected_valid,
                region_features=region_features,
                coverage_grid=coverage_grid,
                fill=normalized_fill,
                region_iou=region_iou,
                region_contact=region_contact,
                cfa_predicate_labels=selected_cfa_labels,
            )
            tokens = reference_evidence.pair_tokens
            _reference_box_tokens = reference_evidence.box_tokens
            _reference_anchors = reference_evidence.anchors
            _reference_geometry_features = (
                reference_evidence.geometry_features
            )
            _reference_subject_features = (
                reference_evidence.subject_features
            )
            _reference_object_features = (
                reference_evidence.object_features
            )
        else:
            assert region_features is not None
            subject_features = _batch_gather(
                region_features,
                subject_index,
            )
            object_features = _batch_gather(
                region_features,
                object_index,
            )
            assert flat_geometry is not None
            assert pair_slot is not None
            selected_geometry = torch.gather(
                flat_geometry,
                1,
                pair_slot.unsqueeze(-1).expand(
                    -1, -1, self.geometry_feature_count
                ),
            )
            geometry_features = self.geometry_encoder(
                selected_geometry
            )
            if flat_rich_geometry is not None:
                selected_rich_geometry = torch.gather(
                    flat_rich_geometry,
                    1,
                    pair_slot.unsqueeze(-1).expand(
                        -1,
                        -1,
                        self.rich_geometry_feature_count,
                    ),
                )
                assert self.rich_geometry_projection is not None
                geometry_features = (
                    geometry_features
                    + self.rich_geometry_projection(
                        selected_rich_geometry
                    )
                )

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

        if (
            self.config.pair_evidence_contract == "legacy"
            and (
                self.union_projection is not None
                or self.contact_projection is not None
            )
        ):
            subject_boxes = _batch_gather(boxes, subject_index)
            object_boxes = _batch_gather(boxes, object_index)
            (
                union_boxes,
                contact_boxes,
                contact_valid,
            ) = _pair_union_contact_boxes(
                subject_boxes,
                object_boxes,
                selected_valid,
            )
            if self.union_projection is not None:
                union_features = self._pool_regions(
                    patch_features,
                    union_boxes,
                    selected_valid,
                )
                tokens = tokens + self.union_projection(
                    union_features
                )

            if self.contact_projection is not None:
                contact_features = self._pool_regions(
                    patch_features,
                    contact_boxes,
                    contact_valid,
                )
                tokens = tokens + self.contact_projection(
                    contact_features
                )

        tokens = tokens * selected_valid.to(tokens.dtype).unsqueeze(-1)
        if self.config.relation_context_contract == "apache":
            if (
                self.apache_relation_transformer is None
                or self.apache_deformable_read is None
                or self.apache_relation_interaction is None
            ):
                raise RuntimeError(
                    "Apache relation context is not configured"
                )
            if self.config.pair_evidence_contract != "apache":
                raise RuntimeError(
                    "Apache context requires Apache pair evidence"
                )
            padding_mask = ~selected_valid
            box_token_drop = None
            if (
                self.training
                and self.config.apache_box_token_dropout > 0.0
            ):
                box_token_drop = (
                    torch.rand(
                        batch,
                        device=tokens.device,
                    )
                    < self.config.apache_box_token_dropout
                )

            tokens = self.apache_relation_transformer(
                tokens,
                patch_features,
                box_tokens=_reference_box_tokens,
                pair_padding_mask=padding_mask,
                box_token_drop=box_token_drop,
            )
            projected_scene = (
                self.apache_relation_transformer.projected_scene_map(
                    patch_features
                )
            )
            tokens = self.apache_deformable_read(
                tokens,
                projected_scene,
                _reference_anchors,
            )
            tokens = self.apache_relation_interaction(
                tokens,
                patch_features,
                query_padding_mask=padding_mask,
            )
        else:
            tokens = self.relation_transformer(tokens)
        tokens = tokens * selected_valid.to(tokens.dtype).unsqueeze(-1)

        pair_logits = (
            apache_pair_logits
            if apache_pair_logits is not None
            else self.pair_head(tokens).squeeze(-1)
        )
        predicate_bank = self.effective_predicate_bank()

        predicate_spatial_query: Tensor | None = None
        predicate_alpha: Tensor | None = None
        object_subject_query: Tensor | None = None
        object_object_query: Tensor | None = None
        if self.config.predicate_head_contract == "apache":
            if (
                self.apache_vocab_head is None
                or self.apache_sub_text_proj is None
                or self.apache_obj_text_proj is None
                or self.apache_compose_norm is None
                or self.apache_compose_gate is None
                or self.apache_spatial_proj is None
            ):
                raise RuntimeError(
                    "Apache predicate head is not configured"
                )
            semantic_base = self.apache_vocab_head.proj(tokens)
            predicate_query_raw = self.apache_compose_norm(
                semantic_base
                + self.apache_compose_gate[0]
                * self.apache_sub_text_proj(
                    _reference_subject_features
                )
                + self.apache_compose_gate[1]
                * self.apache_obj_text_proj(
                    _reference_object_features
                )
            )
            predicate_spatial_query = self.apache_spatial_proj(
                torch.cat(
                    (
                        tokens,
                        _reference_geometry_features,
                    ),
                    dim=-1,
                )
            )
            predicate_query = F.normalize(
                predicate_query_raw,
                dim=-1,
            )
            predicate_alpha = self.apache_vocab_head.routing_alpha(
                predicate_bank
            )
            if pair_targets is not None:
                object_subject_query = self.apache_sub_text_proj(
                    region_features
                )
                object_object_query = self.apache_obj_text_proj(
                    region_features
                )
                valid_object_float = valid_boxes.to(
                    object_subject_query.dtype
                ).unsqueeze(-1)
                object_subject_query = (
                    object_subject_query * valid_object_float
                )
                object_object_query = (
                    object_object_query * valid_object_float
                )
            if encoder_only:
                encoder_runtime = (
                    predicate_query_raw,
                    predicate_spatial_query,
                    pair_logits,
                    subject_index.to(torch.int64),
                    object_index.to(torch.int64),
                    selected_valid.to(torch.bool),
                )
            else:
                pred_logits = self.apache_vocab_head.score_query_dual(
                    predicate_query_raw,
                    predicate_spatial_query,
                    predicate_bank,
                    alpha=predicate_alpha,
                )
        else:
            predicate_query_raw = self.predicate_projection(tokens)
            predicate_query = F.normalize(
                predicate_query_raw,
                dim=-1,
            )
            predicate_spatial_query = predicate_query.clone()
            if encoder_only:
                encoder_runtime = (
                    predicate_query,
                    predicate_spatial_query,
                    pair_logits,
                    subject_index.to(torch.int64),
                    object_index.to(torch.int64),
                    selected_valid.to(torch.bool),
                )
            else:
                scale = self.logit_scale.exp().clamp(max=100.0)
                pred_logits = scale * torch.matmul(
                    predicate_query,
                    predicate_bank.transpose(0, 1),
                )

        if encoder_only:
            return (
                encoder_runtime,
                sampler_logits,
                flat_valid.to(torch.bool),
                predicate_query,
                predicate_query_raw,
                predicate_bank,
                predicate_spatial_query,
                predicate_alpha,
                object_subject_query,
                object_object_query,
                sampler_geo_loss,
                sampler_relatedness_loss,
                sampler_pair_negative_weights,
            )
        runtime = (
            pred_logits,
            pair_logits,
            subject_index.to(torch.int64),
            object_index.to(torch.int64),
            selected_valid.to(torch.bool),
        )
        return (
            runtime,
            sampler_logits,
            flat_valid.to(torch.bool),
            predicate_query,
            predicate_query_raw,
            predicate_bank,
            predicate_spatial_query,
            predicate_alpha,
            object_subject_query,
            object_object_query,
            sampler_geo_loss,
            sampler_relatedness_loss,
            sampler_pair_negative_weights,
        )

    def forward_encoder(
        self, image: Tensor, boxes: Tensor, box_counts: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        runtime, _, _, _, _, _, _, _, _, _, _, _, _ = self._forward_impl(
            image,
            boxes,
            box_counts,
            encoder_only=True,
        )
        if len(runtime) != 6:
            raise RuntimeError(
                "open-vocabulary encoder returned the wrong output count"
            )
        return runtime  # type: ignore[return-value]

    def forward(
        self,
        image: Tensor,
        boxes: Tensor,
        box_counts: Tensor,
        *,
        coverage: Tensor | None = None,
        fill: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        runtime, _, _, _, _, _, _, _, _, _, _, _, _ = self._forward_impl(
            image,
            boxes,
            box_counts,
            coverage=coverage,
            fill=fill,
        )
        return runtime

    def forward_training(
        self,
        image: Tensor,
        boxes: Tensor,
        box_counts: Tensor,
        pair_targets: Tensor | None = None,
        cfa_predicate_labels: Tensor | None = None,
        entity_labels: Tensor | None = None,
        coverage: Tensor | None = None,
        fill: Tensor | None = None,
    ) -> RelationTrainingOutputs:
        if (
            self.config.pair_sampler_contract == "apache"
            and pair_targets is None
        ):
            raise ValueError(
                "Apache sampler training requires dense pair_targets"
            )
        (
            runtime,
            sampler_logits,
            sampler_valid,
            predicate_query,
            predicate_query_raw,
            predicate_bank,
            predicate_spatial_query,
            predicate_alpha,
            object_subject_query,
            object_object_query,
            sampler_geo_loss,
            sampler_relatedness_loss,
            sampler_pair_negative_weights,
        ) = self._forward_impl(
            image,
            boxes,
            box_counts,
            pair_targets=pair_targets,
            cfa_predicate_labels=cfa_predicate_labels,
            entity_labels=entity_labels,
            coverage=coverage,
            fill=fill,
        )
        return RelationTrainingOutputs(
            runtime=runtime,
            sampler_logits=sampler_logits,
            sampler_valid=sampler_valid,
            predicate_query=predicate_query,
            predicate_query_raw=predicate_query_raw,
            predicate_bank=predicate_bank,
            predicate_spatial_query=predicate_spatial_query,
            predicate_alpha=predicate_alpha,
            object_subject_query=object_subject_query,
            object_object_query=object_object_query,
            sampler_geo_loss=sampler_geo_loss,
            sampler_relatedness_loss=sampler_relatedness_loss,
            sampler_pair_negative_weights=sampler_pair_negative_weights,
        )
