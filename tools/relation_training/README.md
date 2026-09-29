# KFCore Relation Model Training

This directory contains **first-party KFCore training/export code** for a
DINOv3-backed scene-relation model. It does not import RelateAnything and does
not vendor DINOv3 source code or weights.

The exported graph deliberately matches the native contract already consumed by
`kfcore::relation::RelateAnything`:

```text
image       FP32 [B,3,S,S] RGB [0,1]
boxes       FP32 [B,N,4]   normalized cx/cy/w/h
box_counts  INT64 [B]

        -> KFRelationModel ->

pred_logits  FP32 [B,K,V]
pair_logits  FP32 [B,K]
sub_idx      INT64 [B,K]
obj_idx      INT64 [B,K]
valid_mask   BOOL  [B,K]
```

The C++ runtime, YOLO/ByteTrack scene graph pipeline and temporal ESN layer do
not need to know whether this graph came from the upstream RelateAnything model
or from the KFCore-owned implementation.

## Model v1

```text
DINOv3 ViT backbone
  |
  +-- hidden tap A -- LayerNorm --+
  +-- hidden tap B -- LayerNorm --+--> learned softmax fusion
  +-- hidden tap C -- LayerNorm --+
                                      |
                                      v
                               dense patch features
                                      |
boxes ------------------------> soft region pooling
                                      |
                       subject/object region features
                                      |
boxes --> 8-D pair geometry --> geometry sampler --> top K ordered pairs
                                      |
                    [s, o, s-o, s*o, geometry]
                                      |
                              pair projection
                                      |
                           relation transformer
                              /             \
                             v               v
                    pair-existence      predicate query
                       logit            cosine with W
                                             |
                                      predicate logits
```

The v1 predicate bank `W [V,D]` is fixed in the checkpoint and ONNX export.
It can come from a text encoder selected by the training pipeline, but no
language model is required by native inference.

## Why this is not a RelateAnything fork

The architecture uses standard ideas needed by the problem—dense foundation
features, region pooling, geometry, bounded pair sampling, contextual pair
tokens and cosine predicate scoring—but the implementation and training
contract in this directory are KFCore-owned.

Not implemented in v1:

- RelateAnything's deformable scene read;
- semantic/spatial expert gates;
- positive-unlabeled relatedness objective;
- its data recipes/checkpoints;
- runtime predicate-bank inputs.

Those can be evaluated independently after the baseline has a measured metric.

## DINOv3 dependency boundary

The relation core is independent of one DINOv3 packaging surface. Three thin
adapters are available:

- `HFDinoV3Backbone` for a Transformers model ID or local model directory;
- `MetaDinoV3Backbone` for an official Meta DINOv3 source checkout plus an
  explicitly supplied official weight file/URL;
- `TimmDinoV3Backbone` for the public timm DINOv3 ViT implementation.

The default training example still uses the Transformers adapter:

```python
backbone = HFDinoV3Backbone.from_pretrained(
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    train_backbone=True,
)
```

The token-free real CI qualification uses the public
`timm/vit_small_patch16_dinov3.lvd1689m` weights. CI starts from an empty
Hugging Face cache, downloads `model.safetensors`, records its SHA-256, then
runs the real backbone through KFRelationModel, ONNX export, ONNX checker and
ONNX Runtime parity.

DINOv3 model/source materials are governed by Meta's **DINOv3 License
Agreement**, not the KFCore Apache-2.0 license. KFCore does not commit or
redistribute those weights. Review the applicable DINOv3 terms before training,
creating derivative checkpoints, or redistributing DINOv3-derived materials.

DINOv3 variants may expose prefix/storage tokens or different intermediate
feature APIs. The adapters normalize those differences into the same BCHW
multi-tap `BackboneAdapter` contract consumed by KFRelationModel.

## Environment

Training/export is an offline tool dependency only.

```powershell
uv venv --python 3.11 build/relation-training-env
uv pip install --python build/relation-training-env/Scripts/python.exe -r tools/relation_training/requirements.txt
```

No Python package in this directory is linked into or required by the native
KFCore SDK.

## Construct a model

Predicate embeddings are intentionally caller supplied. Rows must correspond
exactly to the predicate names used for training and deployment.

```python
import torch

from model import HFDinoV3Backbone, KFRelationModel, RelationModelConfig

predicates = ["beside", "holding", "riding"]
W = torch.load("predicate_embeddings.pt", map_location="cpu")

config = RelationModelConfig(
    image_size=448,
    max_boxes=32,
    pair_budget=128,
    tap_indices=(-6, -3, -1),
)
backbone = HFDinoV3Backbone.from_pretrained(
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    train_backbone=True,
)
model = KFRelationModel(backbone, W, config)
```

For ViT-S/16, the official DINOv3 architecture uses 16x16 patches and a
384-dimensional hidden state. The model core does not hard-code those values;
it reads them from the adapter.

## Supervised baseline training

`supervised_relation_loss()` expects:

- `pair_targets [B,N,N]`: 1 for an annotated related ordered pair, else 0;
- `predicate_targets [B,N,N]`: predicate ID for labeled positives, otherwise -1.

```python
from losses import supervised_relation_loss

outputs = model.forward_training(image, boxes, box_counts)
losses = supervised_relation_loss(outputs, pair_targets, predicate_targets)
losses["loss"].backward()
optimizer.step()
```

The training path adds an explicit dense ordered-pair sampler BCE before the
non-differentiable top-K selection. This is required because gradients through
the selected relation head cannot train the discrete top-K indices themselves.
The inference/export `forward()` contract remains unchanged.

This baseline treats unannotated valid pairs as down-weighted negatives. That is
**not** yet a positive-unlabeled objective and should not be presented as
equivalent to RelateAnything training.

## Checkpoint

```python
from checkpoint import save_checkpoint

save_checkpoint(
    "build/relation-v1.pt",
    model,
    backbone_model="facebook/dinov3-vits16-pretrain-lvd1689m",
    predicates=predicates,
    extra={"dataset": "your-dataset-version"},
)
```

The checkpoint records model configuration, predicate names/embeddings, complete
model state and backbone provenance. Do not commit checkpoints containing
DINOv3-derived weights to this repository.

## Export ONNX

```powershell
build/relation-training-env/Scripts/python.exe tools/relation_training/export_onnx.py ^
  --checkpoint build/relation-v1.pt ^
  --out build/relation-v1.onnx ^
  --check
```

The exporter writes raw logits with the exact output names expected by the
native relation runtime, plus `relation-v1.json` metadata recording model
budgets, predicates, DINOv3 provenance and SHA-256 hashes.

The score fusion remains host-side:

```text
sigmoid(calibration_a * (predicate_logit + pair_weight * pair_logit)
        + calibration_b)
```

so thresholds and pair weighting stay runtime knobs.

## Validation philosophy

Repository CI has two complementary jobs.

The synthetic contract uses tiny fake backbones and verifies:

- HF/Meta/timm multi-tap adapter contracts;
- exact native runtime output shapes;
- padded/self pairs never become valid;
- supervised loss backpropagates into backbone and relation heads;
- checkpoint configuration round-trips;
- ONNX checker, tensor descriptors and PyTorch-to-ORT numerical parity.

The real `real-vits16` job is blocking. It anonymously downloads the public
`timm/vit_small_patch16_dinov3.lvd1689m/model.safetensors` into a fresh cache
and executes:

```text
real DINOv3 ViT-S/16
  -> multi-tap [-6,-3,-1]
  -> KFRelationModel
  -> native-contract ONNX
  -> ONNX checker
  -> ONNX Runtime parity
```

The job uploads only a short-lived JSON evidence artifact containing provenance,
hashes, tensor/model dimensions and ORT parity. It does not upload the DINOv3
weights or generated ONNX model.
