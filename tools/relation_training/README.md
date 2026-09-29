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
- `predicate_targets [B,N,N,V]`: exhaustive multi-hot predicate labels for
  each ordered pair.

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

Predicate logits are independent at runtime, so the supervised baseline uses
BCE-with-logits rather than single-class cross-entropy. A positive ordered pair
must have at least one predicate bit set and may have several. On positive
pairs, zero predicate bits are treated as supervised negatives.

This baseline treats unannotated valid pairs as down-weighted negatives and
assumes exhaustive predicate labels on annotated positive pairs. That is
**not** yet a positive-unlabeled objective and should not be presented as
equivalent to RelateAnything training.

## Canonical GT-box benchmark

Before comparing training recipes, freeze one JSONL split and one vocabulary.
The benchmark intentionally uses ground-truth boxes first so detector errors do
not contaminate model-training decisions.

Vocabulary:

```json
{
  "schema": "kfcore.relation-vocab/1",
  "predicates": ["beside", "holding", "riding"],
  "objects": ["person", "bicycle"]
}
```

Annotation record:

```json
{
  "image": "images/frame-001.jpg",
  "width": 1280,
  "height": 720,
  "boxes_xyxy": [[10, 20, 200, 600], [250, 200, 700, 650]],
  "object_labels": ["person", "bicycle"],
  "relations": [[0, 2, 1]]
}
```

`benchmark.py` validates image-relative paths, finite in-frame boxes, ordered
object indices, predicate indices and duplicate GT relations. The vocabulary
and raw JSONL bytes are SHA-256 fingerprinted.

The baseline report fixes the initial metrics:

- sampler recall over unique GT ordered pairs;
- pair AP from `sigmoid(pair_logits)`, with unsampled GT pairs still in the
  recall denominator;
- predicate top-1 accuracy on sampled GT pairs;
- triplet Recall@20/50/100;
- mean Recall@20/50/100 over predicates with non-zero GT support;
- per-predicate support and recall.

Triplets are ranked with the same logit-space rule used by native inference:

```text
predicate_logit + pair_weight * pair_logit
```

Predicted pairs/triplets are de-duplicated before scoring, so duplicate outputs
cannot increase recall or AP. GT-box evaluation does not use detector
confidence.

## CLIP prototype whitening diagnostic

Raw CLIP text prototypes underperformed the dimension-matched orthogonal
control even though dataset hashes, prototype width and trainable parameter
counts were identical. The next diagnostic isolates row correlation.

`clip-whitened` applies symmetric row whitening:

```text
W_white = (W W^T)^(-1/2) W
```

to the normalized raw CLIP bank. Whitening is computed from a symmetric
eigendecomposition and fails if the predicate Gram matrix is numerically
rank-deficient. Rows are normalized again after whitening.

Prototype sidecars record Gram diagnostics:

```text
max_abs_off_diagonal
mean_abs_off_diagonal
min_eigenvalue
max_eigenvalue
```

and, for `clip-whitened`, also retain the raw CLIP source Gram diagnostics.

Example:

```powershell
python tools/relation_training/make_predicate_embeddings.py ^
  --vocabulary vocabulary.json ^
  --out predicates.pt ^
  --mode clip-whitened ^
  --clip-model openai/clip-vit-base-patch32
```

`.github/workflows/openimages-clip-whitening.yml` runs
`identity512 / clip / clip-whitened` on the exact same K=48 Open Images
slice. The compare job verifies identical trainable parameter counts and
dataset hashes, requires the whitened off-diagonal Gram magnitude to be below
`1e-5`, and reports metric deltas between all three geometries.

This is an attribution experiment. If whitening recovers the orthogonal
control, the next useful model change is a relation-specific shared text
adapter rather than generic raw CLIP prototypes.

## Predicate class-balance ablation

After CLIP prototype whitening, medium/high-K metrics are close to the
orthogonal control, while low-K mean recall remains weak. The next controlled
experiment therefore changes only predicate-positive weighting.

For the canonical train split:

```text
P   = number of unique positive ordered relation pairs
c_i = positive pairs carrying predicate i
r_i = (P - c_i) / c_i
```

The runner supports:

```text
none           pos_weight_i = 1
sqrt-balanced  pos_weight_i = sqrt(r_i)
balanced       pos_weight_i = r_i
```

Non-control weights are capped by
`--predicate-positive-weight-cap` (20 in the experiment). A predicate with
zero positive support in the fixed train split receives neutral
`pos_weight=1`; this does not invent supervision because no positive BCE term
exists for that class. Its index is recorded explicitly as train-zero-support
evidence.

CLI:

```powershell
python tools/relation_training/train_baseline.py ^
  ... ^
  --predicate-positive-weight-mode sqrt-balanced ^
  --predicate-positive-weight-cap 20
```

The train manifest alone determines the weights. `training.json` records:
- weighting mode/cap;
- total positive pair count;
- per-predicate positive counts;
- train-zero-support predicate indices;
- final positive weights.

`.github/workflows/openimages-predicate-balance.yml` runs
`none / sqrt-balanced / balanced` on the exact same 256/64 Open Images split,
K=48 and whitened CLIP prototypes. The final compare job rejects any
dataset/prototype/parameter-count drift before reporting metric deltas.

## CLIP whitening evidence tolerance

The raw CLIP and whitened-CLIP arms encode the same text prototypes in
independent processes. Their diagnostic Gram statistics can therefore differ
at floating-point tail precision even when the source prototype geometry is
equivalent. The whitening comparison checks every Gram diagnostic field with
`rel_tol=1e-6` and `abs_tol=1e-8` rather than requiring byte-for-byte JSON
float equality.

The whitened output still has a separate strict geometry requirement:
`max_abs_off_diagonal < 1e-5`.

## Predicate prototype semantic A/B

After the sampler ablation, K=48 is the practical baseline for the next
controlled experiment. The predicate-prototype A/B compares:

```text
identity512  17 orthonormal rows in 512 dimensions
clip         frozen CLIP text prototypes in 512 dimensions
```

The dimension-matched control is important. Comparing the original `V x V`
identity bank directly against 512-d CLIP embeddings would also change the
trainable `predicate_projection` size and parameter count.

CLIP prototypes use public `openai/clip-vit-base-patch32` projected text
embeddings with three fixed prompt templates:

```text
{predicate}
a photo of one object {predicate} another object
the relation between two objects is {predicate}
```

Underscores are converted to spaces. Each prompt embedding is normalized,
the prompt vectors for one predicate are averaged, and the final predicate
vector is normalized again.

`make_predicate_embeddings.py` remains backward compatible:

```powershell
# Original V x V identity bank
python tools/relation_training/make_predicate_embeddings.py ^
  --vocabulary vocabulary.json ^
  --out predicates.pt

# Dimension-matched 512-d control
python tools/relation_training/make_predicate_embeddings.py ^
  --vocabulary vocabulary.json ^
  --out predicates.pt ^
  --mode identity ^
  --dimension 512

# Frozen CLIP semantic prototypes
python tools/relation_training/make_predicate_embeddings.py ^
  --vocabulary vocabulary.json ^
  --out predicates.pt ^
  --mode clip ^
  --clip-model openai/clip-vit-base-patch32
```

Every prototype file gets a JSON sidecar containing the mode, shape,
vocabulary hash, prompt templates/model provenance and a deterministic hash of
the tensor values.

`.github/workflows/openimages-predicate-prototype-ablation.yml` runs the two
arms on the exact same 256/64 Open Images slice at K=48. The final comparison
job verifies identical dataset hashes, `[17,512]` prototype shapes and
identical trainable parameter counts before reporting metric deltas.

## Pair-sampler ablation

The first sampler-stressing frozen baseline established a non-trivial pair
selection bottleneck: sampler recall is below 1.0 with a 24-pair budget.

`.github/workflows/openimages-sampler-ablation.yml` therefore runs a controlled
five-arm experiment on the exact same deterministic 256/64 Open Images slice:

```text
p24-w1   pair budget 24,  sampler loss weight 1
p48-w1   pair budget 48,  sampler loss weight 1
p132-w1  pair budget 132, sampler loss weight 1
p24-w2   pair budget 24,  sampler loss weight 2
p24-w4   pair budget 24,  sampler loss weight 4
```

All arms keep the same frozen DINOv3 backbone, identity predicate prototypes,
seed, model width/depth, optimizer and three-epoch training schedule. The matrix
jobs execute in parallel and upload JSON-only evidence. A final comparison job
verifies that dataset/vocabulary hashes match across arms and records metric
deltas relative to `p24-w1`.

`train_baseline.py` exposes the three loss-component weights explicitly:

```text
--sampler-loss-weight
--pair-loss-weight
--predicate-loss-weight
```

This experiment is diagnostic. It is used to attribute recall loss to pair
sampling before changing predicate representations or unfreezing DINOv3.

## Open Images V7 visual-relationship baseline data

The first real supervised baseline uses Open Images V7 Visual Relationships.
The converter intentionally keeps only binary object-object relationships and
skips `RelationLabel=is` attribute rows, because the current canonical schema
models ordered object pairs rather than object attributes.

First scan train and validation together to lock one shared vocabulary and
produce the image lists expected by the official Open Images downloader:

```powershell
python tools/relation_training/convert_openimages.py scan ^
  --source train=data/oidv7-train-annotations-vrd.csv ^
  --source validation=data/validation-annotations-vrd.csv ^
  --class-descriptions data/oidv7-class-descriptions.csv ^
  --output-dir build/openimages-v7-scan
```

This writes:

```text
vocabulary.json
train.image_ids.txt
validation.image_ids.txt
scan.manifest.json
```

Download only the referenced images with the official Open Images downloader;
it accepts lines such as `train/<ImageID>` and writes `<ImageID>.jpg` into
the requested download directory.

Then convert each split after the pixels are local:

```powershell
python tools/relation_training/convert_openimages.py convert ^
  --split train ^
  --relationships data/oidv7-train-annotations-vrd.csv ^
  --class-descriptions data/oidv7-class-descriptions.csv ^
  --vocabulary build/openimages-v7-scan/vocabulary.json ^
  --image-root data/openimages-images ^
  --output-dir build/openimages-v7-train
```

The converter builds one deterministic object table per image from exact
`(MID, normalized box)` endpoints, decodes the local image for authoritative
pixel dimensions, converts boxes to pixel XYXY, preserves multiple predicate
labels on the same ordered pair, and validates the emitted bytes through
`DatasetManifest.load`.

Two source annotations are deliberately unrepresentable in the current
canonical binary object graph and are skipped **with explicit counters**:
`RelationLabel=is` attribute rows and exact endpoint self-relations.
Malformed coordinates, missing labels/images and other schema errors still
fail fast. Subset manifests record both skip counts so real-data evidence never
silently drops such rows.

Rows may be reordered without changing the canonical JSONL bytes. Missing
images, malformed/out-of-range boxes, missing class descriptions and
self-relations fail fast. The conversion manifest records source/output and
vocabulary SHA-256 values plus row/image/relation counts.

KFCore does not redistribute Open Images annotations or pixels. Review the
Open Images annotation and per-image license terms before using or
redistributing the dataset.

## Real Open Images smoke experiment

`.github/workflows/openimages-relation-baseline.yml` runs the first real-data
training qualification without making it part of every PR gate.

The fixed experiment uses:

```text
32 train images
16 validation images
<= 8 relation endpoints / image
224 x 224 model input
pair budget 32
1 frozen-DINOv3 epoch
small 64-d relation hidden state
```

The workflow downloads the public Open Images VRD annotation CSVs and class
descriptions, selects the lexicographically first eligible images, downloads
those pixels with the official Open Images downloader, converts both splits
through the canonical converter, creates an identity predicate prototype bank,
and invokes `train_baseline.py`.

Identity prototypes intentionally make this a closed-vocabulary optimization
smoke rather than an open-vocabulary result. They remove text-encoder quality
as a variable while proving the real dataset/model/training/benchmark path.

On its first merge the workflow runs once from the master push. It is also
available through `workflow_dispatch` for later qualification runs.

Only JSON evidence is uploaded:

```text
experiment.json
scan.manifest.json
vocabulary.json
train.manifest.json
validation.manifest.json
training.json
benchmark.json
```

Source images, source annotation CSVs, DINOv3 weights and the trained
DINOv3-derived checkpoint are deliberately excluded from the artifact.

## Open Images frozen baseline v1

After the 32/16 pipeline smoke, the first sampler-stressing baseline uses a
larger deterministic Open Images slice:

```text
256 train / 64 validation
4..12 relation endpoints per image
pair budget 24
224px input
frozen DINOv3 ViT-S/16
identity predicate prototypes
3 epochs
```

The minimum object-count filter matters: with six or more objects, the full
ordered-pair set exceeds the 24-pair budget, so sampler recall is no longer
structurally guaranteed to be 1.0.

`.github/workflows/openimages-relation-baseline-v1.yml` runs once on its first
merge and is also available through `workflow_dispatch`. It uploads only JSON
provenance and metrics; pixels, source CSVs, DINOv3 weights and the trained
checkpoint remain outside artifacts.

This baseline remains closed-vocabulary because predicate prototypes are an
identity matrix. Its purpose is to establish a reproducible optimization and
sampling anchor before introducing a semantic text embedding bank or backbone
fine-tuning.

## Seen vs train-zero-support predicate recall

Canonical benchmark reports can optionally stratify predicate mRecall by
whether each predicate had positive support in the training split.

For a validation-supported predicate `i`:

```text
seen               train_count[i] > 0
train_zero_support train_count[i] == 0
```

When `train_baseline.py` produces `benchmark.json`, it passes the canonical
training predicate counts automatically. The report therefore includes:

```json
{
  "predicate_groups": {
    "seen": {
      "predicate_indices": [],
      "predicate_count": 0,
      "validation_triplet_support": 0,
      "mean_recall_at_k": {}
    },
    "train_zero_support": {
      "predicate_indices": [],
      "predicate_count": 0,
      "validation_triplet_support": 0,
      "mean_recall_at_k": {}
    }
  }
}
```

Predicates without validation support are excluded from both group means. A
group with no validation-supported predicates reports `null` mRecall rather
than zero. Existing overall R@K, mR@K and per-predicate metrics remain
unchanged.

This separation is important for semantic/open-vocabulary experiments: class
reweighting can improve seen long-tail predicates, while train-zero-support
recall measures transfer that no positive training label can directly teach.

## Frozen DINOv3 baseline runner

The first reproducible training recipe keeps the DINOv3 backbone frozen and
trains only KFCore-owned relation parameters.

Required inputs:

- canonical train JSONL;
- canonical validation JSONL;
- one shared vocabulary JSON;
- image root;
- a saved predicate embedding tensor `[V,D]`.

Example:

```powershell
python tools/relation_training/train_baseline.py ^
  --train-annotations data/train.jsonl ^
  --validation-annotations data/validation.jsonl ^
  --vocabulary data/vocabulary.json ^
  --image-root data ^
  --predicate-embeddings data/predicate_embeddings.pt ^
  --output-dir build/relation-v1-frozen ^
  --epochs 5 ^
  --batch-size 4 ^
  --image-size 448 ^
  --max-boxes 32 ^
  --pair-budget 128
```

The default backbone is the public
`hf_hub:timm/vit_small_patch16_dinov3.lvd1689m`. It is loaded pretrained,
frozen with `requires_grad=False`, and held in eval mode while tap fusion,
geometry sampler, pair/context layers and predicate head train.

The runner is fail-fast:

- output directory must not already exist;
- train and validation image paths must be disjoint;
- decoded image dimensions must match the manifest;
- examples exceeding `max_boxes` are rejected rather than truncated;
- multi-label targets use the full predicate vocabulary;
- non-finite loss or gradients abort training.

A successful run writes:

```text
relation-v1.pt
training.json
benchmark.json
```

`training.json` records the split/vocabulary hashes, frozen backbone ID,
training configuration, epoch losses and checkpoint SHA-256.
`benchmark.json` is the canonical GT-box validation report.

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
