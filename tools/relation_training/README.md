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

## Relation-specific shared predicate adapter

Static whitening fixes most of the correlation problem in raw CLIP prototypes,
but the remaining low-K/long-tail trade-off motivates a small relation-specific
alignment layer.

The optional model adapter is one shared low-rank residual mapping over the
entire predicate bank:

```text
W_eff = normalize(W + Up(GELU(Down(W))))
```

It is controlled by `RelationModelConfig.predicate_adapter_rank` and
`--predicate-adapter-rank` in the baseline runner.

Properties:

- rank 0 is the exact legacy behavior;
- no bias terms;
- `Up` is zero-initialized, so the adapter starts numerically at the source
  predicate bank;
- the mapping is shared across every predicate, including train-zero-support
  predicates;
- no predicate gets an individual trainable embedding;
- runtime tensor names/shapes and native ONNX ABI are unchanged.

Example:

```powershell
python tools/relation_training/train_baseline.py ^
  ... ^
  --predicate-adapter-rank 16 ^
  --predicate-positive-weight-mode sqrt-balanced ^
  --predicate-positive-weight-cap 20
```

Training reports record adapter rank/parameter count plus source/effective
predicate-bank hashes, Gram diagnostics and row cosine drift. Checkpoints retain
the adapter config and weights so ONNX export reconstructs the same effective
bank.

The controlled experiment in
`.github/workflows/openimages-predicate-adapter.yml` compares rank 0 against a
rank-16 adapter on the exact same K=48, frozen-DINOv3, shared whitened-CLIP,
sqrt-balanced Open Images baseline. It reports overall, seen and
train-zero-support recall separately.

## Train-zero-support predicate supervision mask

A shared train+validation vocabulary can contain predicates that have validation
ground truth but no positive examples in the fixed training split. Treating
those dimensions as ordinary zero labels on every positive training pair gives
the model repeated negative evidence for a class it has never had a chance to
observe positively.

For semantic/open-vocabulary diagnostics, the runner can exclude only those
train-zero-support dimensions from predicate BCE:

```powershell
python tools/relation_training/train_baseline.py ^
  ... ^
  --predicate-positive-weight-mode sqrt-balanced ^
  --predicate-positive-weight-cap 20 ^
  --mask-zero-support-predicates
```

The mask is training-only. It does not change:
- predicate vocabulary;
- predicate bank;
- runtime logits;
- ONNX ABI;
- sampler or pair-existence objectives.

The loss accepts an optional bool `predicate_supervision_mask [V]`. It must
leave at least one predicate supervised and may never hide a positive predicate
label in the current selected training pairs. Masked predicate-logit dimensions
receive zero predicate-loss gradient.

`training.json` and checkpoint metadata record:

```json
{
  "predicate_supervision": {
    "mode": "all",
    "supervised_predicate_indices": [],
    "masked_predicate_indices": []
  }
}
```

The controlled Open Images experiment in
`.github/workflows/openimages-zero-shot-mask.yml` compares the legacy
all-dimensions BCE against `train-supported-only` BCE while keeping the exact
same K=48, shared whitened CLIP prototypes, sqrt-balanced class weights,
DINOv3 backbone, dataset hashes, optimizer and seed.

The comparison reports both overall metrics and the canonical benchmark's
`seen` / `train_zero_support` mRecall groups. This distinguishes improved
long-tail fitting from genuine transfer to predicates with no positive training
examples.

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

## Consolidated relation qualification report

Release evidence for #102 must bind quality and latency to the same immutable
runtime context. KFCore uses a separate context sidecar rather than changing the
existing detector-ceiling schema.

Context schema:

```json
{
  "schema": "kfcore.relation-qualification-context/1",
  "relation_model_sha256": "...",
  "vocabulary_sha256": "...",
  "relation_config_sha256": "...",
  "detector_model_sha256": "...",
  "detector_config_sha256": "...",
  "backend": "onnxruntime-cpu",
  "device": "cpu",
  "relation_model_type": "relation.open-vocabulary",
  "detector_id": "...",
  "max_boxes": 32,
  "vocabulary_size": 19103
}
```

The stable, sorted JSON form is SHA-256 hashed. The latency artifact must carry
that exact `context_sha256`.

Latency schema:

```text
kfcore.scene-behavior-latency/1

samples
context_sha256
stages_ms:
  detector
  tracker
  region_prepare
  relation
  assembly
  temporal
  total

each stage:
  p50 / p90 / p95 / p99 / mean

cardinality:
  detections_mean
  tracked_objects_mean
  relation_edges_mean
  events_mean
  pair_states_mean
```

Collect production timing samples from the typed pipeline:

```cpp
auto timed = pipeline->process_timed(image, seconds);
timing_log
    << kfcore::scene_interaction::scene_behavior_timing_json(
           timed.timing)
    << '\n';
```

Each line uses:

```text
kfcore.scene-behavior-timing-sample/1
```

and carries the exact stage timings/cardinalities from
`SceneBehaviorTiming`.

Aggregate the JSONL samples under the same qualification context:

```bash
python tools/relation_training/summarize_scene_behavior_latency.py \
  --context qualification-context.json \
  --samples-jsonl timing.jsonl \
  --out scene-behavior-latency.json
```

Percentiles use the deterministic nearest-rank definition. The summary validates
finite/non-negative stage values, count fields, and that scene-graph/temporal
timings do not exceed total frame timing.

Create the final JSON + Markdown report with:

```bash
python tools/relation_training/make_relation_qualification_report.py \
  --context qualification-context.json \
  --quality detector-relation-ceiling.json \
  --latency scene-behavior-latency.json \
  --out-json qualification.json \
  --out-md qualification.md
```

The combiner rejects:
- latency collected under a different context hash;
- detector id/model/config drift between quality evidence and context;
- invalid/non-monotonic latency percentiles;
- malformed detector failure decomposition.

This prevents a fast latency run from one backend/model/vocabulary from being
published beside quality numbers from another configuration.

## Detector-box recoverability ceiling

GT-box relation metrics do not reveal whether a failure came from the detector,
the pair sampler, or predicate scoring.  KFCore therefore evaluates detector
boxes with a separate recoverability layer before relation metrics.

Detector input is class-agnostic:

```json
{
  "schema": "kfcore.detector-boxes/1",
  "image": "relative/path.jpg",
  "width": 1280,
  "height": 720,
  "boxes_xyxy": [[10, 20, 110, 220]],
  "scores": [0.93]
}
```

Object class labels are intentionally absent and are never fed to the relation
model.

For each GT object, the evaluator builds the set of detector boxes whose IoU is
at least the configured threshold.  A directed GT pair is recoverable when
there exists at least one subject candidate and one object candidate using two
distinct detector boxes.

This is an **existence ceiling**, not a greedy or Hungarian one-to-one matching
policy.  In particular, one detector box that overlaps both GT endpoints cannot
make a subject/object relation recoverable by itself.

The report separates:

```text
GT object recoverability
directed GT pair recoverability ceiling
sampler recall | recoverable pairs
sampler pair recall | all GT pairs
predicate R@K / mR@K | recoverable triplets
end-to-end R@K / mR@K | all GT triplets
```

At the largest requested K, every GT triplet is assigned to exactly one failure
bucket:

```text
detector miss
sampler miss
predicate miss
recovered
```

and those buckets must sum back to the total GT triplet count.

Detector predictions are score-capped to the model's `max_boxes` using stable
descending score order.  Reports record both the pre-cap and used box counts,
the detector prediction-file SHA-256, detector/model/config provenance strings,
and the IoU threshold.

Use `evaluate_detector_boxes()` to run the actual relation model on detector
boxes.  The evaluator loads the image from the GT manifest, but the relation
box tensor comes only from the detector manifest.

## Open-world predicate objective v2

The historical relation baseline uses exhaustive multi-label BCE. That remains
the default control, but it assumes that every unannotated predicate column on
an annotated positive pair is false. The explicit zero-support and holdout
experiments showed that assumption can suppress semantic transfer.

The first objective-v2 step is a batch-local multi-positive InfoNCE:

```text
--predicate-objective batch-local-infonce
--predicate-contrastive-temperature 0.07
```

For each training batch:

1. keep sampled positive relation pairs;
2. apply the explicit predicate supervision/holdout mask;
3. build the contrast set from the union of predicate directions that remain
   positive somewhere in that batch;
4. align each visual relation query against all of its known positive text
   directions;
5. do not place vocabulary columns that were unobserved in the batch into the
   denominator.

This is deliberately narrower than the final #97 objective. Source-aware safe
negatives, synonym soft positives and explicit inverse negatives are separate
follow-up steps. The important first contract is that the model no longer
receives an exhaustive negative gradient from every vocabulary column.

Multi-label pairs keep every supervised positive. An explicit-holdout-only pair
naturally has no visible predicate positive under InfoNCE and is therefore
skipped for predicate loss while retaining pair-existence and sampler
supervision.

Training evidence reports:

```text
predicate_contrast_set_size
predicate_positive_cosine
predicate_hard_negative_margin
predicate_query_raw_norm
predicate_unobserved_column_fraction
predicate_rows_skipped
```

BCE-only negative reweighting (`--zero-support-negative-weight != 1`) cannot be
combined with batch-local InfoNCE.

`.github/workflows/openimages-predicate-objective-v2.yml` runs a four-arm
controlled ladder on the canonical 256/64 Open Images slice:

```text
bce-baseline
bce-holdout
infonce-baseline
infonce-holdout
```

All arms share the same whitened CLIP bank, frozen DINOv3 ViT-S/16, K=48,
sqrt-balanced positive weighting, architecture, optimizer, seed and three-epoch
schedule. The comparison separates the objective's in-distribution effect from
its robustness to the explicit `contain / holds / ride` predicate holdout.

## Bounded hard negatives for predicate InfoNCE

The first predicate-objective-v2 experiment showed that positive-only
batch-local InfoNCE produces a very small contrast set and collapses strict
explicit-holdout transfer even though pair AP improves.

The next controlled step keeps the same multi-positive InfoNCE objective but
adds a bounded set of hard predicate negatives:

```text
--predicate-objective batch-local-infonce
--predicate-contrastive-hard-negative-count N
```

The candidate pool is intentionally conservative:

```text
train support > 0
AND predicate remains inside the current supervision mask
```

Therefore:

- explicit holdouts cannot become hard negatives;
- natural train-zero-support predicates cannot become hard negatives;
- candidates already positive somewhere in the batch are not duplicated;
- the loss re-applies the supervision mask internally even if a caller passes
  an overly broad candidate mask.

For each batch, candidate hardness is the maximum current visual-query cosine
against that predicate direction. The top `N` candidates are added to the
contrast set. Mining uses detached scores; gradients flow only through the
subsequent selected contrastive logits.

Training evidence records:

```text
predicate_hard_negative_count
predicate_contrast_set_size
predicate_unobserved_column_fraction
contrastive_negative_candidate_indices
```

`.github/workflows/openimages-predicate-hard-negatives.yml` compares four
arms on the same canonical Open Images 256/64, whitened-CLIP, frozen-DINOv3,
K=48 contract:

```text
positive-only-baseline
positive-only-holdout
hard8-baseline
hard8-holdout
```

The experiment answers one narrow question: whether broadening the contrast set
with bounded train-supported negatives restores global text-space ranking and
strict held-out predicate transfer before adding ontology/source-aware
negative weighting.

## Apache relation context stack

Issue #119 ports the three post-pair context stages from
`Maelic/RelateAnything@4a07de9d06f2e3f14309753b7907cf1d3a263b08`.

Enable the complete context slice with:

```text
--pair-evidence-contract apache
--pair-sampler-contract apache
--relation-context-contract apache
```

The execution order is fixed:

```text
pair projection
  -> RelationTransformer
       2 pair self-attention layers
       2 scene/box-token cross-attention layers
  -> DeformableRelRead
       4 anchors: subject/object/union/contact
       8 heads x 4 sampled points
       2 learned null slots
       zero-initialized residual gate
  -> RelationInteractionBlock
       2 pair-dependency self-attention layers
       1 joint pair+scene grounding layer
```

The RelationTransformer cross-attention memory concatenates projected scene
patches with the four subject/object TL/BR Fourier box tokens produced by #98.
Pair-padding masks also mask those box tokens. During training,
`apache_box_token_dropout=0.3` can hide an image's box tokens while keeping
scene patches visible.

Deformable offsets are expressed in units of each anchor's half extent and the
sampled positions are clamped to the image. The residual gate starts at zero,
so adding this module begins as the exact RelationTransformer result.

The final interaction block first models relation dependencies across pairs,
then concatenates pair queries and scene tokens into one self-attention
sequence and reads back only the pair positions.

The legacy pair-only Transformer stays as a checkpoint-compatible control and
is frozen when the Apache context stack is active.

## Apache two-stage relatedness pair sampler

Issue #118 ports the pair-selection contract from the same last Apache-2.0
RelateAnything snapshot used by #98.

Enable it with:

```text
--pair-evidence-contract apache
--pair-sampler-contract apache
```

The sampler has two stages:

```text
all ordered non-self pairs
  -> exact 19-D geometry MLP
  -> top 400
  -> asymmetric visual relatedness
       dot(f_sub(v_i), f_obj(v_j)) / sqrt(d)
  -> top K (default 128)
```

The stage-2 relatedness logit is the runtime `pair_logit` consumed by the
relation score contract. The historical post-transformer `pair_head` is not
used in this mode.

Training passes the dense pair target matrix into the model only through
`forward_training`. Annotated directed pairs and their swapped copies are
forced through both TopK stages so later direction supervision can address the
same pair slots.

Reference sampler losses are computed inside the sampler:

```text
geometry pre-scorer:
  BCE over every valid ordered pair

relatedness:
  focal BCE over stage-1 survivors
  positive weight = 1
  unlabelled floor = 0.3
```

The outer training loss directly consumes these two values in Apache sampler
mode. The older generic dense sampler / selected-pair BCE remains the legacy
control.

The current slice implements the reference PU floor. Category-pair statistical
negative-rate overrides remain a follow-up in #118; they are training-only and
must never become inference inputs.

## Apache RelateAnything pair-evidence reference path

Issue #98 is now anchored to the last Apache-2.0 RelateAnything snapshot,
`Maelic/RelateAnything@4a07de9d06f2e3f14309753b7907cf1d3a263b08`.

Use:

```text
--pair-evidence-contract apache
```

to select the reference pair-evidence construction. This path is intentionally
separate from the earlier KFCore `pair_visual_evidence` /
`pair_geometry_evidence` ablations.

The Apache contract adds:

```text
DINO fused scene map
  -> box-conditioned global SoftSpatialPool
  -> v_sub / v_obj / v_union / v_contact
  -> exact 19-D RelGeomEncoder
  -> concat [sub,obj,union,contact,geometry]
  -> pair projection
```

`SoftSpatialPool` is global cross-attention over every scene patch. A box
provides a query through top-left / bottom-right Fourier prompt tokens; it does
not restrict the receptive field to patches inside the box.

The reference contact zone is the box intersection when endpoints overlap and
the rectangle between their facing edges when they do not.

The 19-D geometry path implements the Apache feature order and
`10*tanh(x/10)` normalization exactly, including box-only fallbacks for
mask-fill / region-IoU / region-contact fields.

### Transitional boundary

This #98 path deliberately does **not** claim the whole Apache architecture:

- pair selection still uses the existing KFCore sampler until #118 lands;
- BoxPromptEncoder outputs are produced/frozen but are not consumed until #119
  ports the relation context stack;
- predicate scoring remains the current KFCore head until #99;
- the current legacy path remains the default for checkpoint compatibility.

The Apache pair-evidence modules are initialized after existing common modules,
so enabling the contract does not perturb common parameter initialization under
the same seed. Legacy representation modules that leave the forward path are
frozen rather than left as dead trainable parameters.

## Contact-only visual evidence attribution

The first endpoint/union/union-contact A/B showed that union pooling alone was
nearly neutral, while adding contact evidence produced the large gain in pair
AP and predicate top-1.

KFCore therefore exposes a fourth research mode:

```text
--pair-visual-evidence contact
```

The mode adds only the zero-initialized contact residual projection. It does
not allocate or apply the union projection.

All visual-evidence modes share identical common parameter initialization:

```text
endpoint
union
contact
union-contact
```

A zero-initialized optional residual makes every mode start from the exact
endpoint behavior. Union-only and contact-only add the same number of trainable
parameters; union-contact adds exactly twice that projection delta.

`.github/workflows/openimages-pair-visual-evidence.yml` now compares all four
modes under the same hard8 InfoNCE + calibration=0.10 contract. This isolates
whether the previous union-contact gain is primarily attributable to contact
evidence or to the combination.

## Rich pair geometry evidence

After contact-only pooling was selected as the lean open-vocabulary visual
baseline, the next #98 attribution keeps that visual evidence fixed and adds a
zero-initialized rich geometry residual.

Historical geometry remains unchanged:

```text
dx, dy, distance,
log(width ratio), log(height ratio),
subject area, object area, IoU
```

Rich mode adds ten normalized features through separate residual branches:

```text
dx / subject width
dy / subject height
dx / object width
dy / object height
intersection / subject area
intersection / object area
horizontal box gap
vertical box gap
cos(relative direction)
sin(relative direction)
```

Relative offsets are clipped to [-8,8]; overlap fractions stay in [0,1].
Box gaps remain in normalized image coordinates and direction terms in [-1,1].

The existing 8-D geometry encoder and sampler are not widened. Instead rich
mode adds:

```text
10-D rich geometry
   -> zero-init residual -> geometry embedding
   -> zero-init residual -> pair sampler logit
```

All historical/common parameters are initialized identically under the same
seed and rich mode starts with bitwise-identical outputs. On the canonical
experiment configuration (`geometry_dim=32`) rich mode adds exactly 330
trainable parameters:

```text
10 * 32 representation residual
+ 10 * 1 sampler residual
```

`.github/workflows/openimages-pair-geometry-evidence.yml` compares
`basic` vs `rich` while fixing:

- contact-only visual evidence;
- hard8 batch-local InfoNCE;
- calibration weight 0.10;
- frozen DINOv3 ViT-S/16;
- canonical Open Images 256/64;
- K=48, shared whitened CLIP prototypes, optimizer and seed.

Sampler recall is deliberately reported rather than forced equal because rich
geometry is also allowed to improve pair selection.

## Source-aware predicate calibration auxiliary

Bounded hard negatives broaden the InfoNCE contrast set and improve retained
seen ranking, but strict held-out predicate transfer remains weak. The next
objective-v2 component separates semantic ranking from absolute/cross-pair
calibration.

```text
--predicate-objective batch-local-infonce
--predicate-contrastive-hard-negative-count 8
--predicate-calibration-loss-weight 0.25
```

The predicate objective becomes:

```text
predicate_loss =
    contrastive_loss
  + lambda_calibration * sigmoid_calibration_loss
```

The calibration auxiliary is source/holdout aware:

- a relation row participates only when it retains at least one visible
  positive predicate;
- visible positives receive target 1;
- negative columns come only from the train-supported predicate candidate mask
  inside the active supervision mask;
- explicit holdouts are excluded;
- natural train-zero-support predicates are excluded;
- holdout-only rows are skipped entirely rather than becoming all-negative
  seen rows.

This intentionally uses the same runtime predicate logits that KFCore exports,
so the auxiliary also trains their absolute scale while InfoNCE continues to
shape relative text-space ranking.

Training evidence records:

```text
predicate_contrastive_loss
predicate_calibration_loss
predicate_calibration_rows
predicate_calibration_rows_skipped
predicate_calibration_column_fraction
```

The controlled experiment keeps hard-negative count fixed at 8 and compares
`lambda_calibration=0` against `0.25` for both normal and explicit
`contain / holds / ride` holdout arms. No pair/sampler, prototype, visual
encoder, optimizer or dataset variable changes in that attribution.

## Pair visual evidence: endpoint / union / contact

RelationModelConfig now exposes:

```text
pair_visual_evidence =
    endpoint
    union
    union-contact
```

The historical endpoint representation remains unchanged:

```text
subject
object
subject - object
subject * object
geometry
    -> pair projection
```

Union/contact evidence is injected as a residual after the historical pair
projection and before the relation transformer:

```text
base_pair_token
    + union_projection(union_pool)
    + contact_projection(contact_pool)
```

Both optional projections are created only after all common stochastic modules
and are zero-initialized. Therefore:

- endpoint keeps the exact historical architecture;
- common parameters are bitwise-identical across evidence modes under one seed;
- union/union-contact start from the exact endpoint output;
- training must demonstrate value before the residual can affect predictions.

The tight union rectangle covers both selected subject/object boxes. Contact is
their positive-area intersection. Non-overlap and edge-touching pairs use an
invalid contact mask and a stable zero contact feature.

The public runtime/ONNX tensor ABI is unchanged. The evidence mode is an
internal model configuration persisted in the training checkpoint.

The first controlled A/B fixes the predicate objective at hard8 batch-local
InfoNCE plus source-aware calibration weight 0.10 and compares:

```text
endpoint
union
union-contact
```

No holdout, sampler, dataset, prototype, optimizer, backbone or training
schedule variable changes in this attribution.

## Train-zero-support negative supervision sweep

The canonical 256-image training split has three predicates with no positive
training examples. Under exhaustive multi-label BCE, those dimensions still
receive negative targets on every annotated positive relation pair. The
zero-support mask diagnostic showed that removing those negatives recovers
zero-shot recall but damages seen-class ranking.

`train_baseline.py` therefore exposes a continuous control:

```text
--zero-support-negative-weight ALPHA
```

For predicate BCE, positive target terms always keep multiplier 1. Only
target=0 terms whose predicate has zero positive train support receive
`ALPHA`. The loss is normalized by the sum of element weights so changing
`ALPHA` changes relative supervision rather than the overall predicate-loss
scale.

Endpoints:

```text
alpha = 1   legacy exhaustive BCE
alpha = 0   equivalent to masking train-zero-support negative-only dimensions
```

The existing `--mask-zero-support-predicates` flag remains available for
backward-compatible diagnostics; it cannot be combined with a non-default
negative-weight value.

`.github/workflows/openimages-zero-support-negative-sweep.yml` evaluates
`1.0 / 0.5 / 0.25 / 0.1 / 0.0` on the exact same K=48 frozen-DINOv3,
sqrt-balanced, whitened-CLIP baseline. The final comparison requires identical
dataset/prototype/common-model contracts and identical sampler recall before
reporting overall, seen and zero-shot mRecall deltas.

## CLIP whitening evidence tolerance

The raw CLIP and whitened-CLIP arms encode the same text prototypes in
independent processes. Their diagnostic Gram statistics can therefore differ
at floating-point tail precision even when the source prototype geometry is
equivalent. The whitening comparison checks every Gram diagnostic field with
`rel_tol=1e-6` and `abs_tol=1e-8` rather than requiring byte-for-byte JSON
float equality.

The whitened output still has a separate strict geometry requirement:
`max_abs_off_diagonal < 1e-5`.

## Experiment comparison recovery

Long-running real-data matrix workflows keep their per-arm JSON evidence as
short-lived GitHub Actions artifacts. If all training arms succeed but only the
comparison job fails, the zero-support sweep can recover without retraining:

- manual `workflow_dispatch` may provide `recompare_run_id`;
- a maintenance push whose commit message contains `[recompare-only]` skips
  prototype/data/training jobs, finds the latest failed completed sweep on
  `master`, downloads its arm artifacts and reruns only the comparator.

Normal experiment pushes still execute the complete controlled sweep. Recovery
never changes arm metrics; it only revalidates and packages existing evidence.

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

## Explicit predicate holdout

Incidental train-zero-support predicates in a small deterministic slice are
useful diagnostics, but they are not a stable open-vocabulary evaluation.
`train_baseline.py` therefore supports an explicit predicate holdout:

```powershell
python tools/relation_training/train_baseline.py ^
  ... ^
  --holdout-predicate contain ^
  --holdout-predicate holds ^
  --holdout-predicate ride
```

An explicit holdout keeps the original annotations and pair-existence targets
unchanged, but excludes the selected predicate dimensions from predicate BCE.
Unlike the ordinary supervision-mask path, this mode is allowed to hide positive
predicate labels by design.

Safety invariants:

- holdout names must be unique and present in the vocabulary;
- every held-out predicate must have positive support in both train and
  validation;
- `--holdout-predicate` cannot be combined with
  `--mask-zero-support-predicates`;
- ordinary predicate masks still fail fast if they hide a positive label;
- runtime / ONNX outputs are unchanged.

Training evidence records the held-out names/indices plus both original and
effective train predicate support. Benchmark grouping keeps the original train
support semantics: `train_zero_support` contains only predicates that truly had
zero positive support in the source train split, `seen` excludes explicit
holdouts, and `predicate_groups.explicit_holdout` reports validation support
and mRecall@K for exactly the requested predicates.

The first controlled holdout set is `contain / holds / ride`, which have
non-trivial support in both train and validation on the canonical 256/64 Open
Images slice.

`.github/workflows/openimages-explicit-predicate-holdout.yml` runs the real
two-arm diagnostic after merge. Both arms use the exact same canonical 256/64
data, shared whitened CLIP prototype tensor, frozen DINOv3 ViT-S/16, K=48,
sqrt-balanced positive weighting, adapter rank 0 and zero-support negative
weight `alpha=0.10`. The only training difference is whether
`contain / holds / ride` are excluded from predicate BCE. The comparator
reconstructs the same holdout group from per-predicate metrics for both arms,
keeps naturally train-zero-support predicates separate, and rejects
data/prototype/model/loss-contract drift before reporting metric deltas.

### Explicit holdout row policy

The default explicit-holdout behavior is `dimension-only`: held-out predicate
dimensions are removed from BCE, while the remaining predicate dimensions on
the same positive pair keep their ordinary exhaustive supervision.

For attribution experiments, the runner also supports:

```text
--holdout-row-policy skip-holdout-only
```

This mode keeps pair-existence and sampler supervision unchanged, but skips
predicate BCE for a sampled positive pair when all of that pair's positive
predicate labels are explicitly held out. Mixed-label pairs that contain both
held-out and supervised positive predicates still train the supervised
predicate dimensions normally.

Each epoch records `predicate_rows` and `predicate_rows_skipped` in
`training.json`. The top-level predicate-supervision evidence records the
selected `holdout_row_policy`.

`.github/workflows/openimages-explicit-holdout-row-policy.yml` compares:
`baseline`, `dimension-only`, and `skip-holdout-only` using the exact same
Open Images 256/64, DINOv3, whitened CLIP, K=48, sqrt-balanced and alpha=0.10
contract.

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

## Apache released-corpus qualification

The Apache sampling implementation is not enough to claim that a checkpoint
used the released training corpus. A full-reference run must first bind its
three training sources to an explicit corpus manifest:

```text
kfcore.apache-released-corpus/1

megasg_clean
vg_raw
hicodet
```

Every resolved source records an immutable origin/revision, the canonical
training-annotation SHA-256 and its post-exclusion image count. A source may be
recorded as `unresolved`, but unresolved provenance deliberately blocks
full-reference qualification.

This distinction matters for the public artifacts: the released model metadata
names `megasg_clean + vg_raw + hicodet`, while the current public RA-4M
dataset tree exposes the MegaSG pack but does not by itself establish the exact
released `vg_raw` and `hicodet` pack identities. Do not substitute a source
name for an artifact identity.

After a real training run finishes:

```bash
python tools/relation_training/apache_release_qualification.py \
  --corpus released-corpus.json \
  --training build/full-reference/training.json \
  --out build/full-reference/released-qualification.json
```

The command fails unless the run has all of the following:

- the released `img_size=448` square model input;
- the released `max_objects=40` / `max_boxes=40` model shape;
- the released `geo_budget=400 -> final_budget=128` pair sampler;
- the released `d_model=512` relation-head width;
- the released `text_dim=512` predicate/query space;
- CUDA BF16 AMP execution (`amp=true`, `amp_dtype=bf16`, no GradScaler);
- exact released scalar hyperparameter and hard-coded structure contracts;
- the last Apache-2.0 source reference
  `4a07de9d06f2e3f14309753b7907cf1d3a263b08`;
- all three source identities resolved;
- exact source annotation hashes and post-exclusion counts;
- matching exclusion, source-column, ontology and pair-opportunity hashes;
- deterministic source-column derivation from the same released pack metadata;
- deterministic pair-opportunity rebuild evidence from the same MegaSG pack;
- matching vocabulary, named predicate/object banks and derived spatial-routing hashes;
- deterministic predicate/object bank derivation from the released text student + tokenizer bundle;
- the released 503,754-draw sampling stream;
- full Apache pair-evidence/sampler/context/vocab-head contracts;
- 12 complete epochs;
- effective batch 128;
- released EMA decay 0.9998;
- benchmark/checkpoint weights sourced from the EMA model rather than raw weights;
- a concrete checkpoint SHA-256.

The resulting qualification artifact binds the corpus manifest, training report
and checkpoint. Cross-dataset/open-vocabulary quality and final TensorRT
qualification remain separate model-evidence gates.

### Released external shape and pair budget

The Apache released graph uses a 448 × 448 square input, 40 padded object slots,
a stage-1 geometric budget of 400 ordered pairs, and a final relation budget of
128. For `apache-reference`, KFCore requires exactly:

```text
image        [1,3,448,448]
boxes        [1,40,4]
geo_budget   400
final_budget 128
pred_logits  [1,128,V]
```

The 400/128 evidence is read from the constructed
`ApacheRelatednessPairSampler`, not inferred from CLI defaults. A focused
synthetic ONNX/ORT gate uses a tiny large-patch backbone to exercise the exact
external shape without making a real DINOv3 448 export part of every CI run.
Legacy and diagnostic runs may still choose smaller shapes.

### Released scalar hyperparameter contract

`apache-reference` treats the released scalar values as a contract, not merely
defaults. Training fails before the expensive run starts if any mutable
optimizer, regularization, model, or objective scalar drifts.

Pinned values include head/backbone LR, weight decay, warmup/cosine floor,
gradient clipping, multi-scale range/rungs, CFA, context/box-token dropout,
pair-negative floor, InfoNCE temperature/negative sampling, all released
`lambda_*` terms, background top-k, and swap margin.

The same canonical validator is run again by released-run qualification.
Training evidence stores both the resolved scalar report and a hard-coded
structure report. The structure report covers the 2+2 relation context, 2+1
interaction stack, 8-head attention, 2× FFN width, deformable 4/8/2 contract,
16/7 positional encoding, two-layer vocabulary projection, and initial logit
scale 5. Legacy and focused diagnostic fixtures remain free to use alternate
values; they simply cannot claim the Apache released contract.

### Released precision contract

The released Apache training recipe uses CUDA automatic mixed precision with
BF16:

```text
amp = true
amp_dtype = bf16
GradScaler = disabled
```

KFCore wraps the relation-model forward and the Apache objective computation in
the same BF16 autocast context. Backward, gradient clipping, optimizer update,
scheduler update, and EMA update then follow without FP16 loss scaling.

CPU tests may exercise the BF16 autocast software path, but a released-run
qualification is accepted only when the recorded training device is CUDA and
CUDA BF16 autocast actually executed. A CPU/FP32/FP16 run is therefore useful
for development but cannot be labeled a released training reproduction.

### Released pair-opportunity rebuild

The released relatedness/background PU weighting consumes
`pair_opportunity.npz`. Artifact SHA-256 alone proves which table was used,
but not that a rebuilt table followed the Apache denominator/numerator
contract. KFCore can deterministically rebuild and audit the table from
`megasg_clean/train`:

```text
scan_box_cap = 400
min_support  = 50

opportunities(cs,co)
  = sum over images of ordered instance pairs
  = n_cs * n_co
  - self pairs on the diagonal

relations(cs,co)
  = relation rows from the SAME pack
  - rows whose endpoint falls outside the 400-box scan window

rate = min(1, relations / opportunities)
```

The pack's `meta.categories` order must exactly equal the relation
vocabulary's object-label order. The rebuild writes the runtime-compatible NPZ
with deterministic ZIP metadata and records SHA-256 for `meta.json`,
`img_meta.npy`, `box_cats.npy`, `rels.npy`, the object-order hash,
algorithm constants, summary counts and the output NPZ hash.

Released qualification requires the rebuild output hash to equal the
`neg_rate_table_sha256` actually consumed by training. It also cross-checks
the MegaSG `meta.json` and `rels.npy` hashes against the spatial-routing
derivation, proving both derived assets came from the same released/rebuilt
MegaSG pack. Sampled/extrapolated rebuilds cannot qualify.

### Released spatial routing derivation

The Apache dual-vocabulary routing gate is warm-started from a data-derived
spatial/semantic flag for every predicate. The released rule is computed from
the original relation packs, not guessed from predicate strings:

```text
spatial_bit = rels[:,3] & 1
per source + predicate:
  spatial = spatial_count >= 0.5 * relation_count
union:
  spatial if ANY supported source marks the predicate spatial
```

KFCore materializes the existing `kfcore.predicate-spatial-flags/1` sidecar
with `apache_spatial_flags.py`. Derivation evidence records every source
pack's `meta.json` and `rels.npy` SHA-256, the exact `>= 0.5` rule, local
support counts and the emitted sidecar SHA-256. Local predicate order is mapped
by name into the union vocabulary; predicates with no support remain semantic.

Full-reference qualification binds the training warm-start's
`spatial_flags_sha256` to this derivation and cross-checks the derived
spatial/semantic counts against the predicate-bank row count. An arbitrary
hand-authored sidecar cannot qualify as the released routing input.

### Released source-column negative mask

The released recipe restricts negative predicate columns for **HICO only**:

```text
source order:
  megasg_clean
  vg_raw
  hicodet

restrict_neg_sources:
  hicodet
```

The runtime table starts with every source allowed to contrast against every
union predicate. The HICO row is then replaced by the exact intersection of
`hicodet/train/meta.json::predicates` with the union predicate vocabulary.
Unknown HICO-local predicates are ignored, matching upstream; local predicate
order does not affect union-column positions.

`apache_source_columns.py` materializes the existing
`kfcore.predicate-source-allow/1` sidecar deterministically and records:
- source order and exact restricted-source set;
- union predicate order/count/hash;
- every source `meta.json` SHA-256;
- each local predicate order/count/hash;
- ignored local predicates;
- allowed predicate order/count/hash;
- emitted sidecar SHA-256.

Released qualification reconstructs the sidecar from this evidence and requires
its SHA-256 to equal the exact asset consumed by training. It also requires each
source `meta.json` hash to match the same pack metadata used by the spatial
routing derivation, and the union predicate-order hash to match the named
predicate bank from the text-bank derivation.

### Released text-bank derivation

The released predicate and object text banks share one distilled text student,
but they do **not** share the same prompt ensemble. Released model metadata pins
the student checkpoint to:

```text
runs/packed/text_student_v2_512/student.pt
sha256 =
e0317830b68ea51e6711fc90d4a35954d0528e5bd78a8d5afd966601ce4ed119
```

The student architecture is fixed at vocab 49,408, token width 128, model width
256, six blocks, four heads, FFN 1,024, output width 512 and max length 32. A
released rebuild uses the colocated local CLIP tokenizer bundle corresponding
to `openai/clip-vit-base-patch32`; the released environment records
`transformers==5.14.1`.

Predicate bank prompts:

```text
"{p}"
"one object is {p} another object"
"a photo of something {p} something"
```

Object bank prompts:

```text
"{p}"
"a photo of a {p}"
```

For each bank, each template is encoded separately, the already-normalized
template vectors are **summed**, and one final L2 normalization is applied.
KFCore's `apache_text_bank.py` reproduces this rule and writes deterministic
named NPZ artifacts plus `kfcore.apache-text-bank-derivation/1` evidence.

```bash
python tools/relation_training/apache_text_bank.py \
  --student-checkpoint /release/text_student.pt \
  --tokenizer-dir /release/tokenizer \
  --vocabulary released-vocab.json \
  --predicate-out pred_embeds_student_photo.npz \
  --object-out obj_embeds.npz \
  --evidence text-bank-derivation.json
```

For `apache-reference`, `--predicate-embeddings` must be the named predicate
NPZ (`predicates + embeddings + templates`), not a bare tensor. Qualification
binds the student SHA, exact student config, tokenizer file hashes, transformers
version, both template sets, label-order hashes, tensor hashes and the final
predicate/object artifact hashes. Legacy training continues to accept the
historical bare predicate tensor.

### Released object text bank

The released objective uses `lambda_obj=0.10`, so object-category text
embeddings are part of the training input identity. For `apache-reference`,
KFCore accepts the Apache-style `obj_embeds.npz` contract:

```text
names       [O]      exact object-label order
embeddings  [O,512]  object text directions
```

The `names` array must exactly equal the object order in the relation
vocabulary. A bare tensor is not sufficient for released qualification because
its row semantics cannot be proven. Training evidence records the NPZ
SHA-256, shape, source dtype and object-label order; the released corpus
manifest carries the same object-bank SHA-256, and qualification requires the
two artifacts to match exactly.

### Released text-space width

The Apache released recipe pins `text_dim=512`. For `apache-reference`,
KFCore therefore requires predicate/text embeddings shaped `[V,512]`; the
semantic and spatial encoder outputs are `[B,K,512]`, and the dynamic
open-vocabulary graph accepts `W[V,512]`.

Legacy/diagnostic experiments may still use smaller text dimensions, but they
cannot pass released-run qualification. The real-DINO full Apache smoke runs
the dynamic-vocabulary ONNX/ORT contract in the 512-D text space.

### Released photometric augmentation contract

The Apache released recipe uses training-only photometric jitter with strength
`0.3`. KFCore reproduces the same transform order and math after square resize:

```text
brightness factor ~ U(0.7, 1.3)
contrast factor   ~ U(0.7, 1.3)
saturation factor ~ U(0.7, 1.3)
clamp [0,1]
```

Contrast is around each channel's spatial mean; saturation is around luma
weights `[0.299, 0.587, 0.114]`. No horizontal flip or geometry transform is
permitted because directional predicates would be falsified. Validation and
qualification evaluation remain unaugmented.

The reference implementation draws factors from ambient `torch.rand` inside
real DDP DataLoader workers. KFCore preserves that stochastic distribution and
records it explicitly, but the single-process logical-DDP runner does **not**
claim bitwise identity with the upstream 4-rank × 16-worker random-call
trajectory.

### Released EMA weight contract

The Apache released model is the EMA model, not the final raw optimizer state.
For `apache-reference`, KFCore reproduces the upstream update rule after each
optimizer step:

```text
d = 0.9998 * (1 - exp(-updates / 2000))
ema = d * ema + (1 - d) * raw
```

Buffers are copied exactly. The final benchmark and saved relation checkpoint
use EMA weights. Training evidence retains both raw and EMA state SHA-256 values
so qualification fails if a raw checkpoint is accidentally substituted.

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
