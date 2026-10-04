# RTMW real-image qualification provenance

The RTMW real-image qualification downloads its source image at CI runtime.
The repository does not vendor the photograph.

## Source image

- Title: **Full-length portrait of a man, standing, facing front**
- Date: 1870
- Source collection: Library of Congress, Prints and Photographs Division
- Library of Congress item: https://www.loc.gov/pictures/item/2005692477/
- Wikimedia Commons description:
  https://commons.wikimedia.org/wiki/File:Full-length_portrait_of_a_man,_standing,_facing_front_LCCN2005692477.jpg
- Permanent Commons revision used for license review:
  https://commons.wikimedia.org/w/index.php?title=File:Full-length_portrait_of_a_man,_standing,_facing_front_LCCN2005692477.jpg&oldid=1262744244
- CI download:
  https://upload.wikimedia.org/wikipedia/commons/d/de/Full-length_portrait_of_a_man%2C_standing%2C_facing_front_LCCN2005692477.jpg
- Expected dimensions: 3281 x 4096 JPEG.
- SHA-256: `8d515fc2125c1db2d66b0cc16745bdda70b181be12c1fbd8e72ab61859a9ebbe`.

The Commons file page marks the work public domain and notes that it was
published before 1931. It also records the Library of Congress source and
states no known restrictions on publication in the United States.

## Qualification use

The image is decoded by the independent Python/OpenCV reference path and
converted to a raw BGR fixture for the C++ qualification executable.

The normal fixture contains fixed person boxes covering:

- full body;
- upper-body truncation;
- a box crossing the left image boundary;
- an extreme wide-aspect crop.

A second fixture applies one deterministic black rectangle over the upper-right
torso/arm before generating reference outputs. This creates a repeatable
occlusion case without requiring another external image.

Detector output is not part of this gate. Every box is fixed so detector drift
cannot contaminate pose runtime qualification.

The first discovery run records real-image E2E coordinate/confidence deltas
without using them as an acceptance threshold. Thresholds are frozen only after
the baseline evidence is reviewed. G1 preprocessing, G2 raw SimCC, G3
coordinate/confidence/visibility decoding, and G4 source projection remain
strict contract gates throughout.


## Baseline evidence and frozen budgets

The first real-image discovery run used the pinned released RTMW-l ONNX and
KFCore ORT CPU backend.

Normal four-case aggregate:

```text
preprocess max / mean levels     3.000003 / 0.446307
raw SimCC max error              0
model coord / confidence error   0 / 0
visibility decode / wiring       5.96e-08 / 0
source projection max error      0.000366 px
E2E source coord max             5.425598 px
E2E confidence max               0.239246
E2E visibility max               0.463270
same SimCC bin                   482 / 532 = 90.60%
```

Deterministic occlusion aggregate:

```text
preprocess max / mean levels     1.000023 / 0.454716
raw SimCC max error              0
model coord / confidence error   0 / 0
visibility decode / wiring       5.96e-08 / 0
source projection max error      0.000244 px
E2E source coord max             4.720215 px
E2E confidence max               0.022927
E2E visibility max               0.409522
same SimCC bin                   121 / 133 = 90.98%
```

The real-image G1 budget is expressed in original 8-bit interpolation levels,
not normalized tensor units: maximum <= 3.01 levels and mean <= 0.50 level.
This models the measured OpenCV fixed-point interpolation versus KFCore float
bilinear difference on real high-frequency image edges.

The G4 source projection budget is <= 0.001 px, covering float32 geometry at
4K source scale while remaining orders of magnitude below one pose output bin.

E2E geometry is gated by the largest one-SimCC-bin source-space displacement in
the fixed fixtures: <= 5.5 px for the four normal cases and <= 4.8 px for the
occluded full-body case. Across every fixture, at least 90% of valid keypoints
must stay in the same SimCC bin, and every semantic region
(body/foot/face/left-hand/right-hand) must retain at least 70% same-bin
agreement.

Confidence and visibility deltas remain recorded evidence, but are not used as
quality thresholds in this first real-image gate because neither is a
calibrated spatial error metric and both can move substantially while the
argmax pose remains in the same output bin.
