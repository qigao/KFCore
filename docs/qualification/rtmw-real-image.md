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
- SHA-256: **discovery run pending; must be pinned before merge**.

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
