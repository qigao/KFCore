# Third-Party Notices

KFCore first-party code is licensed under the Apache License 2.0.
Bundled third-party components retain their upstream license terms.

| Component | Repository path | Upstream license | Notes |
| --- | --- | --- | --- |
| AprilTag | `apriltag/` | BSD-2-Clause | See `apriltag/LICENSE.md`. |
| PopSift | `sift/vendor/popsift/` | MPL-2.0 | See `sift/vendor/popsift/COPYING.md`. |
| uuid4 | `vendor/uuid/` | MIT OR WTFPL-2.0 | The upstream dual-license notice is embedded in the source. |

Dependencies supplied through Salts, SaltsUtils, vcpkg, or another package
manager are not relicensed by KFCore and remain governed by their respective
upstream licenses.

## Optional browser annotation tool

`tools/gesture_annotator/` downloads (does not bundle) MediaPipe Tasks Vision
`@mediapipe/tasks-vision@0.10.21` from jsDelivr at runtime. MediaPipe code is
licensed under Apache-2.0; see the [upstream license](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)
and [package distribution](https://www.npmjs.com/package/@mediapipe/tasks-vision/v/0.10.21).
The tool separately downloads Google's `hand_landmarker/float16/1` task model;
refer to the [official model documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models)
for its source and applicable terms. These network resources are not relicensed
or redistributed by this repository.

## Optional native gesture conversion

`tools/mediapipe_gesture/` converts Google's official
[`gesture_recognizer/float16/1` model](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/index#models)
to ONNX locally; neither the task bundle nor converted weights are distributed here.
Consult upstream model documentation for applicable model terms.
Offline tools include [TensorFlow (Apache-2.0)](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)
and [tf2onnx (Apache-2.0)](https://github.com/onnx/tensorflow-onnx/blob/main/LICENSE).
They are not additional native runtime dependencies. Pinned versions and model
provenance are recorded in `tools/mediapipe_gesture/`.

## Optional DINOv3 relation-model training

`tools/relation_training/` contains KFCore first-party model, training-loss and
export code. The optional tool environment uses
[PyTorch](https://github.com/pytorch/pytorch) and
[Hugging Face Transformers](https://github.com/huggingface/transformers);
those packages remain governed by their respective upstream licenses and are not
native KFCore runtime dependencies.

The tool can load Meta's DINOv3 models from a caller-selected local directory or
from the Hugging Face DINOv3 collection. DINOv3 source/model materials are
governed by Meta's
[DINOv3 License Agreement](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).
KFCore does not redistribute DINOv3 source code, pretrained weights, or trained
derivative checkpoints. Users are responsible for obtaining model access and
for complying with the applicable DINOv3 terms when using or redistributing
DINOv3-derived materials.
