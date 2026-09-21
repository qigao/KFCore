# Hand models

`vision/core/hand_models` is backend-neutral. Model execution is selected through
`runtime::ExecutionPolicy`; there are no CPU/TensorRT hand model classes and no
`HandPipeline` abstraction.

## Public model boundary

`HandDetector::load` owns three logical models, each with its own Model Package and
execution policy:

```text
Palm detector
  -> hand landmarker (one hand at a time in runtime v1)
  -> gesture classifier (batched across accepted hands)
  -> HandFrame
```

The three routes are independent. A caller may, for example, select TensorRT
for Palm/landmark and ONNX Runtime CPU for the classifier by providing explicit
policies for each package. No implicit fallback is performed.

`HandDetector::load_landmarks` is an explicit two-model entry point accepting
only Palm/landmark packages and their policies. It does not load a classifier,
returns `Gesture::Unknown`, leaves classifier timing at zero, and exposes an
empty classifier execution route. A failed three-model load never selects this
mode automatically. The MediaPipe-specific contract adapter is documented in
[`vision/mediapipe`](../../mediapipe/README.md).

```cpp
#include <kfcore/hand_models/runtime.hpp>

using namespace kfcore;

auto detector = hand_models::HandDetector::load(
    runtime,
    palm_package, palm_policy,
    landmark_package, landmark_policy,
    classifier_package, classifier_policy);

hand_models::HandFrame frame = detector->infer(image);
```

Canonical Model Package types are:

- `hand.palm-detector`
- `hand.landmarker`
- `hand.gesture-classifier`

Palm `[N,8]` is a data-dependent output and uses plugin ABI v1.2 bounded dynamic
Host output. `max_palm_candidates` is a capacity bound, not a fabricated tensor
shape. Hand landmark inference remains batch=1 per accepted hand in runtime v1;
the classifier uses the resulting hand count as its explicit batch.

## Tracking and appearance

Temporal identity is a separate stateful component:

```cpp
#include <kfcore/hand_models/tracking.hpp>

hand_models::HandTrackingOptions options;
options.tracker.minimum_consecutive_frames = 1;
auto tracker = hand_models::HandTracker::create(options);

frame = tracker->update(std::move(frame));
```

When appearance extraction is enabled, provide the source Host image explicitly:

```cpp
options.appearance.enabled = true;
auto tracker = hand_models::HandTracker::create(options);
frame = tracker->update(source_image, std::move(frame));
```

`HandTracker` owns only appearance/ByteTrack temporal state. It does not own an
execution backend or model session. `reset()` clears temporal identity state.
This separation lets inference and temporal tracking be used independently.

A newly created track is intentionally unconfirmed and may return
`track_id == -1`. With `minimum_consecutive_frames = 1`, the next matching frame
confirms it and returns a stable non-negative ID. `reset()` restarts the same
confirmation lifecycle.

ByteTrack is the bounded short-term palm-box association layer. Its raw
`track_id` is a continuity hint, not proof of physical hand identity.
A detection gap strictly shorter than `lost_track_buffer` may preserve a
confirmed ID; at or beyond that boundary the next detection starts unconfirmed.
Appearance extraction remains opt-in because it requires
`update(source_image, frame)` and has additional image sampling cost.

Replacing ByteTrack requires recorded-sequence evidence, including raw ID
switches, track fragmentation, canonical reacquisition rate, ambiguous-frame
duration, gesture cancellation, and false activation measurements. A different
tracker must remain behind the existing `HandTracker` boundary rather than expose
backend-specific state to callers.

The repository does not yet contain a recorded tracking corpus, ByteTrack
baseline, or accepted numerical thresholds for those measurements. They must be
defined together before an alternative tracker can be approved.

`HandAppearanceDescriptor` remains an owned 256-value descriptor split into
Palm plus five finger regions. `valid_parts` and `quality` describe which parts
were sampled reliably.

## Timing

`HandFrame::timings` contains model preprocessing/Palm/landmark/classifier time
from `HandDetector`, plus appearance/tracking time added by `HandTracker`.
`total_ms` accumulates both phases when a detector result is passed through a
tracker.

## Build targets

- `KFCore::hand_model_core` — types, geometry, decoding, appearance, tracking.
- `KFCore::hand_model_runtime` — `HandDetector` and Model Package/runtime execution.
- `KFCore::gesture_interaction` — basic and learned gesture events consuming
  MediaPipe `GestureFrame` values; optional stable identity is supplied by the caller.

The SDK targets are static archives. TensorRT and ONNX Runtime remain runtime-loaded
execution plugins and are not model-specific hand libraries.

## 可选官方 world landmarks

四输出关键点包须声明 `mediapipe-hand-world-v1`，增加 FP32 `world_landmarks [1,63]`。
HandResult 新增可选 world_landmarks（米、旋转至源图像轴）和 right_hand_probability。
该格式的图像 z 按官方 0.4 比例还原；旧三输出路径不改变，两个可选字段为空。
新增字段改变 C++ 二进制布局，调用方须重新编译。官方分类入口见
`vision/mediapipe/include/kfcore/mediapipe/gesture_recognizer.hpp`。
