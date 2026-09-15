# Hand models

`vision/core/hand_models` is backend-neutral. Model execution is selected through
`runtime::ExecutionPolicy`; there are no CPU/TensorRT hand model classes and no
`HandPipeline` abstraction.

## Public model boundary

`HandDetector` owns three logical models, each with its own Model Package and
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
- `KFCore::hand_interaction` — higher-level gesture/temporal semantics consuming
  tracked `HandFrame` values.

The SDK targets are static archives. TensorRT and ONNX Runtime remain runtime-loaded
execution plugins and are not model-specific hand libraries.
