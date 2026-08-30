# KFCore

**KFCore** is a C foundation for spatial localization, rigid-body tracking,
pose fusion, and interaction state. The system direction is based on practical
pose sources such as VIO, VINS, AprilTag/marker observations, IMU prediction,
fixed-camera observations, GNSS/RTK, odometry, depth, and LiDAR. These
observations are fused into stable pose, velocity, attitude, rigid-body, and
interaction state for VR, outdoor games, robotics, and drone assistance.

Kalman-family filters are an implementation layer in this architecture, not the
whole product boundary. The core rule is:

```text
observations produce poses;
poses live in explicit frames;
fusion owns state;
applications consume stable predicted state.
```

## System Architecture

KFCore is organized from bottom to top around spatial state, not around one
filter algorithm:

```text
sensor / observation layer:
  camera / headset camera / fixed camera / IMU / AprilTag / marker /
  wheel odometry / GNSS-RTK / depth / LiDAR / barometer / rangefinder

pose source layer:
  VIO / VINS / AprilTag pose / fixed-camera tag observation /
  optical flow / IMU prediction / GNSS position / odometry delta

spatial primitive layer:
  timestamp / coordinate frame / pose / velocity / covariance /
  rigid body / landmark / observation confidence

fusion layer:
  KF / EKF / UKF / error-state navigation fusion /
  outlier gates / dynamic covariance / dropout handling / prediction

state layer:
  device pose / rigid body pose / prop pose / robot pose / drone pose /
  tag landmark map / camera rig state / tracking confidence

interaction layer:
  region trigger / contact / docking / landing / prop interaction /
  player-object events / tracking lost-recovered

shared service layer:
  scene anchors / prop registry / interaction events / multiplayer session /
  authoritative shared world state / state replication

render asset layer:
  3DGS scene assets / prop visual assets / occlusion metadata /
  renderer-facing spatial state

application layer:
  VR / outdoor games / robotics / drone assistance / AR-MR /
  virtual production
```

### Implementation Principles

- **Observation first**
  - A pose update must come from an explicit observation source: VIO/VINS,
    AprilTag, fixed camera, IMU, GNSS, odometry, depth, or another sensor.
  - Every observation should carry timestamp, frame, confidence or covariance,
    and enough metadata to audit why it was accepted.

- **Explicit coordinate frames**
  - `map/world`, `camera_i`, `tag_j`, `body/device`, `imu`, `enu/ned`, and
    `vr/game` frames must be modeled explicitly.
  - Raw detections should not directly drive application state; they must first
    be converted into frame-aware observations.

- **Fusion owns state**
  - Pose, rigid-body, and interaction state are advanced by prediction and
    correction steps, not by ad hoc overwrites from individual sensors.
  - KF/EKF/UKF, navigation fusion, dynamic covariance, Mahalanobis gates, and
    dropout handling are used to keep state stable and explainable.

- **Applications consume stable state**
  - VR/game/render loops consume predicted pose, confidence, and events.
  - Drone and robot control consume fused localization, velocity, attitude,
    environment confidence, and landing/docking cues.
  - MR headsets consume per-user pose and renderer-facing shared state; shared
    scene, prop, interaction, and multiplayer authority stays in the service
    layer.

## Implemented Layers

Current repository modules map into the architecture as follows:

- **Math and filter core**
  - `kalman/`: Takasu covariance-form Kalman filter, UDU Bierman/Thornton
    filter, EKF helpers, covariance-form UKF helpers, signal filters, gates,
    navigation helpers, and 2D/3D fusion helpers.
  - `vendor/miniblas/`: small BLAS/linalg backend used by the core filters.

- **Spatial perception**
  - `apriltag/`: AprilTag detection, tag pose support, JPEG/PNM image loading,
    and fixed-camera marker observation building blocks.
  - JPEG decoding can use libjpeg-turbo through the `jpeg_loader` path.

- **Tracking**
  - `trackers/`: SORT, ByteTrack, C-BIoU, and OC-SORT style visual tracking
    helpers exposed as `KFCore::trackers`.

- **Device I/O**
  - `turbo_serial/`: serial device access for sensor and embedded integration.

The full repository uses CMake, TurboNet, vendored code, and vcpkg packages.
The low-level Kalman core remains suitable for small C integrations, but the
complete spatial stack is not a no-dependency, no-heap embedded profile.

## Numerical Core Quick Check

Are you calculating the inverse of a matrix in your Kalman filter code?
Or does your Kalman filter implementation handle numerical problems with a measurement sensitivity matrix $\mathbf{H}$ and a covariance matrix $\mathbf{R}$ such as this?
For small values of $\epsilon$ a naive implementation of the well-known Kalman filter equations will run into
numerical issues significantly earlier than the `UDU` filter in this library.

$$
H = \begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1+ \epsilon
\end{bmatrix}
$$

$$
R = \begin{bmatrix}
\epsilon^2 & 0 \\
0 &  \epsilon^2
\end{bmatrix}
$$

You could use the `Joseph Form` to mitigate the numerical issues of the standard/vanilla Kalman equations, but this formulation requires more matrix operations and is thus not as fast and potentially still numerically inferior.

## Fusion And Filtering Features

KFCore currently provides these filter and fusion building blocks:

- **Kalman-family filters**
  - Takasu covariance-form Kalman filter.
  - UDU Bierman/Thornton square-root-style Kalman filter.
  - EKF helpers for nonlinear process and measurement models.
  - Covariance-form UKF helpers.
  - 2D fusion layer for IMU prediction, position, navigation-frame velocity,
    body-frame optical/visual velocity, yaw, and ZUPT updates.
  - 3D quaternion error-state fusion layer with nominal `p, v, q, ba, bg`
    state and `dp, dv, dtheta, dba, dbg` covariance.

- **Signal filters and gates**
  - Low-pass, high-pass, moving average, median, clamp, deadband, slew-rate,
    rate, acceleration, and jerk limiting.
  - Hampel outlier filtering, trimmed mean, winsorized mean, Savitzky-Golay
    smoothing, moving RMS, moving min/max, running quantiles, exponentially
    weighted statistics, MAD noise estimation, variance, and standard
    deviation.
  - FIR, generic IIR, biquad IIR, cascaded SOS, DC blocking, peak hold,
    adaptive EMA, One Euro, alpha-beta, alpha-beta-gamma, complementary
    filtering, hysteresis, debounce, majority voting, Schmitt triggers, edge
    detection, sample-and-hold, and dropout decay.
  - Z-score/IQR gates, Huber/Tukey robust weights, Euclidean norm/distance,
    Euclidean distance-gated updates, and Mahalanobis gates.

- **Navigation and IMU helpers**
  - Gyro Euler integration, quaternion attitude math/integration, Euler
    conversion output, roll/pitch/yaw complementary filtering, accelerometer
    gravity gating, body-frame gravity removal, and static/ZUPT detection.
  - Coordinate-frame transforms for local/global point conversion, pose
    inverse/compose, covariance rotation, ENU/NED conversion, and handedness
    convention changes.

## Project Direction

KFCore is evolving toward a lightweight spatial state engine. The base
capability is pose and localization from VIO/VINS/TAG/IMU style observations;
the higher-level capability is stable rigid-body and interaction state.

- **Robotics**
  - AprilTag pose, visual tracking, IMU fusion, wheel odometry, LiDAR/depth
    terrain detection, static/ZUPT detection, and traversability estimation.

- **AR/VR/MR**
  - VIO/VINS pose, marker anchors, fixed-camera observations, IMU prediction,
    rigid props, player/device pose smoothing, and interaction events.
  - Low-cost room-scale or semi-outdoor tracking with AprilTag anchors for
    calibration, drift correction, relocalization, and rigid-body prop/tool
    simulation.
  - For MR, keep the headset responsible for per-user sensing, display, and
    render timing. KFCore-side services should provide the shared scene model,
    tracked props, interaction state, anchors, and multiplayer session state.
  - Use 3DGS as a renderer-facing scene and prop asset representation; fusion
    and interaction logic should consume metric poses, anchors, masks, and
    confidence rather than 3DGS renderer internals.

- **Outdoor games**
  - Outdoor play areas can combine VIO/VINS, GNSS, AprilTag landmarks,
    fixed-camera zones, IMU prediction, and map priors.
  - The game layer should receive filtered pose, trigger regions, prop state,
    confidence, and tracking lost/recovered events rather than raw detections.

- **Drones**
  - RTK/GNSS + VIO/VINS + AprilTag landing markers + IMU +
    barometer/rangefinder/LiDAR fusion for robust assistance across different
    environments.
  - Target state includes position, velocity, attitude quaternion, IMU bias,
    sensor confidence, environment mode, landing/docking cues, and failover
    status.

Different environments should use different fusion strategies:

| Environment | Primary observations | Fusion strategy |
|-------------|----------------------|-----------------|
| Indoor VR / room-scale | VIO/VINS, AprilTag anchors, IMU, fixed cameras | Use VIO for continuous motion, tag anchors for drift correction, and fixed cameras for region/object events. |
| Shared MR venue | Headset VIO, fixed cameras, AprilTag/marker anchors, prop IMU/IR, server session state | Headset handles per-user pose/display timing; shared services own scene anchors, prop identity, interaction events, and multiplayer replication. |
| Outdoor game area | VIO/VINS, GNSS, AprilTag landmarks, IMU, map priors | Fuse local visual-inertial pose with global anchors; gate GNSS jumps and expose confidence to gameplay. |
| Open sky drone | RTK/GNSS, IMU, barometer | Use GNSS position anchors with high-rate IMU prediction; gate jumps and smooth altitude. |
| Urban / multipath | VIO/VINS, IMU, GNSS, LiDAR/depth | Down-weight or reject unstable GNSS; rely more on visual-inertial prediction and local geometry. |
| Indoor / GNSS-denied robot | VIO/VINS, AprilTag, IMU, LiDAR/depth, rangefinder | Vision or marker pose provides position correction; IMU predicts between frames; range/depth stabilizes height and obstacles. |
| Landing / docking | AprilTag/marker, optical flow, IMU, rangefinder | Use marker-relative pose and range for high-precision low-speed control; apply strict outlier gates. |

Maps and landmarks are part of the fusion strategy:

- **Indoor map strategy**
  - Use VIO/SLAM to build or initialize a metric map.
  - Add AprilTag landmarks as repeatable absolute anchors:
    `tag_id -> pose_in_map`.
  - Use IMU prediction and VIO relative motion for high-rate tracking, then use
    AprilTag landmark observations to correct drift, relocalize, and support
    docking/landing.
  - For AR/VR interaction, inside-out camera tracking can provide continuous
    head/controller/prop motion while AprilTag anchors provide absolute room
    corrections and shared map coordinates.
  - The indoor map can include metric landmarks, keyframes/sparse points,
    rooms/corridors/doors, docking stations, and confidence metadata.

- **Outdoor map strategy**
  - Use RTK/GNSS as the global positioning frame when available.
  - Use OpenStreetMap/GIS context as an environment prior rather than as a
    high-accuracy position source.
  - Select visual and fusion modes from map context, such as open road,
    building facade, marker zone, landing pad, low-texture area, or expected
    GNSS multipath area.
  - Known outdoor AprilTag or marker landmarks can be stored in global
    WGS84/ENU coordinates and fused with RTK/GNSS, IMU, and visual pose.

The common map abstraction is:

```text
map provider:
  indoor VIO/SLAM map / AprilTag landmark map / OSM-GIS context /
  LiDAR-height-terrain map

landmarks:
  tag_id / marker_id / semantic anchor -> pose_in_map + covariance + metadata

runtime:
  sensor observation -> map-frame observation -> gated fusion update
```

The long-term architecture is:

```text
sensor input:
  camera / VIO-VINS / AprilTag / marker / depth / LiDAR / IMU /
  odometry / RTK-GNSS / fixed camera

pose source:
  visual-inertial pose / tag pose / GNSS position / odometry delta /
  optical flow / fixed-camera observation

fusion:
  Kalman / EKF / UKF / navigation fusion / environment-specific update policy

output:
  pose / velocity / attitude / rigid-body state / interaction state /
  terrain state / tracking confidence
```

### Spatial State Engine Roadmap

The consolidated direction is a shared spatial state engine for VR, outdoor
games, drones, robotics, AR/MR, and virtual production. The engine should fuse
localization, recognition, rigid-body tracking, human state estimation, map
anchors, and confidence metadata into one stable world model.

```text
sensors:
  fixed cameras / headset cameras / AprilTag / VIO / IMU / RTK-GNSS /
  depth / LiDAR / barometer / rangefinder

observations:
  tag pose / camera pose / VIO pose delta / IMU delta / person detection /
  body mask / skeleton / prop detection / terrain cue / region event

fusion:
  coordinate transform / timestamp alignment / dynamic covariance /
  Kalman-EKF-UKF prediction and correction / outlier gates / dropout handling

world state:
  map anchors / camera rigs / players or actors / rigid props / vehicles /
  drones / terrain regions / interaction events / multiplayer session /
  confidence

outputs:
  robot navigation state / drone localization state / game state /
  MR shared state / cinematic state / render and compositing metadata
```

Game output prioritizes low latency, stable predicted poses, interaction
events, region triggers, and tracking confidence. Cinematic output prioritizes
virtual camera pose, actor root pose, skeleton or mesh state, rigid prop pose,
person masks, occlusion regions, render passes, and AI-video conditioning data.
MR output prioritizes per-user predicted headset/device pose, shared anchors,
tracked props, interaction events, multiplayer replication state, and
renderer-facing scene metadata. 3DGS can provide scene and prop appearance, but
the fusion layer remains the source of metric pose, identity, confidence, and
interaction truth.

### VR, Outdoor Game, And Drone Requirements

The practical direction is not a direct replacement for one tracking system.
Instead, VIO/VINS, fixed commodity cameras, AprilTag anchors, IMU prediction,
GNSS where available, and KF/EKF/UKF fusion should cooperate to produce stable
VR, outdoor game, robot, drone, or cinematic state.

- **Coordinate model**
  - Define `map/world`, `camera_i`, `tag_j`, `body/device`, `imu`, `enu/ned`,
    and `vr/game` frames explicitly.
  - Use shared pose conventions for point transforms, pose inverse/compose,
    covariance rotation, ENU/NED conversion, and left/right-handed game-engine
    conversion.

- **Fixed camera system**
  - Calibrate camera intrinsics: `fx`, `fy`, `cx`, `cy`, and distortion.
  - Calibrate camera extrinsics as `camera_pose_in_map`.
  - Track timestamps, observation confidence, overlapping camera regions, and
    optional camera self-relocalization from known AprilTag landmarks.
  - Use fixed cameras for people/object detection, area triggers, AprilTag
    observation, and drift correction rather than direct high-rate rendering
    poses.

- **AprilTag anchors and props**
  - Store a landmark database with `tag_id`, `pose_in_map`, physical tag size,
    covariance, usage, region, and attached object metadata.
  - Treat tag detections as observations containing `tag_pose_in_camera`,
    `decision_margin`, `hamming`, pose error, timestamp, and confidence.
  - Reject or down-weight poor observations by distance, view angle, margin,
    reprojection/pose error, and pose jump gates.
  - Use tags for room anchors, semi-outdoor anchors, prop identity, prop pose,
    relocalization, docking, landing, and game triggers.

- **Inside-out/VIO**
  - Use VIO/VINS or inside-out tracking for continuous head, controller,
    device, drone, robot, or prop motion.
  - Maintain `vio_pose_in_vio_map` and an alignment from the VIO map into the
    global `map/world` frame.
  - Correct drift when AprilTag or fixed-camera observations are available.

- **Outdoor localization**
  - Combine VIO/VINS with GNSS/RTK and known outdoor tags or visual markers.
  - Use map priors and zone definitions to decide when GNSS, visual-inertial,
    or marker observations should dominate.
  - Expose confidence and environment mode to game logic or drone assistance.

- **IMU prediction**
  - Use gyroscope integration, accelerometer gravity alignment, bias
    estimation, static detection, and ZUPT where applicable.
  - Target typical rates around 100-500 Hz for IMU, 30-90 Hz for VIO/vision,
    15-60 Hz for fixed cameras, and 75 Hz or higher for game output.

- **Fusion and tracking**
  - Fuse AprilTag pose, fixed-camera object pose, VIO pose/delta, IMU,
    RTK/GNSS, depth, rangefinder, and barometer observations as available.
  - Use dynamic measurement covariance, Mahalanobis gates, outlier rejection,
    dropout handling, prediction/correction separation, and latency
    compensation.
  - Prefer realistic roles: people as position/region/action events; tagged
    props as filtered 6DoF rigid bodies; inside-out devices as continuous local
    pose sources.

- **Game interaction output**
  - Export stable `player_pose`, `device_pose`, `prop_pose`, interaction
    events, region triggers, confidence, tracking-lost/recovered events, and
    predicted render poses.
  - The render loop should consume filtered and predicted state, not raw
    low-rate camera observations directly.

- **MR shared service output**
  - Export `scene_anchor`, `player_pose`, `headset_pose`, `device_pose`,
    `prop_pose`, `interaction_event`, session membership, ownership, authority,
    confidence, and predicted render poses.
  - Treat the MR headset as a per-user terminal for sensing, display timing,
    reprojection, and local rendering; shared scene, prop, interaction, and
    multiplayer state should be owned by the service layer.
  - Keep 3DGS assets behind a renderer boundary. The service should publish
    metric transforms, visibility, masks, occlusion hints, and stable IDs that
    a Unity, Unreal, OpenXR, WebXR, or custom 3DGS renderer can consume.

- **Drone assistance output**
  - Export position, velocity, attitude, landing-marker pose, range confidence,
    obstacle or terrain cues, tracking health, and failover status.
  - Flight control should consume fused and gated state, not raw tag or GNSS
    measurements directly.

- **Virtual production output**
  - Export `virtual_camera_pose`, `actor_root_pose`, `actor_skeleton`,
    `prop_pose`, `person_mask`, `occlusion_region`, interaction events, and
    confidence.
  - Use Blender, Unreal, or Unity to keep camera paths, rigid props,
    collisions, lights, masks, depth, and render passes deterministic.
  - Use AI video models as a downstream cinematic rendering or style layer,
    not as the primary source of physical tracking truth.

Roadmap modules:

```text
frame_transform:
  common map/camera/tag/body/imu/game coordinate conversion

map_landmark_db:
  tag and semantic anchor storage with pose, covariance, and metadata

fixed_camera_observation:
  camera extrinsics + AprilTag/person/object detections -> map-frame observations

object_pose_filter:
  filtered prop/player/object state with confidence and dropout handling

active_ir_keypoint_tracker:
  non-blinking IR LED keypoints for rigid-body edges, axes, tips, muzzles,
  grips, and visibility masks; marker pose provides identity and association

rigid_body_imu_predictor:
  IMU pose in body frame, gyro/accelerometer bias, timestamp sync, latency
  compensation, rolling-shutter compensation, and visual dropout bridging

insideout_tag_relocalization:
  VIO map alignment and drift correction from known anchors

game_state_output:
  stable poses, events, confidence, and predicted render state

mr_shared_state_output:
  shared anchors, per-user headset/device poses, prop poses, interaction
  events, ownership, authority, confidence, and predicted render state

multiplayer_spatial_service:
  authoritative shared world state, session membership, prop ownership,
  interaction event ordering, and state replication

gs3d_render_boundary:
  stable IDs, transforms, masks, occlusion hints, and renderer-facing metadata
  for 3DGS scene and prop assets

person_interaction_tracker:
  person mask, region, skeleton/root state, contact, sit/stand, and motion events

cinematic_state_output:
  virtual camera, actor, prop, mask, occlusion, confidence, and render metadata
```

### Bottom-Up TODO

- [ ] Define the common spatial data contracts:
  `timestamp`, `frame_id`, `pose`, `velocity`, `covariance`, `confidence`,
  `source_id`, and observation status.
- [ ] Define coordinate-frame conventions for `map/world`, `camera_i`,
  `tag_j`, `body/device`, `imu`, `enu/ned`, and `vr/game`, including
  handedness and unit conventions.
- [ ] Finish the frame transform layer for point, pose, covariance, ENU/NED,
  and game-engine frame conversion.
- [ ] Add a tag landmark database with `tag_id`, `pose_in_map`, physical size,
  covariance, region, usage, and attached-object metadata.
- [ ] Add fixed-camera observation adapters that convert camera extrinsics and
  AprilTag/person/object detections into map-frame observations.
- [ ] Add VIO/VINS adapter interfaces for pose, pose delta, covariance,
  timestamp, tracking quality, and relocalization state.
- [ ] Add IMU prediction interfaces for gyro integration, gravity alignment,
  bias estimation, static detection, and ZUPT.
- [ ] Add GNSS/RTK and outdoor marker observation interfaces for outdoor game
  areas and drone assistance.
- [ ] Define the rigid-body state model for devices, props, robots, drones,
  and tracked objects.
- [ ] Add rigid-body marker calibration:
  `body_id`, `marker_id`, `marker_pose_in_body`, physical marker size,
  interaction points, axes, edge keypoints, collision proxies, and visibility
  masks.
- [ ] Add active IR LED keypoint layout for irregular rigid bodies:
  non-blinking IR LEDs define sword tips/blade axes, toy-gun muzzles/sights,
  grips, edges, and other body-local keypoints.
- [ ] Add active IR LED detector:
  IR-pass camera support, thresholding, blob extraction, centroid estimation,
  exposure controls, ambient-light rejection, and per-keypoint confidence.
- [ ] Add marker-guided IR LED association:
  marker identifies `body_id` and gives initial pose; projected expected LED
  positions are matched to non-blinking IR blobs by geometry, distance,
  visibility mask, and temporal consistency.
- [ ] Add pose refinement from IR keypoints:
  refine `body_pose` from known 3D LED keypoints and detected 2D centroids,
  reject outliers, and fall back to marker-only or prediction-only state when
  too few keypoints are visible.
- [ ] Add rigid-body IMU model:
  `imu_pose_in_body`, gyro/accelerometer bias, sampling rate, timestamp sync,
  static detection, calibration state, and confidence.
- [ ] Add IMU visual compensation:
  use IMU prediction to compensate camera latency, fast motion, rolling shutter,
  and short-term marker/IR occlusion.
- [ ] Add marker/IR/IMU fusion:
  marker gives identity and absolute correction, non-blinking IR LEDs refine
  edges/keypoints, and IMU predicts high-rate pose between visual updates.
- [ ] Implement object/rigid-body fusion with prediction, correction, dynamic
  covariance, Mahalanobis gates, dropout handling, and confidence decay.
- [ ] Implement VIO/tag relocalization to align local VIO maps into the shared
  `map/world` frame.
- [ ] Implement interaction-state extraction: region triggers, contact events,
  docking/landing state, tracking lost/recovered, and prop/player events.
- [ ] Implement VR/game output adapters for predicted `player_pose`,
  `device_pose`, `prop_pose`, interaction events, confidence, and latency
  compensation.
- [ ] Define MR shared-state contracts:
  `scene_anchor`, `headset_pose`, `device_pose`, `prop_pose`,
  `interaction_event`, `session_id`, `player_id`, ownership, authority,
  confidence, and predicted render timestamps.
- [ ] Implement multiplayer spatial service state:
  authoritative shared world state, session membership, prop ownership,
  interaction event ordering, conflict resolution, and state replication.
- [ ] Implement MR headset adapter boundaries:
  headset-local VIO/display timing/reprojection stays on the headset; shared
  anchors, props, interactions, and multiplayer state come from KFCore services.
- [ ] Implement 3DGS renderer boundary:
  publish stable scene/prop IDs, metric transforms, masks, occlusion hints,
  visibility, and confidence for Unity, Unreal, OpenXR, WebXR, or custom 3DGS
  renderers without coupling fusion state to renderer internals.
- [ ] Implement outdoor game output adapters for play-area localization,
  global/local anchor confidence, region events, and tracking health.
- [ ] Implement drone assistance output adapters for position, velocity,
  attitude, landing-marker pose, range confidence, terrain/obstacle cues,
  failover state, and control-facing health.
- [ ] Add replayable datasets and regression tests for indoor VR, outdoor game,
  and drone landing/docking scenarios.
- [ ] Add benchmarks for observation ingestion, frame transforms, fusion update,
  prediction latency, and application output latency.

## Implementations

| Feature                               | `UDU`      | `Takasu`       |
|---------------------------------------|:----------:|--------------:|
| Numerical Stability                   | Excellent  | Good          |
| Speed                                 | Fast       | Very fast     |
| Outlier detection                     |   ✅        |       ✅       |
| C implementation available            |   ✅        |       ✅       |
| No measurement preprocessing req.     |   ❌        |       ✅       |
| Covariance matrix directly available  |   ❌        |       ✅       |

The `UDU` formulation offers superior numerical stability but requires an UDU
decomposition of the covariance matrix (hence the name) and additionally a
decorrelation of the measurements before they processing. This is not
the case for the `Takasu` formulation which is basically an efficient
implementation of the vanilla Kalman filter equations.

If speed is the top priority and an optimized LAPACK/BLAS implementation is
available, consider using the Takasu formulation. However, it may struggle with
stability in scenarios involving:

 - A large number of state variables
 - A large number of measurements
 - Very high precision measurements
 - Poorly conditioned measurement sensitivity matrices $\mathbf{H}$

## Benchmarks

The in-tree C benchmark is part of the TinyTest executable:

```sh
kfcore_tests --filter "benchmarks core routines"
```

It benchmarks the C update routines with a 15 element state vector, a 15x15
covariance matrix, and a 3x1 measurement vector:

- `kalman_takasu update 15x3`
- `kalman_udu update 15x3`
- `ekf_takasu update 15x3`
- `ekf_udu update 15x3`
- `ukf update 15x3`

Build configuration matters. Use an optimized build, such as `Release` or
compiler flags equivalent to `-O2`, for meaningful timing numbers. Debug builds
are useful only to confirm that the benchmark runs.

Historical standalone benchmark results:

| Average Run Time Test                                                  | `UDU` C | `Takasu` C |
|------------------------------------------------------------------------|---------|------------|
| Intel i5-13600KF Desktop CPU - Kalman Update Routine                   | 2.95 us | 1.97 us    |
| STM32F429 180 MHz Embedded CPU (ARM Cortex M4) - Kalman Update Routine | 103 us  | 135 us     |
| STM32F429 180 MHz Embedded CPU (ARM Cortex M4) - Kalman Prediction     | 593 us  | 393 us     |

Historical table based on commit `2b35963`, built with `-O2`.


## Getting Started

### Installation

#### Cloning the Repository

Clone the repository:

    git clone https://github.com/jnz/KFCore.git

### CMake Integration

KFCore is built as CMake targets. The core Kalman and signal/navigation helpers
are exposed through `KFCore::kfcore`; the visual tracking helpers are exposed
through `KFCore::trackers`.

```cmake
find_package(KFCore CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE KFCore::kfcore)
```

For tracker support:

```cmake
find_package(trackers CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE KFCore::trackers)
```

### 可选 TensorRT YOLO 跟踪

`tensorrt_yolo/` 提供可安装的 `KFCore::yolo_tracking`、
`KFCore::tensorrt_yolo` 和 `KFCore::yolo_opencv`。它们随 KFCore 构建；TensorRT
从标准 user preset 的 `TENSORRT_ROOT` 查找 SDK，不会把开发机路径或运行时 DLL
复制进工程。部署契约、容量限制、图片序列示例和 tracking-only/full/install 命令见
[tensorrt_yolo/README.md](tensorrt_yolo/README.md)。

### TensorRT 模型能力矩阵

模型资产统一从 `KFCORE_MODEL_ROOT` 解析；仓库的标准 user preset 默认把它设为
`${sourceDir}/yolo-models`。ONNX、测试图片和 sidecar matrix 在模块中使用相对此根的路径，
TensorRT engine 使用
`tensorrt/${KFCORE_TENSORRT_ENGINE_PROFILE}/<model>.engine`；手部 engine 使用
`hand_gesture_model/tensorrt/${KFCORE_TENSORRT_ENGINE_PROFILE}/<model>.engine`。
应用模型路径的解析顺序为：显式命令行路径、进程环境中的
`KFCORE_MODEL_ROOT`/`KFCORE_TENSORRT_ENGINE_PROFILE`、明确报错。CPU ONNX 与
TensorRT engine 都使用固定相对文件名，因此不会扫描目录或猜测模型。GFPGAN、Age/Gender 和
FaceMesh 等可选阶段仍须显式启用。

`CMakeUserPresets.json` 的 `environment` 只传给 CMake、build 和 CTest 子进程，不会修改启动
CMake 的 PowerShell。直接运行 `.exe` 时，应在该终端设置同名环境变量，或继续使用显式 CLI
路径；CLI 始终优先。

`KFCore::tensorrt_runtime` 是模型无关的同步执行边界，`KFCore::face_models`
在它之上提供严格的 Face68、ArcFace 与年龄/性别合约；两者均随 KFCore 构建。Face adapters
只接收调用方已经准备好的 NCHW FP32 Tensor（host 或 CUDA device），不会接收原图，也不会
隐式执行下表所列的对齐、裁剪、通道变换或归一化。

| 模型/adapter | 输入边界 | engine 输出契约 | adapter 返回值 | adapter 外仍需完成的预处理 |
|---|---|---|---|---|
| YOLO Compact NMS / EfficientNMS (`KFCore::tensorrt_yolo`) | `ImageView`；内部使用 `KFCore::image_processor` letterbox | Compact `[N,max_detections,6]`，或 EfficientNMS 的 `num_dets/boxes/scores/labels` | 已校验的 detection frame | 调用方提供正确像素格式；YOLO adapter 已执行其声明的 letterbox/通道及数值变换 |
| Face68 (`TensorRtFace68`) | 已准备的 FP32 `[N,3,256,256]` | FP32 `[N,68,3]`，以及 engine 中存在时仍须绑定的 heatmap | 68 个 raw `(x,y,score)`；`x/y` 位于模型 256 像素坐标系 | 基于 bbox 的 affine crop 到 256×256、BGR、除以 255；调用方还需把输出坐标映射回原图 |
| ArcFace (`TensorRtArcFace`) | 已准备的 FP32 `[N,3,112,112]` | FP32 `[N,512]` | 512 个 raw embedding 值，不隐式 L2 normalize | 基于 5 点人脸关键点的 similarity align 到 112×112、RGB、`value / 127.5 - 1` |
| Age/Gender (`TensorRtAgeGender`) | 已准备的 FP32 `[N,3,224,224]` | FP32 `[N,2]` | 两个 raw logits；语义顺序不命名 | face ROI resize 到 224×224、RGB、ImageNet mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]` normalize |
| InSwapper (`TensorRtInSwapper`) | target FP32 `[1,3,128,128]` + projected source FP32 `[1,512]` | FP32 `[1,3,128,128]` | 拥有内存的 RGB CHW `[0,1]` 输出 | 5 点对齐、RGB `/255`；ArcFace embedding 还须经 512×512 sidecar matrix 投影并 L2 normalize |
| GFPGAN (`TensorRtGfpGan`) | 已准备的 FP32 `[1,3,512,512]` | FP32 `[1,3,512,512]` | 拥有内存的 RGB CHW `[-1,1]` 输出 | 5 点对齐、RGB、`value / 127.5 - 1`；paste-back 与 blend 属于应用层 |

这些 face adapter 不会替调用方完成上述预处理。真实 engine 的 zero-input smoke 只验证
TensorRT/CUDA 绑定、执行、输出有限值与结果尺寸；raw outputs 也只保留模型输出语义。两者都不是
模型 accuracy、标签顺序或业务阈值的验证，部署前仍需使用有 provenance 的 golden samples。

Windows 的标准 `win-release-user` 从 `CMakeUserPresets.json` 提供 `TENSORRT_ROOT`、
`KFCORE_MODEL_ROOT` 与 `KFCORE_TENSORRT_ENGINE_PROFILE`。启用真实模型测试后，Face68、
ArcFace、Age/Gender、InSwapper 与 GFPGAN engine 都按上述 profile 目录和固定模型文件名解析；
缺少或越出模型根目录时在 configure 阶段失败。

### TensorRT 12face 换脸应用

`KFCore::face_applications` 把上述 prepared-tensor adapters 组成同步单脸应用：
YOLOv12-face（唯一检测入口）→ Face68 → ArcFace → InSwapper，并可显式启用
GFPGAN 与 Age/Gender。它依赖 OpenCV Lite 的 `core/imgproc`，命令行示例另外依赖
`imgcodecs`。每帧由 ImageProcessor 上传一次后供检测与 Face68/ArcFace/InSwapper/GFPGAN
CUDA 预处理复用，随后以 device tensor 直接进入 TensorRT；Age/Gender 保留 CPU ROI resize。
模型 I/O、matrix sidecar、所有权、非重入约束、构建和真实模型验证命令见
[face_applications/README.md](face_applications/README.md)。

### Hand 与 MediaPipe Landmark CPU/GPU

`vision_models/` 提供 OpenCV-free 的 ONNX Runtime CPU 与 TensorRT CUDA 两条线路：
Palm → 21 点 hand landmark → INT64 gesture classifier → KFCore ByteTrack/Kalman，以及消费
调用方人脸框的 MediaPipe 468 点 landmarker。GPU 路径接受 Host/CUDA 图片，每帧只 stage
一次并复用同一份 CUDA image；它不会隐式降级到 CPU。目标、模型 I/O、engine 转换、容量、
所有权和分阶段耗时契约见 [vision_models/README.md](vision_models/README.md)。

When building from this repository, all production modules and install rules are enabled. The only project options are:

- `BUILD_TESTS`: build local tests and installed-package consumer checks.
- `BUILD_EXAMPLES`: build demos and example programs.

Machine-specific dependency and test-asset paths belong in `CMakeUserPresets.json`.

The full repository depends on TurboNet and vcpkg packages such as SIMDe, STC,
xxHash, stb, and libjpeg-turbo. For a small embedded integration, use only the
needed `kalman/` and `vendor/miniblas/` sources and avoid the perception,
tracking, and serial layers.

### Source Integration

If you need to embed only the numerical core into a small C project, copy the
specific source files from `kalman/` and `vendor/miniblas/` instead of pulling
the whole spatial stack.

**How to add the KFCore Takasu formulation to your project**
   - Add `kalman/kalman_takasu.c` and `kalman/kalman_takasu.h`.
   - Add the required linear algebra backend from `vendor/miniblas/`.
   - Include the header file in your code:

```c
    #include "kalman_takasu.h"
```

**How to add the KFCore UDU formulation to your project**
   - Add `kalman/kalman_udu.c` and `kalman/kalman_udu.h`.
   - Add the required linear algebra backend from `vendor/miniblas/`.
   - Include the header file in your code:

```c
    #include "kalman_udu.h"
```

**How to add nonlinear EKF/UKF helpers to your project**
   - Add `kalman/kalman_ekf.c/.h` for EKF helpers.
   - Add `kalman/kalman_ukf.c/.h` for covariance-form UKF helpers.
   - These helpers reuse the C Kalman and linear algebra routines above.

```c
    #include "kalman_ekf.h"
    #include "kalman_ukf.h"
```

**How to add signal filters to your project**
   - Add `kalman/signal_filters.c/.h` for scalar/vector low-pass, high-pass,
     moving average, median, limit/deadband/slew-rate, Hampel, FIR, generic
     IIR, biquad IIR, cascaded SOS, moving RMS, running variance/stddev,
     trimmed mean, winsorized mean, Savitzky-Golay, moving min/max, running
     quantiles, exponentially weighted statistics, MAD noise estimation,
     Z-score/IQR gates, Huber/Tukey weights, majority voting, Schmitt triggers,
     edge detection, sample-and-hold, dropout decay, peak hold, DC blocker,
     adaptive EMA, One Euro, alpha-beta, alpha-beta-gamma, complementary,
     hysteresis, debounce, Euclidean distance, and Mahalanobis filtering
     helpers.

```c
    #include "signal_filters.h"
```

## Usage examples

### Quick Start Takasu Filter in C

For a 4x1 state vector and a 3x1 measurement vector an example setup is shown below.
The measurement sensitivity matrix $\mathbf{H}$ is a 3x4 matrix.
Two important things to highlight:

- The column-major format is used (LAPACK/BLAS default)
- The matrix $\mathbf{H}$ is supplied in a transposed way (this makes the implementation more efficient)


```c
        float x[4]    = { 1, 1, 1, 1 }; // 4x1 State vector
        float P[4*4]  = { 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f, 0, 0, 0, 0, 0.04f }; // 4x4 Covariance matrix of state vector
        float R[3*3]  = { 0.25f, 0, 0, 0, 0.25f, 0, 0, 0, 0.25f }; // 3x3 Covariance matrix of measurement
        float dz[3]   = { 0.2688f, 0.9169f, -1.1294f }; // 3x1 Measurement residuals
        float Ht[4*3] = { 8, 1, 6, 1, 3, 5, 7, 2, 4, 9, 2, 3 }; // 4x3 Transposed measurement sensitivity matrix H'
        int result    = kalman_takasu(x, P, dz, R, Ht, 4, 3, 0.0f, NULL); // Call to update routine

```

### Quick Start UDU Filter in C

First a 4x1 state vector `x` and an initial 4x4 covariance matrix `P` of the
state vector is needed, as an example:


```c
        float x[4]     = { 1.0f, 0.0f, 0.0f, 0.0f };
        float P[4 * 4] = { 0.5f, 0, 0, 0, 0, 0.5f, 0, 0, 0, 0, 0.5f, 0, 0, 0, 0, 0.5f };
```

As the UDU filter can be called a square root filter, the matrix `P` needs to
be decomposed first:


 ```c
        float U[4 * 4]; // Upper triangular matrix
        float d[4]; // 4x1 diagonal vector
```

The `udu` function will perform the decomposition of `P` into `U` and `d`:


```c
        udu(P, U, d, 4);
```

`d` is a vector that describes the diagonal matrix of the UDU decomposition.
After this, the `P` matrix is no longer required.

Then let's assume a measurement vector `z` with a covariance matrix `R` that
describes the uncertainty of the measurements. Note that `R` is not purely a
diagonal matrix so it describes correlations between the measurements in `z`


```c
        const float z[3]      = { 16.2688f, 17.9169f, 16.8706f };
        const float R[3 * 3]  = { 0.25f, 0.25f, 0.0f, 0.25f, 0.5f, 0.1f, 0.0f, 0.1f, 0.5f };
```

Then we need a measurement sensitivity matrix `H` but we store it in a
transposed form, that's why it is named `Ht` for transposed. Note that the
column-major form is used in this library.


```c
        const float Ht[4 * 3] = { 8, 1, 6, 1, 3, 5, 7, 2, 4, 9, 2, 3 };
```

The UDU filter can only process one scalar measurement at a time, that's why
we first need to decorrelate the measurements:


```c
        decorrelate(z, Ht, R, 4, 3);
```

After the decorrelation, the measurements have a unit variance:


```c
        float eye[3 * 3]  = { 1, 0, 0,
                              0, 1, 0,
                              0, 0, 1 };
```

Now we can finally process the measurements in the Kalman filter update step:


```c
        kalman_udu(x, U, d, z, eye, Ht, 4, 3, 0.0f, 0);
```

### Quick Start Signal Filters in C

```c
        kf_signal_lowpass1f lp;
        float y;

        kf_signal_lowpass1f_init(&lp, 0.25f, 0.0f);
        kf_signal_lowpass1f_update(&lp, 4.0f, &y); /* y = 1.0f */

        float state[2] = { 0.0f, 0.0f };
        const float sample[2] = { 3.0f, 4.0f };
        float distance;

        /* Accepts if ||sample - state|| <= 6, then applies alpha = 0.5. */
        kf_signal_euclidean_filter(state, sample, 2, 6.0f, 0.5f, &distance);

        float window[5];
        kf_signal_moving_average1f ma;
        kf_signal_moving_average1f_init(&ma, window, 5);
        kf_signal_moving_average1f_update(&ma, 42.0f, &y);

        kf_signal_biquad_coeffs coeffs;
        kf_signal_biquad1f biquad;
        kf_signal_biquad_lowpass(100.0f, 10.0f, 0.70710678f, &coeffs);
        kf_signal_biquad1f_init(&biquad, &coeffs);
        kf_signal_biquad1f_update(&biquad, 1.0f, &y);
```

## Takasu Formulation

| Equation                                                                             | BLAS Function   | Description                      |
|--------------------------------------------------------------------------------------|-----------------|----------------------------------|
| $\mathbf{K} = \mathbf{D} \cdot \mathbf{S}^{-1}$                                      |                 | Kalman Gain                      |
| $\mathbf{D} = \mathbf{P}^{-} \cdot \mathbf{H}^T$                                     | `symm()`        | Symmetric Matrix Product         |
| $\mathbf{S} = \mathbf{H} \cdot \mathbf{D} + \mathbf{R}$                              | `gemm()`        | General Matrix Product           |
| $\mathbf{S} = \mathbf{U}\cdot \mathbf{U}^T$                                          | `potrf()`       | Cholesky Factorization           |
| $\mathbf{E} = \mathbf{D} \cdot \mathbf{U}^{-1}$                                      | `trsm()`        | Solving Triangular Matrix        |
| $\mathbf{K} = \mathbf{E} \cdot \left( \mathbf{U}^{-1} \right)^{T}$                   | `trsm()`        | Solving Triangular Matrix        |
| $\mathbf{x}^{+} = \mathbf{x}^{-} + \mathbf{K} \cdot \mathbf{dz}$                     | `gemv()`        | Matrix Vector Product            |
| $\mathbf{P}^{+} = \mathbf{P}^{-} - \mathbf{E} \cdot \mathbf{E}^T$                    | `syrk()`        | Symmetric Rank Update            |

A priori state vector: $\mathbf{x}^{-}$, a posteriori state vector: $\mathbf{x}^{+}$,
state covariance matrix $\mathbf{P}$,
measurement residual $\mathbf{dz}$,
Measurement sensitivity / design matrix $\mathbf{H}$ such that $\mathbf{dz} = \mathbf{z} - \mathbf{H}\cdot \mathbf{x}$ for a
measurement vector $\mathbf{z}$,
covariance matrix of measurement uncertainty $\mathbf{R}$ of $\mathbf{z}$.

## License

This project is licensed under the modified BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.


## References

1. Chang, G. (2014). *Robust Kalman filtering based on Mahalanobis distance as outlier judging criterion*. **Journal of Geodesy**, **88**(4), 391-401.
2. Carpenter, J. Russell, and Christopher N. D’souza (2018). [*Navigation Filter Best Practices*](https://ntrs.nasa.gov/api/citations/20180003657/downloads/20180003657.pdf). No. NF1676L-29886.

## Acknowledgments

- **Bierman/Thornton Algorithms**: For the work on numerically stable Kalman Filter implementations.
- **Tomoji Takasu**: Appreciation to T. Takasu (the author of RTKLIB) for the efficient formulation.

## Contact

For questions, suggestions, or support:

- **Email**: [jan@zwiener.org](mailto:jan@zwiener.org)
- **GitHub Issues**: [GitHub Issues Page](https://github.com/jnz/KFCore/issues)

------------------------------------------

