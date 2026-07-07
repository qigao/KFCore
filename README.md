# KFCore

**KFCore** is a lightweight and efficient Kalman Filter library implemented in
C. Designed for both embedded systems and research applications, KFCore offers
numerically stable algorithms with minimal dependencies and low memory usage.
By leveraging advanced formulations and optimized computations, KFCore provides
a robust solution for state estimation in various projects.

## Quick check

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

## Features

- **High Numerical Stability**
  - Implements the **UDU** (Bierman/Thornton) algorithms for superior numerical stability compared to the standard Kalman Filter formulations [(2)](https://ntrs.nasa.gov/api/citations/20180003657/downloads/20180003657.pdf).
  - Includes the **Takasu formulation**, a fast and efficient implementation if you don't want a square root formulation such as the `UDU` filter.
  - Provides **EKF** and covariance-form **UKF** helpers for nonlinear process and measurement models.
  - Provides lightweight **signal filters** for scalar/vector low-pass,
    high-pass, moving average, median, clamp, deadband, slew-rate limiting,
    rate limiting, acceleration limiting, jerk limiting, Hampel outlier
    filtering, trimmed mean, winsorized mean, Savitzky-Golay smoothing, moving
    RMS, moving min/max, running quantiles, exponentially weighted statistics,
    MAD noise estimation, variance/stddev, FIR, generic IIR, biquad IIR,
    cascaded SOS, DC blocking, peak hold, adaptive EMA, One Euro adaptive
    filtering, alpha-beta and alpha-beta-gamma tracking, complementary
    filtering, hysteresis, debounce, majority voting, Schmitt triggers, edge
    detection, sample-and-hold, dropout decay, Z-score/IQR gates, Huber/Tukey
    robust weights, Euclidean norm/distance, Euclidean distance-gated updates,
    and Mahalanobis gates.
  - Provides lightweight **navigation/IMU helpers** for gyro Euler integration,
    quaternion attitude math/integration, Euler conversion output,
    roll/pitch/yaw complementary filtering, accelerometer gravity gating,
    body-frame gravity removal, and static/ZUPT detection.
  - Includes a lightweight **2D fusion system layer** for IMU prediction and
    position, navigation-frame velocity, body-frame optical/visual velocity,
    yaw, and ZUPT updates.
  - Includes a lightweight **3D quaternion error-state fusion layer** with
    nominal `p, v, q, ba, bg` state and `dp, dv, dtheta, dba, dbg` covariance
    for IMU prediction, position/velocity, attitude, and ZUPT updates.

- **Focus on Embedded Targets**
  - Uses only **static memory allocation**, ensuring guaranteed runtime and memory usage suitable for resource-constrained environments.
  - No dynamic memory allocation, making it ideal for embedded systems.

- **No External Dependencies**
  - Written in plain **C code** with no external code dependencies.
  - Easy integration into any project without the need for additional libraries.

- **Mathematical Optimizations**
  - Leverages the symmetry and positive semi-definiteness of covariance matrices.
  - Taking advantage of triangular shaped matrices.

- **Robust**
  - Added functionality to detect measurement errors and reject them with
  a Χ² statistical test.
  - Option to reduce the influence of potential outliers based on the Mahalanobis
    distance (Chang, 2014).

- **Optional: Optimized Computations with BLAS Interface**
  - Utilizes a **BLAS interface** to take advantage of optimized BLAS libraries on the target platform.
  - Provides a small built-in **miniblas** library with SIMDE-backed BLAS1
    vector routines, BLAS2 matrix-vector routines, BLAS3 matrix routines,
    vector statistics/distances, and small fixed-size matrix helpers.
  - Exposes project-friendly wrappers for the same helpers through `linalg.h`.

## Supported Filters

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

KFCore is evolving toward a robotics, drone, and AR/VR state-estimation and
perception foundation. The core idea is to combine camera, AprilTag, depth,
LiDAR, IMU, odometry, RTK/GNSS, and other sensor observations with lightweight
filters, gates, and fusion layers.

- **Robotics**
  - AprilTag pose, visual tracking, IMU fusion, wheel odometry, LiDAR/depth
    terrain detection, static/ZUPT detection, and traversability estimation.

- **AR/VR/MR**
  - Marker tracking, camera pose smoothing, IMU + visual pose fusion,
    plane/ground detection, jitter reduction, and spatial interaction support.
  - Low-cost inside-out tracking with AprilTag anchors for room-scale
    calibration, drift correction, relocalization, and rigid-body prop/tool
    simulation when OptiTrack-class global tracking is unavailable.

- **Drones**
  - RTK/GNSS + vision + IMU + barometer/rangefinder/LiDAR fusion for robust
    localization across different environments.
  - Target state includes position, velocity, attitude quaternion, IMU bias,
    sensor confidence, and environment mode.

Different environments should use different fusion strategies:

| Environment | Primary observations | Fusion strategy |
|-------------|----------------------|-----------------|
| Open sky | RTK/GNSS, IMU, barometer | RTK/GNSS position anchors with high-rate IMU prediction; gate GNSS jumps and smooth altitude. |
| Urban / multipath | IMU, vision/VIO, GNSS, LiDAR/depth | Down-weight or reject unstable GNSS; rely more on vision, inertial prediction, and local geometry. |
| Indoor / GNSS-denied | Vision/VIO, AprilTag, IMU, LiDAR/depth, rangefinder | Vision or marker pose provides position correction; IMU predicts between frames; range/depth stabilizes height and obstacles. |
| Landing / docking | AprilTag/marker, optical flow, IMU, rangefinder | Use marker-relative pose and range for high-precision low-speed control; apply strict outlier gates. |
| Rough terrain / ground robots | IMU, wheel odometry, LiDAR/depth, camera terrain cues | Fuse odometry and IMU with terrain slope/roughness; detect slip, impact, steps, and traversability changes. |

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
  camera / AprilTag / depth / LiDAR / IMU / odometry / RTK-GNSS

preprocessing:
  signal filters / feature extraction / outlier gates / confidence estimates

fusion:
  Kalman / EKF / UKF / navigation fusion / environment-specific update policy

output:
  pose / velocity / attitude / terrain state / tracking state / confidence
```

### Spatial State Engine

The consolidated direction is a shared spatial state engine for robotics,
drones, games, AR/VR, and virtual production. The engine should fuse
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
  drones / terrain regions / interaction events / confidence

outputs:
  robot navigation state / drone localization state / game state /
  cinematic state / render and compositing metadata
```

Game output prioritizes low latency, stable predicted poses, interaction
events, region triggers, and tracking confidence. Cinematic output prioritizes
virtual camera pose, actor root pose, skeleton or mesh state, rigid prop pose,
person masks, occlusion regions, render passes, and AI-video conditioning data.

### Low-Cost VR and Interaction Requirements

The practical low-cost direction is not a direct replacement for OptiTrack.
Instead, fixed commodity cameras, inside-out tracking, AprilTag anchors, IMU
prediction, and KF/EKF/UKF fusion should cooperate to produce stable game,
robot, or cinematic state.

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
  - Use inside-out tracking for continuous head, controller, device, or prop
    motion.
  - Maintain `vio_pose_in_vio_map` and an alignment from the VIO map into the
    global `map/world` frame.
  - Correct drift when AprilTag or fixed-camera observations are available.

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
  - Export stable `player_pose`, `prop_pose`, interaction events, region
    triggers, confidence, tracking-lost/recovered events, and predicted render
    poses.
  - The render loop should consume filtered and predicted state, not raw
    low-rate camera observations directly.

- **Virtual production output**
  - Export `virtual_camera_pose`, `actor_root_pose`, `actor_skeleton`,
    `prop_pose`, `person_mask`, `occlusion_region`, interaction events, and
    confidence.
  - Use Blender, Unreal, or Unity to keep camera paths, rigid props,
    collisions, lights, masks, depth, and render passes deterministic.
  - Use AI video models as a downstream cinematic rendering or style layer,
    not as the primary source of physical tracking truth.

Initial implementation modules should be:

```text
frame_transform:
  common map/camera/tag/body/imu/game coordinate conversion

map_landmark_db:
  tag and semantic anchor storage with pose, covariance, and metadata

fixed_camera_observation:
  camera extrinsics + AprilTag/person/object detections -> map-frame observations

object_pose_filter:
  filtered prop/player/object state with confidence and dropout handling

insideout_tag_relocalization:
  VIO map alignment and drift correction from known anchors

game_state_output:
  stable poses, events, confidence, and predicted render state

person_interaction_tracker:
  person mask, region, skeleton/root state, contact, sit/stand, and motion events

cinematic_state_output:
  virtual camera, actor, prop, mask, occlusion, confidence, and render metadata
```

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

### C Version

- Copy `linalg.c` and `linalg.h` from the `c/` directory to your project.
- Copy `miniblas.c` and `miniblas.h` from the `c/` directory to your project.
- If your platform has an optimized BLAS library that you want to use, you
  can exclude miniblas.

**How to add the KFCore Takasu formulation to your project**
   - Add the files `kalman_takasu.c` and `kalman_takasu.h` from the `c/`
     directory to your project.
   - Include the header file in your code:

```c
    #include "kalman_takasu.h"
```

**How to add the KFCore UDU formulation to your project**
   - Add the files `kalman_udu.c` and `kalman_udu.h` from the `c/`
     directory to your project.
   - Include the header file in your code:

```c
    #include "kalman_udu.h"
```

**How to add nonlinear EKF/UKF helpers to your project**
   - Add `kalman_ekf.c/.h` for EKF helpers.
   - Add `kalman_ukf.c/.h` for covariance-form UKF helpers.
   - These helpers reuse the C Kalman and linear algebra routines above.

```c
    #include "kalman_ekf.h"
    #include "kalman_ukf.h"
```

**How to add signal filters to your project**
   - Add `signal_filters.c/.h` for scalar/vector low-pass, high-pass, moving
     average, median, limit/deadband/slew-rate, Hampel, FIR, generic IIR,
     biquad IIR, cascaded SOS, moving RMS, running variance/stddev, trimmed
     mean, winsorized mean, Savitzky-Golay, moving min/max, running quantiles,
     exponentially weighted statistics, MAD noise estimation, Z-score/IQR
     gates, Huber/Tukey weights, majority voting, Schmitt triggers, edge
     detection, sample-and-hold, dropout decay, peak hold, DC blocker, adaptive
     EMA, One Euro, alpha-beta, alpha-beta-gamma, complementary, hysteresis,
     debounce, Euclidean distance, and Mahalanobis filtering helpers.

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

