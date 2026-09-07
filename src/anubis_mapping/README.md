# anubis_mapping

FAST-LIO based mapping for the Anubis robot. Produces `map.pcd`,
`map.pgm`/`map.yaml` (for Nav2), `map_scd.bin` (scan-descriptor database
for global relocalization), and `apriltag_anchors.yaml` (validated
AprilTag anchor poses in map frame, for `anubis_localization` to load).

## Provenance

This package was **not** written from scratch. It was built by
bulk-copying `darmawan_ws`'s proven, already-debugged mapping stack
module by module, applying a single project-wide namespace rename
(`robot::slam` -> `anubis_mapping`), and rewriting only the one piece
that genuinely needed new logic: the AprilTag anchor integration
(darmawan_ws used ArUco).

This is a deliberate strategy, not a shortcut taken to save effort at
the cost of quality: darmawan_ws's mapping stack represents months of
real debugging (timestamp handling, Z-drift, map leveling, loop-closure
false-locks) that would be wasteful and risky to redo from memory.
Copying it exactly, then verifying the copy with the original's own
test suite, preserves that work instead of discarding it.

## What was bulk-copied vs. rewritten

| Category | Files | How it got here |
|---|---|---|
| Vendored FAST-LIO2 + ikd-Tree + MTK core | `process/imu_process.*`, `process/lidar_process.*`, `use_ikfom.*`, `ikd_tree/*`, `common/*`, `so3_math.h`, `mtk_iekf/**` | Bulk-copied, namespace renamed. Not modified otherwise -- this is the upstream FAST-LIO2 algorithm, not darmawan_ws's own code. |
| Custom modules | `lio_time_guard`, `imu_init_gate`, `keyframe_store`, `loop_closure`, `map_z_drift_correction`, `map_leveling_sidecar`, `map_artifact_digest`, `map_artifact_transaction`, `pcd2grid` + variants | Bulk-copied, namespace renamed. **Each module's existing gtest suite was ported alongside it and run** -- see Verification below. None of these touch ArUco/AprilTag at all. |
| Orchestrating node | `mapping_alg.h`, `mapping_alg.cpp`, `mapping_node_main.cpp` (entry point) | Bulk-copied, namespace renamed, **then surgically rewritten** for the ArUco -> AprilTag migration (see below). This is the one file in the package that could not be verified by compiling in a sandbox -- see Verification. |

## The ArUco -> AprilTag migration

darmawan_ws's `arucoDetectionCallBack` read a 6-DOF pose directly off the
detection message (the ArUco node did its own PnP solve and published
`geometry_msgs/Pose`). **apriltag_ros does not do this** -- its
`AprilTagDetection` message carries only `family`, `id`, `hamming`,
`decision_margin`, pixel `centre`/`corners`, and a `homography` matrix.
The actual 3D pose is published separately, on `/tf`, with
`child_frame_id = "tag<family>:<id>"`.

This forced a real architecture change, not just a type rename:

- **Old**: `T_camera_marker` read from the message, then multiplied by a
  hand-maintained `aruco.lidar_to_camera_T` parameter (a 16-value 4x4
  matrix kept in sync with the URDF by hand -- this is exactly the class
  of bug darmawan_ws lost real time to: the same physical transform
  duplicated across `mapping`'s config, `localization`'s config, and a
  `static_transform_publisher`, silently drifting out of sync).
- **New**: `T_lidar_marker` comes from a single TF lookup --
  `tf_buffer_->lookupTransform(apriltag_lidar_frame_id_, "tag<family>:<id>", stamp)`.
  Because `anubis_description` publishes the full
  `base_link -> livox_frame` and `base_link -> camera_link -> ... ->
  camera_color_optical_frame` chain (the latter extended at runtime by
  whatever `realsense2_camera` publishes for its own optical-frame
  sub-tree), tf2 composes the *entire* chain for us. **There is no
  extrinsic-matrix parameter left to duplicate or drift.** This is a
  direct, structural fix for the recurring problem above, not just a
  side effect of switching tag families.

Quality gating changed to match what AprilTag detections actually
expose:

| Old (ArUco) | New (AprilTag) |
|---|---|
| `reprojection_error_px` (from the node's own PnP) | `hamming` (corrected bit count, 0 = best) |
| `depth_validated` (RealSense depth cross-check) | `decision_margin` (detector confidence) |

The `aruco.yaml` output format is preserved as `apriltag_anchors.yaml`
with the *same schema* (`frame_id`, `markers: [{id, position,
orientation, observation_count, max_position_deviation_m, validated}]`)
on purpose, so `anubis_localization`'s loader is a straightforward port,
not a new format to design.

### New: live debug TF during mapping

Per a request during development: every accepted AprilTag observation
now also broadcasts a live TF frame (`odom -> apriltag_debug_<id>`) at
capture time, gated by `apriltag.publish_debug_tf` (default `true`), so
markers are visible in RViz *while mapping is still running*. This is
**not** the final anchor pose -- that only exists after save-time
resolution against the loop-closure-corrected keyframe pose in
`saveAprilTagMap()`. The debug frame is a raw per-sighting pose relative
to the online (uncorrected) LIO estimate, for visual sanity-checking
only.

## Verification

Every bulk-copied module was compiled and its existing test suite run
directly with `g++` in a sandbox (no ROS2 available there), against the
real dependencies (Eigen, PCL, GTSAM, OpenSSL) at the versions available
via `apt` on Ubuntu 24.04:

| Module | Tests | Result |
|---|---|---|
| `lio_time_guard` | 13 | PASS |
| `imu_init_gate` | 10 | PASS |
| `keyframe_store` | 2 | PASS |
| `map_z_drift_correction` | 13 | PASS |
| `map_artifact_digest` | 12 | PASS |
| `map_artifact_transaction` | 7 | PASS |
| `map_leveling_sidecar` | 15 | PASS |
| `loop_closure` | 24 | PASS |
| `pcd2grid` (height/ground-plane) | 24 | PASS |
| **Total** | **120** | **PASS** |

The FAST-LIO2/ikd-Tree/MTK core files compile cleanly (`imu_process`,
`lidar_process`, `use_ikfom`, `ikd_tree`, `common/math`) but have no
dedicated unit tests of their own (they didn't in darmawan_ws either --
they're validated in that codebase's history by actual mapping runs, not
gtest).

### What was **not** verified this way

`mapping_alg.cpp`/`.h` (the orchestrating node) and `pcd2grid.cpp`/the
`pcd2grid_node` CLI tool both hard-depend on `rclcpp/rclcpp.hpp`, which
is not installable outside a real ROS2 environment. **Neither could be
compiled in the sandbox used to build this package.** A real
`colcon build` on your devbox/Jetson is the first actual compile these
two get. Please treat `mapping_alg_node` in particular as unverified
until you've built and run it -- it is by far the largest, most
structurally-changed file in this package, and the one where a mistake
would be least obvious from a code read alone.

Things worth specifically checking on first build/run:
- Does it compile at all (missing include, wrong `apriltag_msgs` field
  name, etc.)?
- Does the TF lookup actually succeed once `anubis_perception`
  (apriltag_ros) and `anubis_sensors` (realsense2_camera) are both
  running -- check for `apriltag id=%d: TF lookup ... failed` warnings
  in the log.
- Does `apriltag_anchors.yaml` get written on save, with `validated:
  true` for markers you deliberately viewed from >=3 angles?

## Parameters (apriltag.*)

| Parameter | Default | Notes |
|---|---|---|
| `apriltag.enable` | `false` | |
| `apriltag.detection_topic` | `/perception/apriltag/detections` | Matches `anubis_perception`'s remap. |
| `apriltag.max_keyframe_time_gap_s` | `1.0` | |
| `apriltag.min_observations` | `3` | |
| `apriltag.max_position_deviation_m` | `0.05` | |
| `apriltag.max_hamming` | `0` | |
| `apriltag.min_decision_margin` | `50.0` | Tune against your own lighting/distance; apriltag_ros's own defaults are a reasonable starting reference. |
| `apriltag.tag_family` | `"36h11"` | Must match `anubis_perception`'s `apriltag_params.yaml`. |
| `apriltag.lidar_frame_id` | `"livox_frame"` | |
| `apriltag.publish_debug_tf` | `true` | |

## Build

```bash
cd ~/anubis_ws
colcon build --packages-select anubis_mapping --symlink-install
colcon test --packages-select anubis_mapping
colcon test-result --verbose
```

Requires, beyond a standard ROS2 Humble desktop install:
`libeigen3-dev`, `libpcl-dev` (or equivalent PCL packages),
`libgtsam-dev`, `libssl-dev`, `libopencv-dev`, `ros-humble-apriltag`,
`ros-humble-apriltag-ros`, and `anubis_interfaces` +
`scan_descriptor` built earlier in this workspace.

## Known gaps

- `mapping_alg_node` and `pcd2grid_node`: not compile-verified (see
  Verification above).
- `scan_descriptor`'s exact exported CMake target name
  (`scan_descriptor::scan_descriptor` is assumed in
  `CMakeLists.txt`) has not been independently confirmed against your
  actual port of that package.
- `apriltag.min_decision_margin`'s default (`50.0`) is a starting point,
  not a value tuned against your specific tags/camera/lighting.
