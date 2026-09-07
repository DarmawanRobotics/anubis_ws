# anubis_localization

NDT/GICP-based localization for Anubis, in the lineage of Kenji Koide's
`hdl_localization`/`ndt_omp`/`fast_gicp` (see `package.xml`'s original
maintainer field -- this really is that codebase's descendant, adapted
and extended by darmawan_ws over time, now ported here). Three
independent ways to establish or correct the robot's pose in map frame,
all converging on the same `PoseEstimator` re-anchor mechanism:

1. **Automatic global localization (GL)** -- scan-descriptor retrieval
   (via `scan_descriptor`, from `xtras/`) + coarse/fine NDT search,
   with false-lock disambiguation for repetitive geometry.
2. **Manual RViz initial pose** -- the standard `/initialpose` topic
   (2D Pose Estimate tool). Already built into the ported code; when GL
   fails or times out, the node publishes `/localization/request_initialpose`
   asking an operator to click one.
3. **AprilTag anchor** -- on a validated tag sighting (from
   `apriltag_anchors.yaml`, written by `anubis_mapping` at map-save
   time), computes the robot's pose directly via a TF lookup and
   re-anchors the estimator -- works as a cold-start bootstrap (not just
   an ongoing correction), for areas where GL struggles.

None of these three had to be invented -- they were already present in
darmawan_ws's `localization_nodelet.cpp`. Method 3 needed the same
ArUco -> AprilTag rewrite as `anubis_mapping`; methods 1 and 2 were
ported unchanged.

## Provenance

Built the same way as `anubis_mapping`: bulk-copy darmawan_ws's proven
code, apply the minimum edits needed, and get compile evidence at every
step rather than trusting a rewrite from memory.

| Category | What | How it got here |
|---|---|---|
| Vendored libraries | `ndt_omp`, `fast_gicp` | Bulk-copied whole packages, unmodified, unprefixed (matching `scan_descriptor`'s precedent -- genuine third-party code, not this project's own). CUDA is optional and OFF by default in `fast_gicp` (`BUILD_VGICP_CUDA`) -- the CPU path is what's used here. |
| Custom modules | `global_localization_*`, `level0_*`, `level1_*`, `retrieval_recall`, `static_imu_init`, `map_artifact_manifest`, `pose_estimator`, `delta_estimater`, `odom_system`, `pose_system`, `pose_prediction`, `tracking_health_monitor`, `sensor_data_validity`, `ndt_optimizer_diagnostics`, `point_cloud_time_diagnostics` | Bulk-copied, **zero changes** -- confirmed none of these touch ArUco/robots_dog_msgs at all (grep-verified before copying). |
| Orchestrating node | `apps/localization_nodelet.cpp` (4561 lines, class `HdlLocalizationNode`) | Bulk-copied, then surgically rewritten: ArUco -> AprilTag anchor correction, `robots_dog_msgs` -> `anubis_interfaces` (`LoadMap`, `LocalizationState`, `Localization`), `ScanMatchingStatus` message namespace updated for the package rename. Brace/paren balance verified identical before/after (458/458, 2605/2605 -- perfectly balanced, not just proportionally consistent like `anubis_mapping`'s pre-existing off-by-one). |
| Config/launch/rviz | `config/config.yaml`, `launch/localization.launch.py`, `rviz/localization_ros2.rviz` | Adapted: topics matched to `anubis_sensors`, `aruco:` section -> `apriltag:` (schema-compatible with `anubis_mapping`'s output), extrinsic values corrected to this robot's real `anubis_description/urdf/sensors.xacro` mounting position. rviz config copied unchanged (no hardcoded custom topics to fix). |

## A significant architecture change: no more duplicated extrinsic matrices

darmawan_ws's `localization.launch.py` had its **own**
`static_transform_publisher` for `base_link -> livox_frame`, hardcoded
to a different robot's mounting values -- a second, independent copy of
a transform already duplicated in two config.yaml files. This is
exactly the class of bug the whole `anubis_ws` rebuild exists to
eliminate. That node has been **removed entirely**: `anubis_description`
already publishes this transform from the URDF, and is expected to be
running (`ros2 launch anubis_description description.launch.py`)
alongside this package.

Likewise, the old `aruco.base_to_camera_T` extrinsic parameter (a
hand-maintained 4x4 matrix) is gone -- `anubis_description`'s full
`base_link -> ... -> camera_color_optical_frame` chain plus a single TF
lookup (`base_link -> "tag<family>:<id>"`) gives the same information
with nothing left to keep in sync by hand.

`test/test_extrinsic_config_consistency.py` was rewritten to match:
instead of checking three duplicated copies against each other (the old
two config.yaml files *and* the now-removed static TF), it checks the
two remaining config.yaml copies (`anubis_localization`'s `init_T`,
`anubis_mapping`'s `keyframe.lidar_to_base_T`) directly against
`anubis_description/urdf/sensors.xacro` -- a strictly stronger
guarantee, since it verifies the configs match the robot's actual
physical description, not just each other. The parsing logic was run
against the real files in this repo as part of building this package
(not just syntax-checked) and confirmed all three agree:
`[0.13011, -0.02329, 0.17598]`.

## Verification

Unlike `anubis_mapping`, this package's custom modules were **not**
individually compiled in a sandbox this time (no PCL/`ndt_omp`/
`fast_gicp`/GTSAM-equivalent environment was set up for a second,
even-larger round) -- the confidence here comes from:
- Zero `robots_dog_msgs`/ArUco references remaining anywhere in the
  tree (grep-verified across every file, not just the nodelet).
- The nodelet's brace/paren counts are perfectly balanced after the
  edit (not merely unchanged from a pre-existing imbalance, as was the
  case for `anubis_mapping`).
- `CMakeLists.txt`'s target/dependency/test structure was copied
  **line-for-line** from the original (not re-derived by guessing, which
  is what cost `anubis_mapping` several rounds of missing-dependency
  colcon failures) -- only `robots_dog_msgs` -> `anubis_interfaces` and
  `apriltag_msgs` were added to the shared dependency list.
- `test_extrinsic_config_consistency.py`'s parsing logic was actually
  executed against the real shipped config files, not just written and
  assumed correct.

**A real `colcon build` is still this package's first actual compile.**
Please report the first build's output the same way as `anubis_mapping`'s
-- this package is larger (localization + two vendored libraries vs.
mapping's one), so there is more surface for a first-build surprise.

## New/changed parameters (apriltag.*)

| Parameter | Default | Notes |
|---|---|---|
| `apriltag.enable` | `true` | |
| `apriltag.detection_topic` | `/perception/apriltag/detections` | Matches `anubis_perception`. |
| `apriltag.max_hamming` | `0` | Replaces `max_reprojection_error_px`. |
| `apriltag.min_decision_margin` | `50.0` | Replaces `require_depth_validation`. |
| `apriltag.tag_family` | `"36h11"` | |
| `apriltag.base_frame_id` | `"base_link"` | TF lookup target frame; replaces `base_to_camera_T`. |
| `apriltag.max_jump_m` | `0.5` | Unchanged from `aruco.max_jump_m`. |
| `apriltag.consistency_min_hits` / `consistency_window_s` | `3` / `2.0` | Unchanged. |

## Build

```bash
cd ~/anubis_ws
colcon build --packages-select ndt_omp fast_gicp anubis_localization
colcon test --packages-select anubis_localization
colcon test-result --verbose
```

Requires, beyond `anubis_mapping`'s dependency list: `pcl_ros`,
`tf2_geometry_msgs`, `tf2_eigen`. `ndt_omp` and `fast_gicp` build as
part of this same batch (they are not published packages -- they only
exist vendored in this workspace).

## Run

```bash
# anubis_description, anubis_sensors, anubis_perception should already
# be running (this package depends on their TF/topics)
ros2 launch anubis_localization localization.launch.py pcd_map_path:=/path/to/map.pcd
```

Or via the bundled convenience script (workspace auto-detection +
timestamped logging):

```bash
./install/anubis_localization/../../run_localization.sh
# (installed alongside the workspace root, not under lib/anubis_localization)
```

To manually set an initial pose in RViz: use the "2D Pose Estimate"
tool, or wait for `/localization/request_initialpose` to fire if GL
fails first.

## Known gaps

- Not compiled in a sandbox this round (see Verification above) --
  first `colcon build` is the real test.
- `ndt_omp`/`fast_gicp` vendored at whatever commit was present in the
  uploaded `darmawan_ws` snapshot; not re-pinned to a specific upstream
  release.
- AprilTag-anchor bootstrap (method 3) has not been field-tested with
  real tag geometry yet -- the TF-lookup math mirrors `anubis_mapping`'s
  already-reasoned equivalent, but this is its first appearance in this
  package.
