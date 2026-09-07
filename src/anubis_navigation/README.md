# anubis_navigation

Nav2 configuration for Anubis. Contains **no Nav2 source code of its
own** -- just a tuned `params.yaml` and a launch file that includes
standard, apt-installed Nav2 unmodified.

```bash
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
```

## Why no forked Nav2 source

darmawan_ws had 14 Nav2 packages, each a renamed fork of the real thing:

```
navigo_behaviors  navigo_behavior_tree  navigo_bt_navigator
navigo_collision_monitor  navigo_core  navigo_costmap_2d
navigo_map_server  navigo_mppi_controller  navigo_navfn_planner
navigo_path_controller  navigo_path_planner  navigo_util
navigo_velocity_optimizer  navigo_waypoint_follower
```

Every other package ported into `anubis_ws` so far (`anubis_mapping`,
`anubis_localization`) has consistently used a `[PATCH]`/dated-comment
convention to mark every real change from upstream. **Zero files across
all 14 of these packages (348 files total) carry that marker.** They are
unmodified `nav2_*` source, renamed to `navigo_*`. There is nothing to
preserve by porting them -- using real, current, apt-maintained Nav2
directly is strictly better: less to build, less to keep patched against
security/bugfix updates, and zero behavioral difference.

The one genuinely valuable file was `robot_navigo/params/navigo_params.yaml`
-- real tuning work (footprint matching the physical ZSL-1W outline,
self-filter bounds derived from observed self-collision points, dated
MPPI/inflation tuning history). That file is `config/nav2_params.yaml`
here, with `navigo_*` plugin class names restored to their real `nav2_*`
identity (a safe, distinctive-prefix rename -- unlike renaming the word
"localization" in `anubis_localization`, "navigo_" cannot collide with
unrelated text) and comments translated to English.

## `robot_navigo`'s custom nodes: also not ported

| File | Disposition |
|---|---|
| `dog_sdk_bridge_node.py`, `dog_sdk_bridge.launch.py`, `vel_cmd_udp_publisher.cpp`, `vel_cmd_lcm_publisher.cpp`, `vel_with_mc_trajectory_cmd_udp.cpp`, `odom_communication_node*.cpp`, `udp/highlevel_connector.cpp` | Superseded -- the old ZSL-1 SDK / UDP-LCM relay backend. `anubis_control` talks to `genisom_l1_sdk` directly. |
| `mode_status_publisher.cpp` | Superseded -- its whole job (gate motion on `/cmd_vel` freshness) is already `anubis_control`'s internal watchdog logic. |
| `tf_publisher.cpp` (Odometry -> TF bridge) | Superseded -- `anubis_control` already publishes `odom->base_link` TF directly from the SDK. *(Open question worth revisiting separately: this file gated the TF on `/localization/valid`; `anubis_control` currently does not. Not addressed here -- flagging so it isn't silently forgotten.)* |
| `static_map_publisher.cpp` | Hand-rolled reimplementation of `nav2_map_server`. Use the real one. |
| `costmap_listener.cpp` | Debug tool that prints costmaps as ASCII to console. RViz already does this better. |
| `obstacle_detector.cpp` | Needs a `/scan` (2D LaserScan) topic this workspace's LiDAR pipeline doesn't produce; naive line-fitting; output feeds nothing (visualization only). Nav2's own costmap obstacle layer is the real mechanism. |
| `custom_odom_baselink.cpp` | A fake odometry generator for testing (hardcoded constant velocity, no sensor input). Dev stub. |
| `mutiple_goal_nav.cpp` | Demo script with two hardcoded goal poses. The pattern (multi-goal sender) is fine but not reusable as shipped -- write fresh if wanted. |
| `send_nav_goal.py` | CLI for the custom topic-based `StartNavigation` protocol -- deferred along with that message type when scoping `anubis_interfaces`. Standard Nav2 actions (`NavigateToPose`/`NavigateThroughPoses`, via `ros2 action send_goal`) already cover this. |
| `get_optimized_path_client.py`, `pp_fem_get_optimized_path.py` | Offline analysis/plotting scripts tied to a custom service (`GetOptimizedPath`) that was never in scope for `anubis_interfaces` (0 usages when that package was scoped). One hardcodes the old `jszr_workspace` folder name. |
| `emap_publisher.py` | Test publisher for `ElectronicMapLayer` -- a costmap layer that, per `darmawan_ws`'s own history, was never actually implemented. Nothing to port. |

## What changed in `config/nav2_params.yaml`

- `navigo_*` plugin class names -> real `nav2_*` (e.g.
  `navigo_mppi_controller::MPPIController` -> `nav2_mppi_controller::MPPIController`).
- LiDAR observation topic: `/livox/lidar` -> `/sensors/lidar/points`
  (matches `anubis_sensors`' remap).
- Comments translated from Mandarin to English.
- **Flagged, not silently changed**: `obstacle_min_range` (0.10m) and the
  self-filter bounds (`self_filter_min_x/max_x/min_y/max_y/min_z/max_z`)
  were reasoned from this robot's OLD LiDAR extrinsic
  (`x=+0.220m`) and two specific observed self-collision points under
  that mount. The real mount
  (`anubis_description/urdf/sensors.xacro`, `x=+0.13011m`) is different
  -- the robot's own legs/body will land at different points in the
  LiDAR frame now. **Re-verify these against a live point cloud on the
  real robot before trusting them**; see the inline `[PATCH -- verify...]`
  comments at both spots.

Footprint (`[[0.305,0.185],[0.305,-0.185],[-0.305,-0.185],[-0.305,0.185]]`,
610x370mm) reflects real physical hardware and needed no changes.

## Build

```bash
cd ~/anubis_ws
colcon build --packages-select anubis_navigation
```

No compilation -- this is a config/launch-only package (`CMakeLists.txt`
just installs `config/` and `launch/`).

## Run

Requires `anubis_description`, `anubis_sensors`, `anubis_control`, and
`anubis_localization` already running.

```bash
ros2 launch anubis_navigation navigation.launch.py use_rviz:=true
```

RViz here is `nav2_bringup`'s own default view
(`rviz_launch.py`/`nav2_default_view.rviz`) rather than a custom config
-- unlike `anubis_mapping`/`anubis_localization`, there was no existing
working `.rviz` file for this package to adapt, and Nav2's own default
view already shows exactly what's needed (costmaps, path, goal-pose
tool, footprint).

To send a goal without RViz:
```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.0}, orientation: {w: 1.0}}}}"
```

## Known gaps

- `obstacle_min_range` and the self-filter bounds need re-verification
  against the real LiDAR mount (see above) -- not yet field-tested.
- `tf_publisher.cpp`'s `/localization/valid` gate on `odom->base_link`
  TF was not carried into `anubis_control` -- worth a deliberate decision
  later, not addressed by this package.
- Not compiled/tested in a sandbox this round (nothing to compile --
  the real risk surface is entirely in whether `nav2_params.yaml`'s
  values behave well on the real robot, which only a live test can show).
