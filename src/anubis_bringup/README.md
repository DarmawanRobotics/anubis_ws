# anubis_bringup

Top-level orchestration for Anubis. tmux-based scripts that bring up
every subsystem package together, for two operating modes. Runs
**bare-metal on the Jetson** -- no Docker at runtime (Docker was only
used for development, on a separate devbox).

See the full PDF operating guide for a complete walkthrough. This
README covers just the scripts themselves.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/bringup_mapping.sh` | Opens a 6-window tmux session to record a new map: description, sensors, perception, control, mapping, + a free shell. |
| `scripts/bringup_autonomous.sh` | Opens a 7-window tmux session for autonomous operation against an existing map: description, sensors, perception, control, localization, navigation, + a free shell. Refuses to start with a clear error if no map is found. |
| `scripts/stop_all.sh` | Sends Ctrl-C to every window, waits 3s, kills the session. Prefer this over closing the terminal directly -- gives `anubis_control` time to stop the robot cleanly. |
| `scripts/estop.sh` | **Immediate** emergency stop -- calls `/control_node/emergency_stop` directly, does not touch tmux, does not wait. Use when something is already going wrong; use `stop_all.sh` for normal shutdown. |
| `scripts/set_map.sh <name>` | Persistently changes the active map (edits `config/anubis.env`). Run with no argument to just print the current active map. |
| `scripts/_common.sh` | Shared helpers (not run directly). |

All scripts source `config/anubis.env` automatically.

## Configuration: `config/anubis.env`

The single file to edit before first use. Covers:
- `ANUBIS_WS_DIR` -- where the workspace actually lives on this machine
- `ANUBIS_MAP_NAME` / `ANUBIS_MAP_DIR` -- which map is active
- `ANUBIS_DOG_IP` -- the robot's SDK target IP
- `ANUBIS_APRILTAG_SIZE_M` / `ANUBIS_APRILTAG_FAMILY` -- **reference only**;
  the value that actually takes effect is
  `anubis_perception/config/apriltag_params.yaml`'s own `size:` field.
  Edit that file directly, then update this variable to match so it
  stays a useful reminder.
- `ANUBIS_TMUX_SESSION` -- session name every script uses

## Verified

Unlike most other packages in this workspace, this one doesn't compile
anything -- the risk is entirely in the bash/tmux logic, which **was
actually executed** in a sandbox (not just syntax-checked):
- Window creation/naming/command-dispatch (`_anubis_new_window`):
  created a real 6-window tmux session, confirmed every window's
  command actually ran (captured each pane's output).
- `stop_all.sh`'s Ctrl-C-then-kill sequence: confirmed the session is
  actually gone afterward.
- `set_map.sh`'s persistent edit: confirmed the `sed` correctly updates
  `anubis.env` and that the file still sources cleanly afterward, with
  `ANUBIS_MAP_DIR` correctly re-deriving from the new name.

What's **not** verified: the actual `ros2 launch ...` commands inside
each window, since no ROS2 install exists in the sandbox this was
built in. First real bringup on the robot is the actual test of those.

## `abandoned_detection_node.py` (person / bag / abandoned-object detection)

A launch integration point exists in `anubis_perception/launch/apriltag.launch.py`:
a `Node()` entry (`executable="abandoned_detection_node.py"`), gated behind
the `enable_abandoned_detection` launch argument (default `false`), so it
has zero effect on normal operation unless explicitly turned on.

Per the operator: the node script itself has been written separately
and this integration is considered complete on the robot's actual
checkout -- it was not re-verified in this session (no access to the
node's own source, topics, or parameters). Before relying on it:
- Confirm `abandoned_detection_node.py` is actually installed as an
  executable anubis_perception can find (`ros2 pkg executables anubis_perception`
  should list it).
- Set `enable_abandoned_detection:=true` when launching
  `anubis_perception apriltag.launch.py` (directly, or by adding it to
  `bringup_mapping.sh`/`bringup_autonomous.sh` if it should run by
  default during normal operation).
- Document its actual output topic/message type here once confirmed --
  not guessed ahead of time.
