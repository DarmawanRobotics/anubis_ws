#!/bin/bash
# bringup_mapping.sh -- opens a tmux session with everything needed to
# record a new map: description, sensors, perception (AprilTag),
# control, and anubis_mapping itself.
#
# Usage:
#   ./bringup_mapping.sh                 # uses ANUBIS_MAP_DIR from anubis.env
#   ANUBIS_MAP_NAME=warehouse_v2 ./bringup_mapping.sh   # new named map
#
# This does NOT drive the robot for you -- once windows are up, drive
# with the physical remote (Role::ROLE_REMOTE on the SDK side; this
# workspace's ROS nodes never fight the remote for control) and use
# window 4 (mapping) to start/save the session:
#   ros2 service call /slam_state_service anubis_interfaces/srv/MapState "{data: 3}"   # start
#   ros2 service call /slam_state_service anubis_interfaces/srv/MapState "{data: 5}"   # save

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

_anubis_check_prereqs
_anubis_kill_existing_session

mkdir -p "$ANUBIS_MAP_DIR"
echo "Map will be saved to: $ANUBIS_MAP_DIR"

tmux new-session -d -s "$ANUBIS_TMUX_SESSION"

_anubis_new_window 0 "description" \
    "ros2 launch anubis_description description.launch.py"

sleep 1
_anubis_new_window 1 "sensors" \
    "ros2 launch anubis_sensors all_sensors.launch.py"

sleep 2
_anubis_new_window 2 "perception" \
    "ros2 launch anubis_perception apriltag.launch.py"

_anubis_new_window 3 "control" \
    "ros2 launch anubis_control control.launch.py dog_ip:=$ANUBIS_DOG_IP require_localization_valid:=false"
    # ^ anubis_localization isn't running yet during mapping (there's no
    # map to localize against), so /localization/valid never publishes.
    # Without this override, anubis_control's fail-closed default would
    # permanently block any /cmd_vel -- this override just keeps
    # ROS-based movement (e.g. `ros2 topic pub /cmd_vel ...` for testing)
    # available as an OPTION; you can still drive entirely with the
    # physical remote instead, which never goes through this at all.

sleep 1
_anubis_new_window 4 "mapping" \
    "ros2 launch anubis_mapping slam.launch.py map_output_dir:=$ANUBIS_MAP_DIR"

_anubis_new_window 5 "shell" "bash"

tmux select-window -t "$ANUBIS_TMUX_SESSION:4"
echo ""
echo "Session '$ANUBIS_TMUX_SESSION' started. Attaching..."
echo "(Ctrl-b d to detach without stopping anything; ./stop_all.sh to shut down)"
sleep 1
tmux attach -t "$ANUBIS_TMUX_SESSION"
