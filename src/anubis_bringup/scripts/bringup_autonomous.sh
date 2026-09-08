#!/bin/bash
# bringup_autonomous.sh -- opens a tmux session for full autonomous
# operation against an EXISTING map: description, sensors, perception
# (AprilTag), control, localization, and navigation (Nav2).
#
# Usage:
#   ./bringup_autonomous.sh                        # uses ANUBIS_MAP_DIR from anubis.env
#   ANUBIS_MAP_NAME=warehouse_v2 ./bringup_autonomous.sh   # a different saved map
#
# Requires a map already produced by bringup_mapping.sh -- fails fast
# with a clear message if map.pcd/map.yaml aren't found.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

_anubis_check_prereqs

if [ ! -f "$ANUBIS_MAP_PCD" ]; then
    echo "ERROR: No map.pcd at $ANUBIS_MAP_PCD" >&2
    echo "       Record a map first: ./bringup_mapping.sh" >&2
    echo "       Or point at an existing one: ANUBIS_MAP_NAME=<name> $0" >&2
    exit 1
fi
if [ ! -f "$ANUBIS_MAP_YAML" ]; then
    echo "WARNING: No map.yaml at $ANUBIS_MAP_YAML" >&2
    echo "         Nav2's costmaps will run without a static /map topic." >&2
    echo "         Generate one: ros2 run anubis_mapping pcd2grid_node ..." >&2
fi

_anubis_kill_existing_session

echo "Using map: $ANUBIS_MAP_DIR"

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
    "ros2 launch anubis_control control.launch.py dog_ip:=$ANUBIS_DOG_IP"

sleep 1
_anubis_new_window 4 "localization" \
    "ros2 launch anubis_localization localization.launch.py pcd_map_path:=$ANUBIS_MAP_PCD"

sleep 2
if [ -f "$ANUBIS_MAP_YAML" ]; then
    _anubis_new_window 5 "navigation" \
        "ros2 launch anubis_navigation navigation.launch.py map:=$ANUBIS_MAP_YAML"
else
    _anubis_new_window 5 "navigation" \
        "ros2 launch anubis_navigation navigation.launch.py"
fi

_anubis_new_window 6 "shell" "bash"

tmux select-window -t "$ANUBIS_TMUX_SESSION:6"
echo ""
echo "Session '$ANUBIS_TMUX_SESSION' started. Attaching..."
echo "(Ctrl-b d to detach without stopping anything; ./stop_all.sh to shut down)"
sleep 1
tmux attach -t "$ANUBIS_TMUX_SESSION"
