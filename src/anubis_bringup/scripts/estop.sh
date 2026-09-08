#!/bin/bash
# estop.sh -- IMMEDIATE emergency stop. Calls anubis_control's
# emergency_stop service directly; does not touch tmux at all, so it
# works from any sourced terminal (even outside the tmux session) and
# does not wait 3 seconds like stop_all.sh's graceful Ctrl-C sequence.
#
# Use this when something is already going wrong and you need the
# robot to stop NOW. Use stop_all.sh for normal end-of-session shutdown
# (it also lets other nodes -- mapping, localization -- close down
# cleanly, which this script does not attempt).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${ROS_DISTRO:-}" ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi
if ! ros2 pkg prefix anubis_control >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/../config/anubis.env"
    # shellcheck disable=SC1091
    source "$ANUBIS_WS_DIR/install/setup.bash"
fi

echo "Sending emergency stop to /control_node/emergency_stop ..."
ros2 service call /control_node/emergency_stop std_srvs/srv/Trigger "{}"
