#!/bin/bash
# _common.sh -- shared helpers for anubis_bringup scripts. Not meant to
# be run directly; sourced by bringup_mapping.sh / bringup_autonomous.sh
# / stop_all.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../config/anubis.env"

_anubis_check_prereqs() {
    if [ ! -f "/opt/ros/humble/setup.bash" ]; then
        echo "ERROR: ROS2 Humble not found at /opt/ros/humble/setup.bash" >&2
        exit 1
    fi
    if [ ! -f "$ANUBIS_WS_DIR/install/setup.bash" ]; then
        echo "ERROR: Workspace not built at $ANUBIS_WS_DIR/install/setup.bash" >&2
        echo "       Run: cd $ANUBIS_WS_DIR && colcon build" >&2
        exit 1
    fi
    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: tmux not installed. Run: sudo apt install tmux" >&2
        exit 1
    fi
}

# Sourced inside each tmux pane, not in this script's own shell --
# every pane needs both environments independently.
_anubis_source_cmd() {
    echo "source /opt/ros/humble/setup.bash && source $ANUBIS_WS_DIR/install/setup.bash && source $SCRIPT_DIR/../config/anubis.env"
}

_anubis_kill_existing_session() {
    if tmux has-session -t "$ANUBIS_TMUX_SESSION" 2>/dev/null; then
        echo "An existing '$ANUBIS_TMUX_SESSION' tmux session is already running."
        read -r -p "Kill it and start fresh? [y/N] " reply
        if [[ "$reply" =~ ^[Yy]$ ]]; then
            tmux kill-session -t "$ANUBIS_TMUX_SESSION"
        else
            echo "Attaching to the existing session instead (nothing new started)."
            tmux attach -t "$ANUBIS_TMUX_SESSION"
            exit 0
        fi
    fi
}

# _anubis_new_window <index> <name> <command>
# Window 0 is created by `tmux new-session` itself; call this for 0 too
# (it renames + runs the command in the already-existing window 0).
_anubis_new_window() {
    local index="$1" name="$2" command="$3"
    if [ "$index" -eq 0 ]; then
        tmux rename-window -t "$ANUBIS_TMUX_SESSION:0" "$name"
        tmux send-keys -t "$ANUBIS_TMUX_SESSION:0" "$(_anubis_source_cmd) && $command" C-m
    else
        tmux new-window -t "$ANUBIS_TMUX_SESSION" -n "$name"
        tmux send-keys -t "$ANUBIS_TMUX_SESSION:$index" "$(_anubis_source_cmd) && $command" C-m
    fi
}
