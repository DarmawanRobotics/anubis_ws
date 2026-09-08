#!/bin/bash
# stop_all.sh -- cleanly stops the Anubis tmux session (Ctrl-C to every
# window, then kills the session). Prefer this over closing the
# terminal window directly, so anubis_control's destroy_node() gets a
# chance to run (stops the robot + calls passive() on the SDK).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

if ! tmux has-session -t "$ANUBIS_TMUX_SESSION" 2>/dev/null; then
    echo "No '$ANUBIS_TMUX_SESSION' tmux session is running."
    exit 0
fi

echo "Sending Ctrl-C to every window in '$ANUBIS_TMUX_SESSION'..."
for pane in $(tmux list-panes -s -t "$ANUBIS_TMUX_SESSION" -F '#{pane_id}'); do
    tmux send-keys -t "$pane" C-c
done

echo "Waiting 3s for clean node shutdown..."
sleep 3

tmux kill-session -t "$ANUBIS_TMUX_SESSION"
echo "Session stopped."
