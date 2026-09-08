#!/bin/bash
# set_map.sh -- persistently changes the active map (writes to
# config/anubis.env, not just this shell session) so both
# bringup_mapping.sh and bringup_autonomous.sh pick it up next time.
#
# Usage:
#   ./set_map.sh warehouse_v2      # sets ANUBIS_MAP_NAME=warehouse_v2
#   ./set_map.sh                   # prints the current active map, no change

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../config/anubis.env"

# shellcheck disable=SC1090
source "$ENV_FILE"

if [ "$#" -eq 0 ]; then
    echo "Current active map: $ANUBIS_MAP_NAME"
    echo "  (directory: $ANUBIS_MAP_DIR)"
    exit 0
fi

NEW_NAME="$1"
sed -i "s|^export ANUBIS_MAP_NAME=.*|export ANUBIS_MAP_NAME=\"$NEW_NAME\"|" "$ENV_FILE"

echo "Active map set to: $NEW_NAME"
echo "  (directory will be: $HOME/anubis_maps/$NEW_NAME)"
echo ""
echo "Note: this only takes effect on the NEXT bringup_mapping.sh /"
echo "bringup_autonomous.sh run (or any new shell that sources anubis.env)."
