#!/bin/bash
# Run this once, right after cloning a repo made from the ros2_humble_docker
# template. It creates .env for you — no other file needs to be touched
# (see .env.example for what each value controls).
#
# Usage:
#   ./scripts/init.sh                  interactive — prompts for each value
#   ./scripts/init.sh myproject        non-interactive — sets WORKSPACE_NAME,
#                                       defaults everything else, still prompts
#                                       for passwords unless -y is also passed
#   ./scripts/init.sh myproject -y     fully non-interactive, uses .env.example
#                                       defaults for anything not overridden
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then
    read -rp ".env already exists — overwrite? [y/N] " confirm
    [ "$confirm" = "y" ] || [ "$confirm" = "Y" ] || { echo "aborted"; exit 1; }
fi

WORKSPACE_NAME="$1"
YES_FLAG="$2"
[ "$1" = "-y" ] && { YES_FLAG="-y"; WORKSPACE_NAME=""; }

cp .env.example .env

# Portable in-place edit: GNU sed (Linux) accepts `sed -i 'expr' file`, but
# BSD sed (macOS) treats the next arg after -i as a backup suffix and eats
# the sed expression as if it were that suffix — silently misparsing
# everything downstream. Piping through a temp file instead of using -i
# sidesteps the GNU/BSD difference entirely.
sed_inplace() {
    local expr="$1" file="$2" tmp
    tmp="$(mktemp)"
    sed "$expr" "$file" > "$tmp" && mv "$tmp" "$file"
}

prompt_set() {
    # prompt_set VAR_NAME "question" default
    var="$1"; question="$2"; default="$3"
    if [ "$YES_FLAG" = "-y" ]; then
        value="$default"
    else
        read -rp "$question [$default]: " value
        value="${value:-$default}"
    fi
    # escape / and & for sed replacement safety
    escaped=$(printf '%s' "$value" | sed -e 's/[\/&]/\\&/g')
    sed_inplace "s/^${var}=.*/${var}=${escaped}/" .env
}

if [ -z "$WORKSPACE_NAME" ]; then
    prompt_set WORKSPACE_NAME "Project / workspace name" "ros2_ws"
else
    sed_inplace "s/^WORKSPACE_NAME=.*/WORKSPACE_NAME=${WORKSPACE_NAME}/" .env
    echo "WORKSPACE_NAME set to ${WORKSPACE_NAME}"
fi

prompt_set USERNAME          "Linux username inside the container" "devuser"
prompt_set USER_UID          "Host UID (run 'id -u' to check)"      "$(id -u 2>/dev/null || echo 1000)"
prompt_set USER_GID          "Host GID (run 'id -g' to check)"      "$(id -g 2>/dev/null || echo 1000)"
prompt_set LINUX_PASSWORD    "Linux account password"               "changeme"
prompt_set MAINTAINER_NAME   "Maintainer name (for ros2 pkg create)" "your-name"
prompt_set MAINTAINER_EMAIL  "Maintainer email"                      "you@example.com"

echo
echo "done — .env written."
if [ "$LINUX_PASSWORD" = "changeme" ]; then
    echo "note: you left a default password in place — fine for local dev, change it before exposing any port."
fi
echo "next: ./scripts/dev.sh up   (or up-linux)"