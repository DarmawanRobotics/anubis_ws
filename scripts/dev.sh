#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

set -a
source .env 2>/dev/null
set +a

WORKSPACE_NAME="${WORKSPACE_NAME:-ros2_ws}"
CONTAINER="${WORKSPACE_NAME}_dev"
DEV_USER="${USERNAME:-devuser}"
ROS_SOURCE="source /opt/ros/humble/setup.bash; [ -f ~/${WORKSPACE_NAME}/install/setup.bash ] && source ~/${WORKSPACE_NAME}/install/setup.bash;"

usage() {
    cat << EOF
Usage: ./scripts/dev.sh <command> [args]

  up                 start container (mac / bridge network)
  up-linux           start container (linux / host network + usb)
  down               stop container (bridge)
  down-linux         stop container (host network + usb)
  ssh                ssh into the running container
  tmux               attach tmux session (workspace/ros2/mapping/monitor/logs)
  tmux-kill          kill the tmux session (stops all panes/processes inside it)
  build              rosdep install + colcon build (skips rosdep update if already cached)
  rosdep-update      force-refresh the rosdep cache (init + update)
  test [pkg]         colcon test, optionally scoped to one package
  create-pkg <name> [ament_cmake|ament_python]
                     ros2 pkg create inside src/
  run "<cmd>"        run an arbitrary command inside the container (ros2 env sourced)
  clean              remove build/install/log
  rebuild-image      full rebuild: down, build --no-cache, up (bridge)
  rebuild-image-linux  same, but keeping host network + usb passthrough
EOF
}

cmd="${1:-}"
shift || true

case "$cmd" in
  up)
    docker compose up -d --build
    echo "${WORKSPACE_NAME} up (bridge network). VNC: localhost:${VNC_PORT:-5901} · SSH: localhost:${SSH_PORT:-2222}"
    ;;
  up-linux)
    docker compose -f docker-compose.yml -f docker-compose.linux.yml up -d --build
    echo "${WORKSPACE_NAME} up (linux — host network + usb). VNC: localhost:${VNC_PORT:-5901} · SSH: localhost:${SSH_PORT:-2222}"
    ;;
  down)
    docker compose down
    ;;
  down-linux)
    docker compose -f docker-compose.yml -f docker-compose.linux.yml down
    ;;
  ssh)
    ssh -p "${SSH_PORT:-2222}" "${DEV_USER}@localhost"
    ;;
  tmux)
    docker exec -it -u "$DEV_USER" "$CONTAINER" bash -lc "tmuxp load -y ~/tmux/session.yaml"
    ;;
  tmux-kill)
    docker exec -it -u "$DEV_USER" "$CONTAINER" bash -lc "tmux kill-session -t ${WORKSPACE_NAME}"
    ;;
  build)
    docker exec -it -u "$DEV_USER" "$CONTAINER" bash -lc "$ROS_SOURCE [ -f ~/.ros/rosdep/sources.cache/index ] || { sudo rosdep init 2>/dev/null; rosdep update; }; cd ~/${WORKSPACE_NAME} && rosdep install --from-paths src --ignore-src -r -y --skip-keys=librealsense2 && colcon build --symlink-install"
    ;;
  rosdep-update)
    docker exec -it -u "$DEV_USER" "$CONTAINER" bash -lc "$ROS_SOURCE sudo rosdep init 2>/dev/null; rosdep update"
    ;;
  test)
    pkg_filter=""
    [ -n "$1" ] && pkg_filter="--packages-select $1"
    docker exec -it -u "$DEV_USER" "$CONTAINER" bash -lc "$ROS_SOURCE cd ~/${WORKSPACE_NAME} && colcon test ${pkg_filter} && colcon test-result --verbose"
    ;;
  create-pkg)
    pkg_name="${1:?usage: dev.sh create-pkg <package_name> [ament_cmake|ament_python]}"
    build_type="${2:-ament_cmake}"
    docker exec -it -u "$DEV_USER" "$CONTAINER" bash -lc "$ROS_SOURCE cd ~/${WORKSPACE_NAME}/src && ros2 pkg create --build-type ${build_type} ${pkg_name} --maintainer-name '${MAINTAINER_NAME:-your-name}' --maintainer-email '${MAINTAINER_EMAIL:-you@example.com}' --license ${LICENSE_SPDX:-MIT}"
    ;;
  run)
    docker exec -it -u "$DEV_USER" "$CONTAINER" bash -lc "$ROS_SOURCE $*"
    ;;
  clean)
    rm -rf build install log
    echo "build/install/log cleaned"
    ;;
  rebuild-image)
    docker compose down
    docker compose build --no-cache
    docker compose up -d
    ;;
  rebuild-image-linux)
    docker compose -f docker-compose.yml -f docker-compose.linux.yml down
    docker compose -f docker-compose.yml -f docker-compose.linux.yml build --no-cache
    docker compose -f docker-compose.yml -f docker-compose.linux.yml up -d
    ;;
  *)
    usage
    exit 1
    ;;
esac