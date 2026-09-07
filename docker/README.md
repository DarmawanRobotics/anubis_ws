# ROS2 Humble dev container

Ubuntu 22.04 + XFCE desktop + ROS2 Humble desktop-full, running inside Docker. Access via VNC (full GUI) or SSH (terminal only). tmux session pre-built from `tmux/session.yaml`.

## Prerequisites

- Docker + Docker Compose v2 (`docker compose`, not the standalone `docker-compose`)
- On Linux, USB passthrough (`up-linux`) requires the user running Docker to have access to `/dev/bus/usb`

## Setup

```bash
./scripts/init.sh
```

Interactive prompt, writes `.env` from `.env.example`: `WORKSPACE_NAME`, `USERNAME`, `USER_UID`/`USER_GID`, `LINUX_PASSWORD`,  `MAINTAINER_NAME`/`MAINTAINER_EMAIL`. Non-interactive: `./scripts/init.sh myproject -y` (fills everything else with `.env.example` defaults).

SSH defaults to password auth (`LINUX_PASSWORD`) and the host port is bound to `127.0.0.1` in bridge mode. For pubkey-only login: uncomment the `authorized_keys` volume line in `docker-compose.yml`, point it at your own `.pub` key, then set `SSH_PASSWORD_AUTH=no` in `.env`. The entrypoint checks for a mounted key first and refuses to disable password auth (falls back with a warning) if none is found, so you can't lock yourself out. Note this only fully closes password auth on the SSH side — `up-linux` (host network) still exposes the port on every host interface regardless of the `127.0.0.1` binding, since `network_mode: host` bypasses Docker's port mapping entirely.

`USER_UID`/`USER_GID` matter — they're passed as build args and used to create the in-container user, so files created inside the container (build artifacts, new packages) are owned by *you* on the host, not root. Match your host's `id -u`/`id -g`.

## Run

```bash
./scripts/dev.sh up          # mac, or linux without host networking — bridge network
./scripts/dev.sh up-linux    # linux — host network + /dev/bus/usb passthrough
```

`up-linux` is required (not optional) if you're connecting sensor hardware that uses UDP discovery over Ethernet rather than pure USB — e.g. Livox Mid-360. Bridge networking can't see that traffic; USB device mapping alone doesn't cover it.

Both commands run `docker compose build` first, so editing the Dockerfile and re-running `up` picks up the change — no separate build step needed.

## Use

```bash
./scripts/dev.sh ssh                    # ssh in on port 2222 — lands in ~/$WORKSPACE_NAME with ROS2 already sourced
./scripts/dev.sh tmux                   # attach the pre-built tmux session (see tmux/session.yaml for pane layout)
./scripts/dev.sh tmux-kill              # kill the tmux session and everything running inside it
./scripts/dev.sh build                  # rosdep install + colcon build --symlink-install
./scripts/dev.sh rosdep-update          # force-refresh the rosdep cache (build skips this once cached)
./scripts/dev.sh test [package]         # colcon test, optionally scoped to one package, then colcon test-result --verbose
./scripts/dev.sh create-pkg <name> [ament_cmake|ament_python]
                                         # ros2 pkg create in src/, using MAINTAINER_NAME/EMAIL from .env
./scripts/dev.sh run "<command>"        # run anything else inside the container, with ROS2 env sourced first
./scripts/dev.sh clean                  # rm -rf build install log
./scripts/dev.sh rebuild-image          # down, build --no-cache, up — for when the Dockerfile itself changed
./scripts/dev.sh rebuild-image-linux    # same, but keeps host network + usb passthrough (up-linux)
```

GUI apps (rqt, rviz2, rqt_graph): connect a VNC client to `localhost:5901`, password from `LINUX_PASSWORD`. Resolution/color depth are set via `VNC_RESOLUTION`/`VNC_DEPTH` in `.env`.

## Stop

```bash
./scripts/dev.sh down          # or down-linux — must match whichever `up` variant started it
```

## How the pieces fit together

- **Bind mount, not a copy**: the whole repo (including `.git`) is mounted at `/home/$USERNAME/$WORKSPACE_NAME`. Edits made on the host (your normal editor) and edits made inside the container (e.g. `ros2 pkg create`) are visible to both sides instantly — no rebuild, no `docker cp`.
- **Auto-cd + auto-source on interactive shells**: `.bashrc` sources `/opt/ros/humble/setup.bash` (and `install/setup.bash` if it exists) and `cd`s into `~/$WORKSPACE_NAME`, so `./scripts/dev.sh ssh` and `docker exec -it <container> bash` land you ready to build — no manual sourcing. This only fires for real interactive shells; `dev.sh run`/`build`/`test` source ROS explicitly via their own command string, since `bash -lc "<cmd>"` doesn't reliably trigger `.bashrc`.
- **`WORKSPACE_NAME` drives naming end-to-end**: it's passed as a Docker build arg and sed-replaced into `tmux/session.yaml` and `tmux.conf` at image build time, so the tmux session name, window name, and status bar label all match the container/image name automatically — set it once in `.env`, nothing else to edit.
- **XFCE, not GNOME**: GNOME-shell over software-rendered VNC is too heavy for RViz2 pointcloud rendering at usable framerates.
- **`rosdep` cache persists in a named volume**: `~/.ros` is mounted as the `rosdep-cache` volume (not the bind mount, not the container's writable layer), so it survives `down`/`up` *and* `rebuild-image`. `build` only runs `rosdep init`/`update` the first time (checks for `~/.ros/rosdep/sources.cache/index`) — run `rosdep-update` manually if you need fresher package data sooner. `docker compose down --volumes` (or `docker volume rm`) clears it if it ever needs a hard reset.
- **Container health check**: `docker ps`/`docker compose ps` show real health (not just "running") based on whether sshd is accepting connections on `127.0.0.1:2222` inside the container — the one service the entrypoint always tries to keep alive even if VNC/X11 fails to start.

## Troubleshooting

- **VNC connects but shows a black/gray screen**: the X session may not have started — check `docker exec -it <container> cat ~/.vnc/*.log`.
- **`create-pkg` fails with a permissions error**: `USER_UID`/`USER_GID` in `.env` don't match your host user — fix `.env` and `./scripts/dev.sh rebuild-image`.
- **Livox/RealSense not detected under `up` (non-linux variant)**: expected — USB/Ethernet device passthrough only works with `up-linux`.