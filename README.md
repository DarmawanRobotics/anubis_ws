<div align="center">

# ros2-humble-devbox

**Full ROS2 Humble dev environment in Docker — code, build, test, and run GUI tools (RViz, rqt) from any OS. No local ROS install required.**

[![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?style=for-the-badge&logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)](https://releases.ubuntu.com/22.04/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## Demo

<!-- record a GIF/MP4 of: up → ssh → build → open RViz over VNC, save it to docs/demo.gif, then re-add the embed below -->

<div align="center">

*Spin up the container, SSH into a sourced ROS2 workspace, build a package, open RViz over VNC — all from a Mac.*

</div>

---

## Why this exists

No native ROS2 on macOS, no `colcon build` without a Linux box, no RViz without dual-booting or babysitting a VM. This runs a full Ubuntu 22.04 + ROS2 Humble desktop inside Docker instead — any host OS, as long as Docker runs.

- **Create and test real packages from a Mac** — `ros2 pkg create`, `colcon build`, `colcon test` run the same as native Ubuntu, because it *is* Ubuntu, just inside the container.
- **Full GUI, not just a shell** — VNC gives you RViz2, rqt, and any other tool at full framerate. Files sync live between host and container both ways (bind mount, not a copy), so edits in your own editor show up instantly with no rebuild.
- **One script for the whole workflow** — `scripts/dev.sh up/ssh/build/test/create-pkg/tmux` covers day-to-day use, no `docker exec` incantations to memorize.

## Quickstart

```bash
git clone https://github.com/<you>/ros2-humble-devbox.git
cd ros2-humble-devbox
./scripts/init.sh      # writes .env
./scripts/dev.sh up    # or up-linux on Linux
./scripts/dev.sh ssh   # lands in ~/WORKSPACE_NAME with ROS2 sourced
```

Full setup, all commands, and troubleshooting: [`docker/README.md`](docker/README.md)

Prefer an IDE over SSH/tmux? `.devcontainer/devcontainer.json` attaches VSCode's Dev Containers extension straight to the same service — reopen the folder in a container after `./scripts/init.sh`.

## Using this as a template

Click **Use this template** on GitHub, run `./scripts/init.sh`, replace this header with your project. `WORKSPACE_NAME` in `.env` drives the container/image/tmux naming everywhere else — nothing else to edit.

## License

[MIT](LICENSE)