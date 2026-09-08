# anubis_command_center

Robot-side half of a minimal proof-of-concept: **can an image actually
cross the robot's cellular/5G link to a browser, or not?** Pushes the
camera image, battery status, and active-map file info OUT over an
outbound WebSocket connection to a relay running on a VPS. See the
sibling `anubis_vps_relay` (not a ROS package -- runs on the VPS, not
here) for the other half.

```
this node (WebSocket client) --> [VPS relay] <-- browser (viewer)
```

## Why this pushes OUT instead of serving a page directly

The robot is behind carrier/cellular NAT with no public IP -- nothing
can connect IN to it from the internet. It can only ever be the side
that connects OUT. So this node is a WebSocket **client**, not a
server: it opens one outbound connection to the VPS and keeps pushing
updates on it, reconnecting with backoff if the link drops (which is
the exact condition this whole exercise exists to test on the real 5G
connection, not a bug to avoid).

## Why the image is re-encoded here, not read from a compressed topic

`anubis_sensors` publishes raw `sensor_msgs/Image`, not
`sensor_msgs/CompressedImage` -- whether a `.../compressed` topic
exists depends on which `image_transport` plugins happen to be
enabled, which this package doesn't want to depend on. Re-encoding to
JPEG here directly (`cv_bridge` + OpenCV) works regardless of that, and
keeps quality/size under this package's own control -- `jpeg_quality`
and `max_image_width` are both launch arguments, directly relevant over
a bandwidth-constrained cellular link.

## Why `websockets` (a new pip dependency), when other packages avoid that

Every other robot-side package in this workspace deliberately avoids
new pip dependencies (bare-metal deployment, prefer `apt` covering
everything). Implementing the WebSocket client handshake and framing
by hand with only the standard library is realistic but genuinely
error-prone to get exactly right; `websockets` is small, single-purpose,
and about as minimal a dependency as this specific job allows. Install:

```bash
pip install websockets --break-system-packages
```

## Build & run

```bash
cd ~/anubis_ws
colcon build --packages-select anubis_command_center
source install/setup.bash

ros2 launch anubis_command_center command_center.launch.py \
  vps_url:=ws://<vps-ip>:8090/ws/robot \
  map_dir:=$HOME/anubis_maps/default
```

Default values for the other parameters (`jpeg_quality`,
`max_image_width`, etc.) live in `config/command_center.yaml`, matching
every other package in this workspace -- edit that file to change them
permanently, or override on the command line for a one-off run.

`vps_url` and `map_dir` are deliberately **not** in that YAML file --
see the comments there for why (no sane default for the VPS address;
YAML can't expand `$HOME` the way a launch argument can).

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `image_topic` | `/sensors/camera/color/image_raw` | |
| `battery_topic` | `/battery_status` | |
| `map_dir` | `~/anubis_maps/default` | Matches `anubis_bringup`'s `ANUBIS_MAP_DIR` convention. |
| `vps_url` | *(required)* | e.g. `ws://1.2.3.4:8090/ws/robot`, or `wss://your-domain.example:8090/ws/robot` once the VPS has TLS. |
| `jpeg_quality` | `60` | 0-100, lower = smaller/faster over 5G. |
| `max_image_width` | `480` | Frames wider than this are downscaled before encoding. |
| `frame_interval_s` | `0.5` | How often a new frame is pushed, if a new one has arrived. |
| `status_interval_s` | `2.0` | How often battery/map-file status is pushed. |

## Verified

The full pipeline was run end-to-end for real in a sandbox: this
package's actual `_uplink_loop` function (unmodified, the same code
that ships here) was pointed at a real, running instance of
`anubis_vps_relay`'s relay server, fed a fake camera frame + battery
reading + a real `map.pgm` file through the same `_SharedState` object
the real ROS callbacks would write to, and a real WebSocket viewer
client confirmed it received all three message types
(frame/status/map), with the image byte-for-byte correct after the
base64 + relay round-trip and the map PNG's pixel dimensions correct
after PGM conversion.

**Not verified**: the actual `rclpy` subscription callbacks
(`_on_image`/`_on_battery`) and `cv_bridge` conversion, since no ROS2
install exists in the sandbox this was built in -- and, more
importantly, **real cellular/5G network conditions** (latency, packet
loss, how the carrier's NAT handles a long-lived outbound WebSocket,
whether it gets silently dropped after some idle period). Only
localhost was tested. Deploying both halves for a real field test is
the actual point of this package.

## Known gaps

- No authentication -- see `anubis_vps_relay`'s README for the same
  note from the relay side.
- Single map file check per status push (no caching) -- negligible
  cost at a 2-second interval, not worth optimizing yet.
- If the 5G link is slow enough that `frame_interval_s` can't keep up,
  frames simply queue up in send order (no explicit backpressure
  handling) -- watch for growing latency between frame timestamp and
  arrival if this becomes visible during testing.
