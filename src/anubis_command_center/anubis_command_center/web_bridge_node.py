
"""anubis_command_center / web_bridge_node.py

Pushes the camera image, battery status, and map file info OUT to a
relay running on a VPS, over an outbound WebSocket connection. This is
the shape that actually works over cellular/5G: the robot is behind
carrier NAT with no public IP, so it must be the one that connects
OUT -- nothing can connect IN to it. The VPS (public IP + domain) runs
anubis_vps_relay's relay_server.py; browsers talk to that relay, never
to the robot directly. See anubis_vps_relay/README.md for the other
half of this system.

    this node (WebSocket client) --> [VPS relay] <-- browser (viewer)

Reconnects with backoff if the link drops -- exactly the condition
this whole exercise exists to test on the robot's real 5G connection.

MESSAGE PROTOCOL -- see anubis_vps_relay/relay_server.py's own
docstring for the authoritative definition; duplicated here only
enough to keep this file's intent readable on its own:
    {"type": "frame",  "jpg_b64": ..., "width": W, "height": H, "ts": ...}
    {"type": "status", "battery_pct": F|null, "map_dir": ...,
                        "has_pcd": bool, "has_pgm": bool, "has_yaml": bool, "ts": ...}
    {"type": "map",    "png_b64": ..., "ts": ...}
"""

import asyncio
import base64
import json
import os
import threading
import time

import cv2
import rclpy
import websockets
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, Image

try:
    from PIL import Image as PILImage
    _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False


class _SharedState:
    """Thread-safe box for the latest image/battery, written from ROS
    callbacks (rclpy's executor thread) and read from the asyncio
    WebSocket send-loop (a separate background thread)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.jpeg_bytes = None
        self.image_width = None
        self.image_height = None
        self.image_seq = 0  # bumped on every new frame, lets the sender
                             # detect "is this a frame I haven't sent yet"
        self.battery_pct = None

    def set_image(self, jpeg_bytes, width, height):
        with self._lock:
            self.jpeg_bytes = jpeg_bytes
            self.image_width = width
            self.image_height = height
            self.image_seq += 1

    def set_battery(self, pct):
        with self._lock:
            self.battery_pct = pct

    def snapshot(self):
        with self._lock:
            return {
                "jpeg_bytes": self.jpeg_bytes,
                "image_width": self.image_width,
                "image_height": self.image_height,
                "image_seq": self.image_seq,
                "battery_pct": self.battery_pct,
            }


async def _uplink_loop(state: _SharedState, map_dir: str, vps_url: str,
                        frame_interval_s: float, status_interval_s: float,
                        logger):
    """Runs forever in its own thread's asyncio event loop. Connects to
    the VPS relay, pushes updates, and reconnects with backoff on any
    failure -- dropped/flaky connections are the expected, normal case
    over a cellular link, not an exceptional one to crash on."""
    last_sent_image_seq = -1
    last_status_sent_at = 0.0
    last_map_mtime_sent = None
    backoff_s = 1.0
    max_backoff_s = 30.0

    while True:
        try:
            logger.info(f"Connecting to VPS relay: {vps_url}")
            async with websockets.connect(vps_url, ping_interval=15, ping_timeout=10) as ws:
                logger.info("Connected to VPS relay")
                backoff_s = 1.0  # reset backoff after a successful connect

                while True:
                    snap = state.snapshot()

                    if (snap["jpeg_bytes"] is not None
                            and snap["image_seq"] != last_sent_image_seq):
                        await ws.send(json.dumps({
                            "type": "frame",
                            "jpg_b64": base64.b64encode(snap["jpeg_bytes"]).decode("ascii"),
                            "width": snap["image_width"],
                            "height": snap["image_height"],
                            "ts": time.time(),
                        }))
                        last_sent_image_seq = snap["image_seq"]

                    now = time.time()
                    if now - last_status_sent_at >= status_interval_s:
                        await ws.send(json.dumps({
                            "type": "status",
                            "battery_pct": snap["battery_pct"],
                            "map_dir": map_dir,
                            "has_pcd": os.path.isfile(os.path.join(map_dir, "map.pcd")),
                            "has_pgm": os.path.isfile(os.path.join(map_dir, "map.pgm")),
                            "has_yaml": os.path.isfile(os.path.join(map_dir, "map.yaml")),
                            "ts": now,
                        }))
                        last_status_sent_at = now

                        pgm_path = os.path.join(map_dir, "map.pgm")
                        if _HAVE_PIL and os.path.isfile(pgm_path):
                            mtime = os.path.getmtime(pgm_path)
                            if mtime != last_map_mtime_sent:
                                try:
                                    with PILImage.open(pgm_path) as im:
                                        import io
                                        buf = io.BytesIO()
                                        im.convert("L").save(buf, format="PNG")
                                        await ws.send(json.dumps({
                                            "type": "map",
                                            "png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                                            "ts": now,
                                        }))
                                    last_map_mtime_sent = mtime
                                    logger.info("Pushed updated map.pgm to relay")
                                except Exception as exc:
                                    logger.warn(f"Failed to convert/send map.pgm: {exc}")

                    await asyncio.sleep(frame_interval_s)

        except (websockets.exceptions.ConnectionClosed, OSError) as exc:
            logger.warn(f"VPS relay connection lost ({exc}); retrying in {backoff_s:.0f}s")
        except Exception as exc:
            logger.warn(f"Unexpected uplink error ({exc}); retrying in {backoff_s:.0f}s")

        await asyncio.sleep(backoff_s)
        backoff_s = min(backoff_s * 2, max_backoff_s)


class WebBridgeNode(Node):
    def __init__(self):
        super().__init__("web_bridge_node")

        self.declare_parameter("image_topic", "/sensors/camera/color/image_raw")
        self.declare_parameter("battery_topic", "/battery_status")
        self.declare_parameter("map_dir", os.path.expanduser("~/anubis_maps/default"))
        self.declare_parameter("jpeg_quality", 60)
        self.declare_parameter("max_image_width", 480)
        self.declare_parameter(
            "vps_url", "ws://YOUR_VPS_HOST_OR_DOMAIN:8090/ws/robot",
            # No sane default exists here -- this MUST point at a real
            # VPS. Left as an obviously-fake placeholder so a forgotten
            # override fails immediately and loudly (DNS/connection
            # error) instead of silently trying to reach something that
            # sounds plausible.
        )
        self.declare_parameter("frame_interval_s", 0.5)
        self.declare_parameter("status_interval_s", 2.0)

        self._image_topic = self.get_parameter("image_topic").value
        self._battery_topic = self.get_parameter("battery_topic").value
        self._map_dir = self.get_parameter("map_dir").value
        self._jpeg_quality = self.get_parameter("jpeg_quality").value
        self._max_image_width = self.get_parameter("max_image_width").value
        self._vps_url = self.get_parameter("vps_url").value
        self._frame_interval_s = self.get_parameter("frame_interval_s").value
        self._status_interval_s = self.get_parameter("status_interval_s").value

        if "YOUR_VPS_HOST_OR_DOMAIN" in self._vps_url:
            self.get_logger().fatal(
                "vps_url is still the placeholder default -- set it to your "
                "real VPS, e.g. vps_url:=ws://1.2.3.4:8090/ws/robot or "
                "wss://your-domain.example:8090/ws/robot"
            )
            raise RuntimeError("vps_url not configured")

        self._state = _SharedState()
        self._bridge = CvBridge()

        self.create_subscription(Image, self._image_topic, self._on_image, 5)
        self.create_subscription(BatteryState, self._battery_topic, self._on_battery, 1)

        self._start_uplink_thread()

        self.get_logger().info(
            f"web_bridge_node ready | pushing to {self._vps_url} | "
            f"image_topic={self._image_topic} map_dir={self._map_dir}"
        )

    def _on_image(self, msg: Image):
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        height, width = cv_image.shape[:2]
        if self._max_image_width and width > self._max_image_width:
            scale = self._max_image_width / float(width)
            cv_image = cv2.resize(cv_image, (self._max_image_width, int(height * scale)))
            height, width = cv_image.shape[:2]

        ok, encoded = cv2.imencode(
            ".jpg", cv_image, [cv2.IMWRITE_JPEG_QUALITY, int(self._jpeg_quality)]
        )
        if not ok:
            self.get_logger().warn("JPEG encode failed")
            return
        self._state.set_image(encoded.tobytes(), width, height)

    def _on_battery(self, msg: BatteryState):
        if 0.0 <= msg.percentage <= 1.0:
            self._state.set_battery(msg.percentage * 100.0)

    def _start_uplink_thread(self):
        def run():
            asyncio.run(_uplink_loop(
                self._state, self._map_dir, self._vps_url,
                self._frame_interval_s, self._status_interval_s,
                self.get_logger(),
            ))
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self._uplink_thread = thread


def main(args=None):
    rclpy.init(args=args)
    node = WebBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
