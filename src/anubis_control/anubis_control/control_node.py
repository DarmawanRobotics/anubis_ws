"""anubis_control / control_node.py

Bridges /cmd_vel to the ZSL-1W SDK and publishes odometry + diagnostics
back to ROS2.

Deliberately does NOT implement gamepad/remote mode arbitration (the
170/171 mode-switch dance from darmawan_ws's dog_sdk_bridge_node.py) --
that is a separate concern, planned as its own node/package later so
this one stays focused on a single job: cmd_vel in, SDK calls out.

Safety model:
  - Motion is blocked (fail closed) unless /localization/valid == True,
    when require_localization_valid is set (default: True). For bench
    testing before anubis_localization exists, either set
    require_localization_valid:=false, or publish a manual True:
      ros2 topic pub /localization/valid std_msgs/msg/Bool "{data: true}"
  - A watchdog stops the robot if no /cmd_vel has arrived within
    watchdog_timeout seconds.
  - /control_node/emergency_stop (std_srvs/Trigger) always works,
    regardless of the above.
"""

import time
from threading import Lock
from typing import Optional, Tuple

import rclpy
from diagnostic_updater import DiagnosticStatusWrapper, Updater
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from anubis_control.math_utils import (
    apply_deadzone,
    clamp,
    localization_allows_motion,
    normalize_battery_percentage,
    rpy_to_quaternion,
    wrapped_yaw_rate,
)
from anubis_control.sdk_loader import SdkLoadError, load_sdk


# zsl-1w's getCurrentCtrlmode() only ever returns one of these three
# values (docs/api_zsl-1w.md in genisom_l1_sdk_old) -- do not copy a
# 6-value table from a different robot's SDK docs; that mismatch cost a
# lot of confused debugging in darmawan_ws.
DOG_STATE_DAMPING = 0
DOG_STATE_STANDING = 1
DOG_STATE_MOVING = 3
MOVABLE_STATES = {DOG_STATE_STANDING, DOG_STATE_MOVING}


class ControlNode(Node):
    def __init__(self):
        super().__init__("control_node")

        # ---- Parameters ----
        self.declare_parameter("dog_ip", "192.168.234.1")
        self.declare_parameter("local_ip", "0.0.0.0")  # must be set explicitly
        self.declare_parameter("local_port", 43988)
        self.declare_parameter("sdk_model", "zsl-1w")
        self.declare_parameter("control_frequency_hz", 10.0)
        self.declare_parameter("watchdog_timeout_s", 1.0)
        self.declare_parameter("max_linear_vel_mps", 0.6)
        self.declare_parameter("max_angular_vel_rps", 1.0)
        self.declare_parameter("linear_deadzone_mps", 0.05)
        self.declare_parameter("auto_stand", True)
        self.declare_parameter("publish_odom", True)
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("require_localization_valid", True)
        self.declare_parameter("localization_valid_topic", "/localization/valid")
        self.declare_parameter("low_battery_threshold_pct", 15.0)

        self._dog_ip = self.get_parameter("dog_ip").value
        self._local_ip = self.get_parameter("local_ip").value
        self._local_port = self.get_parameter("local_port").value
        self._sdk_model = self.get_parameter("sdk_model").value
        self._control_period_s = 1.0 / self.get_parameter("control_frequency_hz").value
        self._watchdog_timeout_s = self.get_parameter("watchdog_timeout_s").value
        self._max_linear_vel = self.get_parameter("max_linear_vel_mps").value
        self._max_angular_vel = self.get_parameter("max_angular_vel_rps").value
        self._linear_deadzone = self.get_parameter("linear_deadzone_mps").value
        self._auto_stand = self.get_parameter("auto_stand").value
        self._publish_odom = self.get_parameter("publish_odom").value
        self._odom_frame_id = self.get_parameter("odom_frame_id").value
        self._base_frame_id = self.get_parameter("base_frame_id").value
        self._require_localization_valid = self.get_parameter(
            "require_localization_valid"
        ).value
        self._localization_valid_topic = self.get_parameter(
            "localization_valid_topic"
        ).value
        self._low_battery_threshold_pct = self.get_parameter(
            "low_battery_threshold_pct"
        ).value

        if self._local_ip in ("0.0.0.0", "", None):
            raise RuntimeError(
                "local_ip parameter must be set explicitly to this machine's "
                "IP on the robot's network (check with `hostname -I`) -- "
                "0.0.0.0 is a guaranteed bind failure, kept as the default "
                "specifically so this is never accidentally left unset."
            )

        # ---- Runtime state ----
        self._cmd_lock = Lock()
        self._send_lock = Lock()
        self._latest_cmd: Optional[Twist] = None
        self._last_cmd_time = 0.0
        self._localization_valid: Optional[bool] = None
        self._standing_confirmed = False
        self._last_standup_attempt = 0.0
        self._last_odom_yaw: Optional[float] = None
        self._last_odom_time: Optional[float] = None
        self._sdk_connected: Optional[bool] = None

        # ---- SDK ----
        try:
            handle = load_sdk(model=self._sdk_model)
        except SdkLoadError as exc:
            self.get_logger().fatal(str(exc))
            raise
        self._dog = handle.module.HighLevel()
        self._dog.initRobot(self._local_ip, self._local_port, self._dog_ip)
        self.get_logger().info(
            f"SDK loaded: model={handle.model} arch={handle.arch} "
            f"lib_path={handle.lib_path}"
        )
        self._sdk_connected = self._wait_for_connection(timeout_s=5.0)
        if not self._sdk_connected:
            self.get_logger().error(
                "Initial connection check failed after 5s; will keep "
                "re-checking every ~10s (see _diagnose_connection)"
            )

        # ---- Subscriptions ----
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, 10)

        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            Bool, self._localization_valid_topic, self._localization_valid_cb, latched_qos
        )

        # ---- Publishers ----
        self._odom_pub = self.create_publisher(Odometry, "/odom", 10) if self._publish_odom else None
        self._tf_broadcaster = TransformBroadcaster(self) if self._publish_odom else None

        battery_qos = QoSProfile(depth=1)
        battery_qos.reliability = ReliabilityPolicy.RELIABLE
        battery_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._battery_pub = self.create_publisher(BatteryState, "/battery_status", battery_qos)

        # ---- Services ----
        self.create_service(Trigger, "~/emergency_stop", self._emergency_stop_cb)

        # ---- Timers ----
        self.create_timer(self._control_period_s, self._control_loop)
        self.create_timer(10.0, self._diagnose_connection_and_battery)

        # ---- Diagnostics (diagnostic_updater, same convention as anubis_sensors) ----
        self._diag_updater = Updater(self)
        self._diag_updater.setHardwareID("anubis_control")
        self._diag_updater.add("SDK connection", self._diagnostic_sdk_connection)
        self._diag_updater.add("Battery", self._diagnostic_battery)
        self.create_timer(1.0, self._diag_updater.update)
        self._last_battery_pct: Optional[float] = None

        self.get_logger().info(
            f"control_node ready | sdk_model={self._sdk_model} dog_ip={self._dog_ip} "
            f"local_ip={self._local_ip}"
        )

    # ------------------------------------------------------------------
    # SDK connection
    # ------------------------------------------------------------------

    def _wait_for_connection(self, timeout_s: float, poll_interval_s: float = 0.5) -> bool:
        """Poll checkConnect() for up to timeout_s.

        initRobot()'s handshake is asynchronous; checking once immediately
        after initRobot() races it and can report failure even though the
        connection succeeds moments later. Confirmed with this exact
        symptom on this SDK during darmawan_ws bring-up.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                if self._dog.checkConnect():
                    return True
            except Exception:
                pass
            time.sleep(poll_interval_s)
        return False

    def _diagnose_connection_and_battery(self):
        """Periodic (10s) recheck -- logs actual connect/disconnect transitions."""
        try:
            connected_now = bool(self._dog.checkConnect())
        except Exception:
            connected_now = False
        if connected_now != self._sdk_connected:
            if connected_now:
                self.get_logger().info("SDK connection (re)established")
            else:
                self.get_logger().error("SDK connection lost")
        self._sdk_connected = connected_now

        try:
            battery_pct = float(self._dog.getBatteryPower())
        except Exception:
            return
        self._last_battery_pct = battery_pct

        normalized = normalize_battery_percentage(battery_pct)
        if normalized is None:
            self.get_logger().warn(f"Ignoring invalid battery reading: {battery_pct!r}")
            return
        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.percentage = normalized
        msg.present = True
        msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_UNKNOWN
        self._battery_pub.publish(msg)

        if battery_pct <= self._low_battery_threshold_pct:
            self.get_logger().warn(
                f"Battery low: {battery_pct:.0f}% (threshold: {self._low_battery_threshold_pct:.0f}%)"
            )

    def _diagnostic_sdk_connection(self, stat: DiagnosticStatusWrapper):
        if self._sdk_connected:
            stat.summary(stat.OK, "Connected")
        else:
            stat.summary(stat.ERROR, "Not connected")
        return stat

    def _diagnostic_battery(self, stat: DiagnosticStatusWrapper):
        if self._last_battery_pct is None:
            stat.summary(stat.WARN, "No battery reading yet")
        elif self._last_battery_pct <= self._low_battery_threshold_pct:
            stat.summary(stat.WARN, f"Low: {self._last_battery_pct:.0f}%")
        else:
            stat.summary(stat.OK, f"{self._last_battery_pct:.0f}%")
        return stat

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _cmd_vel_cb(self, msg: Twist):
        with self._cmd_lock:
            if not localization_allows_motion(
                self._require_localization_valid, self._localization_valid
            ):
                return
            self._latest_cmd = msg
            self._last_cmd_time = time.time()

    def _localization_valid_cb(self, msg: Bool):
        with self._cmd_lock:
            previous = self._localization_valid
            self._localization_valid = bool(msg.data)
            if not msg.data:
                self._latest_cmd = None
                self._last_cmd_time = 0.0

        if not msg.data:
            self._send_stop()
            if previous is not False:
                self.get_logger().error(
                    "Localization invalid: motion blocked, zero velocity sent"
                )
        elif previous is not True:
            self.get_logger().info("Localization valid: motion unblocked")

    def _emergency_stop_cb(self, request, response):
        self.get_logger().warn("Emergency stop requested")
        with self._cmd_lock:
            self._latest_cmd = None
            self._last_cmd_time = 0.0
        self._send_stop()
        response.success = True
        response.message = "stopped"
        return response

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self):
        now = time.time()

        if self._publish_odom:
            self._publish_odometry()

        with self._cmd_lock:
            cmd = self._latest_cmd
            last_time = self._last_cmd_time
            localization_valid = self._localization_valid

        if not localization_allows_motion(self._require_localization_valid, localization_valid):
            return  # already stopped by _localization_valid_cb

        if cmd is None or (now - last_time) > self._watchdog_timeout_s:
            return  # nothing to send; robot stays at rest

        if self._auto_stand and not self._standing_confirmed:
            self._ensure_standing()

        vx = apply_deadzone(cmd.linear.x, self._linear_deadzone)
        vy = apply_deadzone(cmd.linear.y, self._linear_deadzone)
        vx = clamp(vx, -self._max_linear_vel, self._max_linear_vel)
        vy = clamp(vy, -self._max_linear_vel * 0.6, self._max_linear_vel * 0.6)
        yaw_rate = clamp(cmd.angular.z, -self._max_angular_vel, self._max_angular_vel)

        self._send_velocity(vx, vy, yaw_rate)

    def _ensure_standing(self):
        now = time.time()
        if now - self._last_standup_attempt < 5.0:
            return
        self._last_standup_attempt = now
        try:
            mode = self._dog.getCurrentCtrlmode()
            if mode in MOVABLE_STATES:
                self._standing_confirmed = True
                return
            self.get_logger().info(f"mode={mode}, calling standUp()")
            self._dog.standUp()
            time.sleep(1.0)
            new_mode = self._dog.getCurrentCtrlmode()
            self._standing_confirmed = new_mode in MOVABLE_STATES
            if not self._standing_confirmed:
                self.get_logger().warn(f"mode={new_mode} after standUp(), will retry")
        except Exception as exc:
            self.get_logger().error(f"standUp() failed: {exc}")

    def _send_velocity(self, vx: float, vy: float, yaw_rate: float):
        with self._send_lock:
            try:
                self._dog.move(vx, vy, yaw_rate)
            except Exception as exc:
                self.get_logger().warn(f"move() failed: {exc}")

    def _send_stop(self):
        with self._send_lock:
            try:
                self._dog.move(0.0, 0.0, 0.0)
            except Exception as exc:
                self.get_logger().error(f"stop failed: {exc}")

    # ------------------------------------------------------------------
    # Odometry
    # ------------------------------------------------------------------

    def _publish_odometry(self):
        try:
            pos = self._dog.getPosition()
            rpy = self._dog.getRPY()
            body_vel = self._dog.getBodyVelocity()
        except Exception:
            return

        try:
            qx, qy, qz, qw = rpy_to_quaternion(rpy[0], rpy[1], rpy[2])
        except ValueError:
            self.get_logger().warn("Non-finite RPY from SDK; odom sample dropped")
            return

        now = time.time()
        yaw_rate = 0.0
        if self._last_odom_yaw is not None and self._last_odom_time is not None:
            yaw_rate = wrapped_yaw_rate(self._last_odom_yaw, rpy[2], now - self._last_odom_time)
        self._last_odom_yaw = rpy[2]
        self._last_odom_time = now

        stamp = self.get_clock().now().to_msg()

        if self._tf_broadcaster is not None:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self._odom_frame_id
            t.child_frame_id = self._base_frame_id
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2])
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self._tf_broadcaster.sendTransform(t)

        if self._odom_pub is not None:
            msg = Odometry()
            msg.header.stamp = stamp
            msg.header.frame_id = self._odom_frame_id
            msg.child_frame_id = self._base_frame_id
            msg.pose.pose.position.x = float(pos[0])
            msg.pose.pose.position.y = float(pos[1])
            msg.pose.pose.position.z = float(pos[2])
            msg.pose.pose.orientation.x = qx
            msg.pose.pose.orientation.y = qy
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw
            msg.twist.twist.linear.x = float(body_vel[0])
            msg.twist.twist.linear.y = float(body_vel[1])
            msg.twist.twist.linear.z = float(body_vel[2])
            msg.twist.twist.angular.z = yaw_rate
            self._odom_pub.publish(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy_node(self):
        self.get_logger().info("control_node shutting down, stopping robot")
        with self._cmd_lock:
            self._latest_cmd = None
        self._send_stop()
        try:
            self._dog.passive()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
