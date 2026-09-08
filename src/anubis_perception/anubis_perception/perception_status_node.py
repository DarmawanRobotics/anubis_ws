#!/usr/bin/env python3
import time
import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from diagnostic_updater import DiagnosticStatusWrapper, Updater
from rclpy.node import Node
from std_msgs.msg import Bool, String


class PerceptionStatusNode(Node):
    """Monitor health status of AprilTag and abandoned detection perception pipelines."""

    def __init__(self):
        super().__init__("perception_status_node")
        self.last_apriltag_time = None
        self.last_apriltag_count = 0
        self.last_apriltag_ids = []
        self.last_abandoned_time = None
        self.abandoned_alarm = False

        self.create_subscription(
            AprilTagDetectionArray,
            "/perception/apriltag/detections",
            self._apriltag_cb,
            10,
        )
        self.create_subscription(
            Bool,
            "/abandoned_detection/alarm",
            self._abandoned_cb,
            10,
        )
        self.updater = Updater(self)
        self.updater.setHardwareID("anubis_perception")
        self.updater.add("AprilTag pipeline", self._check_apriltag)
        self.updater.add("Abandoned detection pipeline", self._check_abandoned)
        self.create_timer(1.0, self.updater.update)

    def _apriltag_cb(self, msg: AprilTagDetectionArray):
        self.last_apriltag_time = time.time()
        self.last_apriltag_count = len(msg.detections)
        self.last_apriltag_ids = [d.id for d in msg.detections]

    def _abandoned_cb(self, msg: Bool):
        self.last_abandoned_time = time.time()
        self.abandoned_alarm = msg.data

    def _check_apriltag(self, stat: DiagnosticStatusWrapper):
        if self.last_apriltag_time is None:
            stat.summary(stat.WARN, "No AprilTag message received yet")
            return stat

        age = time.time() - self.last_apriltag_time
        if age > 2.0:
            stat.summary(stat.ERROR, f"AprilTag pipeline stalled ({age:.1f}s)")
        elif self.last_apriltag_count == 0:
            stat.summary(stat.OK, "AprilTag pipeline alive, no tag visible")
        else:
            stat.summary(
                stat.OK,
                f"AprilTag pipeline alive, {self.last_apriltag_count} tag(s): {self.last_apriltag_ids}",
            )
        return stat

    def _check_abandoned(self, stat: DiagnosticStatusWrapper):
        if self.last_abandoned_time is None:
            stat.summary(stat.WARN, "No abandoned detection message received yet")
            return stat

        age = time.time() - self.last_abandoned_time
        if age > 2.0:
            stat.summary(stat.ERROR, f"Abandoned detection pipeline stalled ({age:.1f}s)")
        elif self.abandoned_alarm:
            stat.summary(stat.WARN, "Abandoned object detected")
        else:
            stat.summary(stat.OK, "Abandoned detection pipeline alive")
        return stat


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionStatusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
