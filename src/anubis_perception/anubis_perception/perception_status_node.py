import time

import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from diagnostic_updater import DiagnosticStatusWrapper, Updater
from rclpy.node import Node


class PerceptionStatusNode(Node):
    def __init__(self):
        super().__init__("perception_status_node")
        self._last_message_time = None
        self._last_tag_count = 0
        self._last_tag_ids = []

        self.create_subscription(
            AprilTagDetectionArray,
            "/perception/apriltag/detections",
            self._detections_cb,
            10,
        )

        self.updater = Updater(self)
        self.updater.setHardwareID("anubis_perception")
        self.updater.add("AprilTag pipeline", self._check_pipeline)
        self.create_timer(1.0, self.updater.update)

    def _detections_cb(self, msg: AprilTagDetectionArray):
        self._last_message_time = time.time()
        self._last_tag_count = len(msg.detections)
        self._last_tag_ids = [d.id for d in msg.detections]

    def _check_pipeline(self, stat: DiagnosticStatusWrapper):
        if self._last_message_time is None:
            stat.summary(
                stat.WARN,
                "No message received yet from apriltag_ros -- is it running?",
            )
            return stat

        age = time.time() - self._last_message_time
        if age > 2.0:
            stat.summary(
                stat.ERROR,
                f"No detections message for {age:.1f}s -- apriltag_ros "
                f"pipeline appears stalled (image feed lost, or node died)",
            )
            return stat

        if self._last_tag_count == 0:
            stat.summary(stat.OK, "Pipeline alive, no tag currently in view")
        else:
            stat.summary(
                stat.OK, f"Pipeline alive, {self._last_tag_count} tag(s) visible: {self._last_tag_ids}"
            )
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
