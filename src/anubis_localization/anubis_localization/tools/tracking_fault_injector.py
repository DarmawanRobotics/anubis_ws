#!/usr/bin/env python3

import math
import struct
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_srvs.srv import Trigger


def _outlier_payload():
    points = []
    for layer in range(2):
        for row in range(16):
            for column in range(16):
                points.append(
                    struct.pack(
                        "<ffff",
                        1000.0 + 0.2 * column,
                        1000.0 + 0.2 * row,
                        -1.0 + 2.0 * layer,
                        1.0,
                    )
                )
    return b"".join(points)


OUTLIER_DATA = _outlier_payload()
OUTLIER_COUNT = len(OUTLIER_DATA) // 16


class TrackingFaultInjector(Node):
    def __init__(self):
        super().__init__("tracking_fault_injector")
        self.declare_parameter("input_topic", "/livox/lidar")
        self.declare_parameter("output_topic", "/localization_test/lidar")
        self.declare_parameter("reject_duration_s", 1.0)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publisher = self.create_publisher(PointCloud2, output_topic, qos)
        self._subscription = self.create_subscription(
            PointCloud2, input_topic, self._point_cloud_callback, qos
        )
        self._service = self.create_service(
            Trigger, "~/inject_rejections", self._inject_callback
        )
        self._reject_until = 0.0
        self._was_injecting = False
        self.get_logger().info(
            f"Relaying {input_topic} to {output_topic}; fault injection is idle"
        )

    def _inject_callback(self, _request, response):
        duration = float(self.get_parameter("reject_duration_s").value)
        if not math.isfinite(duration) or duration <= 0.0:
            response.success = False
            response.message = "reject_duration_s must be finite and greater than zero"
            return response

        self._reject_until = time.monotonic() + duration
        self._was_injecting = True
        response.success = True
        response.message = f"publishing deterministic outlier clouds for {duration:.3f}s"
        self.get_logger().warn(response.message)
        return response

    def _point_cloud_callback(self, message):
        injecting = time.monotonic() < self._reject_until
        if not injecting:
            if self._was_injecting:
                self.get_logger().info("Fault interval ended; normal point-cloud relay resumed")
                self._was_injecting = False
            self._publisher.publish(message)
            return

        outlier = PointCloud2()
        outlier.header = message.header
        outlier.height = 1
        outlier.width = OUTLIER_COUNT
        outlier.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        outlier.is_bigendian = False
        outlier.point_step = 16
        outlier.row_step = outlier.point_step * outlier.width
        outlier.data = OUTLIER_DATA
        outlier.is_dense = True
        self._publisher.publish(outlier)


def main(args=None):
    rclpy.init(args=args)
    node = TrackingFaultInjector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
