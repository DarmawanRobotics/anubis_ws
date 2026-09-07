#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from diagnostic_updater import DiagnosticStatusWrapper, Updater
from sensor_msgs.msg import Image, PointCloud2, Imu, CameraInfo

class SensorMonitor:
    """Monitor sensor topic and report its status."""
    def __init__(self, node, name, topic, msg_type, timeout=1.0):
        self.node = node
        self.name = name
        self.topic = topic
        self.timeout = timeout
        self.last_msg_time = None
        self.logged_unavailable = False
        self.logged_timeout = False
        self.sub = node.create_subscription(msg_type, topic, self.callback, qos_profile_sensor_data)

    def callback(self, msg):
        self.last_msg_time = self.node.get_clock().now()
        if self.logged_unavailable or self.logged_timeout:
            self.node.get_logger().info(f"[{self.name}] Topic available again: {self.topic}")
        self.logged_unavailable = False
        self.logged_timeout = False

    def diagnostic(self, stat: DiagnosticStatusWrapper):
        stat.add("Topic", self.topic)
        if self.last_msg_time is None:
            stat.summary(DiagnosticStatusWrapper.ERROR, "No data")
            if not self.logged_unavailable:
                self.node.get_logger().error(f"[{self.name}] Topic not available / no data: {self.topic}")
                self.logged_unavailable = True
            return stat

        age = (self.node.get_clock().now().nanoseconds - self.last_msg_time.nanoseconds) * 1e-9
        stat.add("Data age", f"{age:.2f} s")

        if age > self.timeout:
            stat.summary(DiagnosticStatusWrapper.ERROR, "Sensor timeout")
            if not self.logged_timeout:
                self.node.get_logger().error(f"[{self.name}] Sensor timeout: {self.topic} (last data {age:.2f}s ago)")
                self.logged_timeout = True
        else:
            stat.summary(DiagnosticStatusWrapper.OK, "OK")
        return stat

class SensorStatusNode(Node):
    """Monitor the status of all Anubis sensors."""
    def __init__(self):
        super().__init__("sensor_status_node")
        self.updater = Updater(self)
        self.updater.setHardwareID("ANUBIS-SENSORS")
        self.sensors = [
            SensorMonitor(self, "Front LiDAR", "/front_lidar", PointCloud2, 1.0),
            SensorMonitor(self, "Front LiDAR IMU", "/front_lidar/imu", Imu, 1.0),
            SensorMonitor(self, "Front Camera", "/camera/color/image_raw", Image, 1.0),
            SensorMonitor(self, "Front Camera Info", "/camera/color/camera_info", CameraInfo, 1.0),
            SensorMonitor(self, "Front Depth", "/camera/aligned_depth_to_color/image_raw", Image, 1.0),
        ]
        for sensor in self.sensors:
            self.updater.add(sensor.name, sensor.diagnostic)
        self.timer = self.create_timer(1.0, self.updater.update)
        self.get_logger().info("Anubis Sensor Monitor started")

def main(args=None):
    rclpy.init(args=args)
    node = SensorStatusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
