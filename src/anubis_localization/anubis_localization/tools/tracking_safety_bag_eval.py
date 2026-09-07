#!/usr/bin/env python3

import argparse
import bisect
import json
import math
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TRACKED_TOPICS = {
    "/status",
    "/odom",
    "/odom/localization_odom",
    "/cmd_vel",
    "/localization/valid",
    "/localization/request_initialpose",
    "/tf",
}


def percentile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def yaw_from_quaternion(quaternion):
    sin_yaw = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y
    )
    cos_yaw = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z
    )
    return math.atan2(sin_yaw, cos_yaw)


def wrapped_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def message_time(message, bag_time_ns):
    header = getattr(message, "header", None)
    if header is not None:
        stamp = header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if stamp_ns > 0:
            return stamp_ns / 1_000_000_000.0
    return bag_time_ns / 1_000_000_000.0


def normalized_planar_trajectory(samples):
    if not samples:
        return []
    _, x0, y0, yaw0 = samples[0]
    cos_yaw = math.cos(yaw0)
    sin_yaw = math.sin(yaw0)
    normalized = []
    for stamp, x, y, _yaw in samples:
        dx = x - x0
        dy = y - y0
        normalized.append(
            (
                stamp,
                cos_yaw * dx + sin_yaw * dy,
                -sin_yaw * dx + cos_yaw * dy,
            )
        )
    return normalized


def trajectory_errors(localization_samples, robot_samples, max_pair_age_s=0.15):
    localization = normalized_planar_trajectory(localization_samples)
    robot = normalized_planar_trajectory(robot_samples)
    if not localization or not robot:
        return []

    robot_times = [sample[0] for sample in robot]
    errors = []
    for stamp, x, y in localization:
        insertion = bisect.bisect_left(robot_times, stamp)
        candidates = []
        if insertion < len(robot):
            candidates.append(robot[insertion])
        if insertion > 0:
            candidates.append(robot[insertion - 1])
        if not candidates:
            continue
        nearest = min(candidates, key=lambda sample: abs(sample[0] - stamp))
        if abs(nearest[0] - stamp) <= max_pair_age_s:
            errors.append(math.hypot(x - nearest[1], y - nearest[2]))
    return errors


def validity_intervals(events, end_time):
    intervals = []
    invalid_start = None
    for stamp, valid in sorted(events):
        if not valid and invalid_start is None:
            invalid_start = stamp
        elif valid and invalid_start is not None:
            intervals.append((invalid_start, stamp))
            invalid_start = None
    if invalid_start is not None:
        intervals.append((invalid_start, end_time))
    return intervals


def tf_interval_report(intervals, map_to_odom):
    reports = []
    for start, end in intervals:
        grace_end = min(end, start + 2.0)
        grace_samples = [sample for sample in map_to_odom if start <= sample[0] <= grace_end]
        max_xy_change = None
        max_yaw_change = None
        if grace_samples:
            _, x0, y0, yaw0 = grace_samples[0]
            max_xy_change = max(
                math.hypot(x - x0, y - y0) for _, x, y, _yaw in grace_samples
            )
            max_yaw_change = max(
                abs(wrapped_angle(yaw - yaw0))
                for _, _x, _y, yaw in grace_samples
            )
        post_timeout_samples = sum(
            1 for stamp, _x, _y, _yaw in map_to_odom if start + 2.0 < stamp < end
        )
        reports.append(
            {
                "start_s": start,
                "duration_s": max(0.0, end - start),
                "grace_tf_samples": len(grace_samples),
                "grace_max_xy_change_m": max_xy_change,
                "grace_max_yaw_change_rad": max_yaw_change,
                "post_timeout_tf_samples": post_timeout_samples,
            }
        )
    return reports


def evaluate_bag(bag_path):
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path), storage_id="sqlite3"
    )
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }
    message_types = {
        topic: get_message(topic_types[topic])
        for topic in TRACKED_TOPICS
        if topic in topic_types
    }

    scores = []
    nonconverged_count = 0
    localization_velocity = []
    command_linear = []
    command_angular = []
    localization_poses = []
    robot_poses = []
    validity_events = []
    request_initialpose_true_count = 0
    map_to_odom = []
    last_bag_time = 0.0

    while reader.has_next():
        topic, serialized, bag_time_ns = reader.read_next()
        last_bag_time = max(last_bag_time, bag_time_ns / 1_000_000_000.0)
        message_type = message_types.get(topic)
        if message_type is None:
            continue
        message = deserialize_message(serialized, message_type)

        if topic == "/status":
            if math.isfinite(message.matching_error):
                scores.append(float(message.matching_error))
            if not message.has_converged:
                nonconverged_count += 1
        elif topic in ("/odom", "/odom/localization_odom"):
            stamp = message_time(message, bag_time_ns)
            pose = message.pose.pose
            sample = (
                stamp,
                float(pose.position.x),
                float(pose.position.y),
                yaw_from_quaternion(pose.orientation),
            )
            if topic == "/odom":
                robot_poses.append(sample)
            else:
                localization_poses.append(sample)
                velocity = message.twist.twist.linear
                localization_velocity.append(
                    math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                )
        elif topic == "/cmd_vel":
            command_linear.append(math.hypot(message.linear.x, message.linear.y))
            command_angular.append(abs(message.angular.z))
        elif topic == "/localization/valid":
            validity_events.append(
                (bag_time_ns / 1_000_000_000.0, bool(message.data))
            )
        elif topic == "/localization/request_initialpose":
            request_initialpose_true_count += int(bool(message.data))
        elif topic == "/tf":
            for transform in message.transforms:
                parent = transform.header.frame_id.lstrip("/")
                child = transform.child_frame_id.lstrip("/")
                if parent == "map" and child == "odom":
                    translation = transform.transform.translation
                    map_to_odom.append(
                        (
                            bag_time_ns / 1_000_000_000.0,
                            float(translation.x),
                            float(translation.y),
                            yaw_from_quaternion(transform.transform.rotation),
                        )
                    )

    relative_errors = trajectory_errors(localization_poses, robot_poses)
    intervals = validity_intervals(validity_events, last_bag_time)
    return {
        "matching": {
            "samples": len(scores),
            "nonconverged_samples": nonconverged_count,
            "score_p95": percentile(scores, 0.95),
        },
        "localization_velocity": {
            "samples": len(localization_velocity),
            "max_norm_mps": max(localization_velocity, default=None),
            "p95_norm_mps": percentile(localization_velocity, 0.95),
        },
        "command_envelope": {
            "samples": len(command_linear),
            "max_linear_mps": max(command_linear, default=None),
            "max_angular_rps": max(command_angular, default=None),
        },
        "relative_trajectory": {
            "paired_samples": len(relative_errors),
            "max_xy_error_m": max(relative_errors, default=None),
            "p95_xy_error_m": percentile(relative_errors, 0.95),
        },
        "localization_valid": {
            "messages": len(validity_events),
            "invalid_intervals": tf_interval_report(intervals, map_to_odom),
        },
        "request_initialpose_true_count": request_initialpose_true_count,
        "map_to_odom_samples": len(map_to_odom),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate localization tracking and safety metrics from a ROS 2 bag"
    )
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = evaluate_bag(args.bag)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
