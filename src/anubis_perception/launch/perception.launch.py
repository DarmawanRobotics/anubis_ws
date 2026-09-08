from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("anubis_perception")

    abandoned_detection_launch = PathJoinSubstitution([
        package_share,
        "launch",
        "abandoned_detection.launch.py",
    ])

    apriltag_launch = PathJoinSubstitution([
        package_share,
        "launch",
        "apriltag.launch.py",
    ])

    return LaunchDescription([
        # ============================================================
        # Abandoned Object / Person Detection
        # ============================================================
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                abandoned_detection_launch
            )
        ),

        # ============================================================
        # AprilTag Detection
        # ============================================================
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                apriltag_launch
            )
        ),
    ])