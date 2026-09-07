from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "local_ip",
            description="This machine's IP on the robot's network (check with `hostname -I`) -- required, no default on purpose",
        ),
        DeclareLaunchArgument("dog_ip", default_value="192.168.234.1"),
        DeclareLaunchArgument("sdk_model", default_value="zsl-1w"),
        DeclareLaunchArgument("require_localization_valid", default_value="true"),

        Node(
            package="anubis_control",
            executable="control_node",
            name="control_node",
            output="screen",
            parameters=[
                "config/control_params.yaml",
                {
                    "local_ip": LaunchConfiguration("local_ip"),
                    "dog_ip": LaunchConfiguration("dog_ip"),
                    "sdk_model": LaunchConfiguration("sdk_model"),
                    "require_localization_valid": LaunchConfiguration(
                        "require_localization_valid"
                    ),
                },
            ],
        ),
    ])
