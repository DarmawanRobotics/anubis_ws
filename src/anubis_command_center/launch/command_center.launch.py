import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_path = PathJoinSubstitution(
        [FindPackageShare("anubis_command_center"), "config", "command_center.yaml"]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "vps_url",
            description="REQUIRED, no default on purpose: e.g. "
                        "ws://1.2.3.4:8090/ws/robot or "
                        "wss://your-domain.example:8090/ws/robot once TLS "
                        "is set up on the VPS. Deliberately not in "
                        "config/command_center.yaml either -- see that "
                        "file's comment for why.",
        ),
        DeclareLaunchArgument(
            "map_dir",
            default_value=os.path.expanduser("~/anubis_maps/default"),
            description="Active map directory (map.pcd/map.pgm/map.yaml). "
                        "Matches anubis_bringup's ANUBIS_MAP_DIR convention. "
                        "Also deliberately not in command_center.yaml -- "
                        "YAML has no shell variable expansion, so a literal "
                        "$HOME there would not resolve; this launch "
                        "argument uses Python's os.path.expanduser() "
                        "instead, same as the node's own parameter default.",
        ),

        Node(
            package="anubis_command_center",
            executable="web_bridge_node",
            name="web_bridge_node",
            output="screen",
            parameters=[
                config_path,
                {
                    "vps_url": LaunchConfiguration("vps_url"),
                    "map_dir": LaunchConfiguration("map_dir"),
                },
            ],
        ),
    ])
