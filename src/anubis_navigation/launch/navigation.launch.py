"""anubis_navigation / navigation.launch.py

Thin wrapper around the standard nav2_bringup bringup_launch.py, with
Anubis's tuned params.yaml. Deliberately does NOT vendor or rebuild any
Nav2 source -- darmawan_ws had 14 forked Nav2 packages
(navigo_bt_navigator, navigo_costmap_2d, navigo_mppi_controller, etc.)
that turned out to be unmodified renamed copies of the real thing (zero
files carried this codebase's own "[PATCH]"/dated-comment convention
anywhere), so there was nothing worth preserving in the fork itself --
only the actual tuned config/params.yaml had real value, which is what
this package carries. Install real Nav2 via apt:

    sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

Requires anubis_description, anubis_sensors, anubis_control, and
anubis_localization already running (this package supplies none of
those -- just Nav2 itself, configured for this robot).

RViz: rather than write a custom config from scratch with no working
example to adapt from (unlike anubis_mapping/anubis_localization, which
had real .rviz files to start from), this reuses nav2_bringup's own
rviz_launch.py + nav2_default_view.rviz directly -- it already shows
exactly what Nav2 users need (costmaps, path, goal-pose tool, footprint)
and is guaranteed to load correctly.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")
    anubis_navigation_dir = get_package_share_directory("anubis_navigation")

    default_params_file = os.path.join(
        anubis_navigation_dir, "config", "nav2_params.yaml"
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params_file,
            description="Full path to the Nav2 params.yaml to use",
        ),
        DeclareLaunchArgument(
            "map",
            default_value="",
            description="Full path to a map.yaml (from anubis_mapping's pcd2grid_node output). "
                        "Leave empty to only use costmaps fed by anubis_localization's live map, "
                        "without map_server's static /map topic.",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("autostart", default_value="true"),
        DeclareLaunchArgument(
            "use_rviz", default_value="false",
            description="Open RViz with Nav2's own standard displays pre-configured",
        ),

        # Standard nav2_bringup -- this is real, unmodified Nav2, matching
        # the goal of not building any of it from source.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, "launch", "bringup_launch.py")
            ),
            launch_arguments={
                "map": LaunchConfiguration("map"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "params_file": LaunchConfiguration("params_file"),
                "autostart": LaunchConfiguration("autostart"),
            }.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, "launch", "rviz_launch.py")
            ),
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        ),
    ])
