import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    parameter_config = os.path.join(get_package_share_directory(
        'anubis_mapping'), 'config', 'config.yaml')
    default_rviz_config_path = os.path.join(
        get_package_share_directory('anubis_mapping'), 'rviz', 'mapping.rviz')
    map_output_dir = LaunchConfiguration('map_output_dir')
    use_rviz = LaunchConfiguration('use_rviz')
    loop_enable = LaunchConfiguration('loop_enable')
    return LaunchDescription([
        DeclareLaunchArgument(
            'map_output_dir',
            default_value='',
            description='Directory for map.pcd, map.pgm/map.yaml, map_scd.bin, apriltag_anchors.yaml'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Whether to start RViz for mapping'
        ),
        DeclareLaunchArgument(
            'loop_enable',
            default_value='true',
            description='Enable loop closure during mapping'
        ),
        Node(
            package='anubis_mapping',
            executable='mapping_alg_node',
            parameters=[
                parameter_config,
                {'map_output_dir': map_output_dir,
                 'loop.enable': ParameterValue(loop_enable, value_type=bool)}
            ]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', default_rviz_config_path],
            condition=IfCondition(use_rviz)
        )
    ])
