from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'image_topic',
            default_value='/front_camera/image',
            description='Topic sensor_msgs/Image untuk input kamera',
        ),

        DeclareLaunchArgument(
            'engine_path',
            default_value='',
            description=(
                'Path absolut ke file .engine TensorRT '
                '(kosong = pakai default dari package share, '
                'models/yolo11s.engine)'
            ),
        ),

        DeclareLaunchArgument(
            'dist_threshold',
            default_value='150.0',
            description='Jarak (px) ke orang terdekat dianggap jauh',
        ),

        DeclareLaunchArgument(
            'time_threshold',
            default_value='2.0',
            description='Lama (detik) sendirian sebelum dianggap abandoned',
        ),

        Node(
            package='anubis_perception',
            executable='abandoned_detection_node',
            name='abandoned_detection_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'engine_path': LaunchConfiguration('engine_path'),
                'dist_threshold': LaunchConfiguration('dist_threshold'),
                'time_threshold': LaunchConfiguration('time_threshold'),
            }],
        ),
    ])