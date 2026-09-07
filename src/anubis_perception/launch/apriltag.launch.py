from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_path = PathJoinSubstitution(
        [FindPackageShare("anubis_perception"), "config", "apriltag_params.yaml"]
    )
    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare("anubis_perception"), "rviz", "perception.rviz"]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "camera_image_topic",
            default_value="/sensors/camera/color/image_raw",
            description="Source RGB image topic from anubis_sensors",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/sensors/camera/color/camera_info",
            description="Source CameraInfo topic from anubis_sensors",
        ),
        DeclareLaunchArgument(
            "enable_debug_image",
            default_value="true",
            description="Publish an annotated debug image with tag outlines/IDs drawn on it (apriltag_draw)",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            description="Open RViz with TF + the debug image pre-configured",
        ),
        Node(
            package="apriltag_ros",
            executable="apriltag_node",
            name="apriltag",
            namespace="apriltag",
            output="screen",
            parameters=[config_path],
            remappings=[
                ("image_rect", LaunchConfiguration("camera_image_topic")),
                ("camera_info", LaunchConfiguration("camera_info_topic")),
                ("detections", "/perception/apriltag/detections"),
            ],
        ),

        # ---- Live debug view: apriltag_draw ----
        Node(
            package="apriltag_draw",
            executable="apriltag_draw",
            name="apriltag_draw",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_debug_image")),
            remappings=[
                ("tags", "/perception/apriltag/detections"),
                ("image", LaunchConfiguration("camera_image_topic")),
                ("image_tags", "/perception/apriltag/image_tags"),
            ],
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config_path],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        ),
    ])
