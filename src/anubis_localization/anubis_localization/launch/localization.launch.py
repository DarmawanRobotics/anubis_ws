import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def _rviz_config_path():
    return os.path.join(
        get_package_share_directory('anubis_localization'),
        'rviz', 'localization_ros2.rviz'
    )


# Parameters that live in config/config.yaml and may be overridden from the
# command line. Each declares an empty default meaning "not specified": the
# value from config.yaml is then used untouched. Only explicitly passed
# arguments are injected as node parameters, so config.yaml stays the single
# source of truth.
_OVERRIDABLE = (
    ('points_topic', str),
    ('enable_vertical_velocity_prediction', bool),
    ('ndt_score_threshold', float),
    ('localization_rejection_timeout_s', float),
    ('localization_recovery_valid_frames', int),
    ('gl_online_shadow_enabled', bool),
    ('gl_recall_source', str),
    ('gl_confirm_frames', int),
    ('gl_probe_period_s', float),
    ('gl_attempt_timeout_s', float),
    ('retrieval_topk', int),
    ('retrieval_fallback_to_grid', bool),
    ('retrieval_nms_xy', float),
    ('retrieval_max_distance', float),
    ('retrieval_min_points', int),
    ('retrieval_yaw_half_range_deg', float),
    ('retrieval_yaw_step_deg', float),
    ('retrieval_search_radius_xy', float),
    ('retrieval_seed_stride_xy', float),
    ('auto_confirm_enabled', bool),
    ('auto_confirm_verify_timeout_s', float),
    ('per_call_deadline_ms', float),
    ('episode_timeout_s', float),
    ('level0_gravity_max_age_ms', float),
    ('flat_single_level_map', bool),
    ('m2b_approval_id', str),
    ('dump_candidates_csv', bool),
    ('candidates_csv_path', str),
    ('candidates_csv_episode_id', str),
)


def _coerce(raw, kind):
    if kind is bool:
        return raw.strip().lower() in ('1', 'true', 'yes', 'on')
    if kind is float:
        return float(raw)
    if kind is int:
        return int(raw)
    return raw


def _launch_setup(context, *args, **kwargs):
    localization_dir = get_package_share_directory('anubis_localization')
    config_file = os.path.join(localization_dir, 'config', 'config.yaml')

    overrides = {
        'initial_pcd_map_path': LaunchConfiguration('pcd_map_path').perform(context),
        'use_sim_time': _coerce(
            LaunchConfiguration('use_sim_time').perform(context), bool),
    }

    # Keep the old launch argument usable without allowing the config-file
    # default for the new name to look like a conflicting explicit request.
    # When only the alias is supplied, pass the same value under both names;
    # when both are supplied, preserve both values so the node can fail closed
    # on disagreement.
    official_runtime = LaunchConfiguration(
        'gl_episode_runtime_enabled').perform(context)
    legacy_runtime = LaunchConfiguration('gl_auto_init_enabled').perform(context)
    if official_runtime != '':
        overrides['gl_episode_runtime_enabled'] = _coerce(official_runtime, bool)
    if legacy_runtime != '':
        legacy_value = _coerce(legacy_runtime, bool)
        overrides['gl_auto_init_enabled'] = legacy_value
        if official_runtime == '':
            overrides['gl_episode_runtime_enabled'] = legacy_value

    for name, kind in _OVERRIDABLE:
        raw = LaunchConfiguration(name).perform(context)
        if raw == '':
            continue  # not specified on the command line -> keep config.yaml value
        overrides[name] = _coerce(raw, kind)

    return [
        Node(
            package='anubis_localization',
            executable='localization_node',
            name='localization',
            output='screen',
            parameters=[config_file, overrides]
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'pcd_map_path',
            default_value='',
            description='Optional absolute PCD path loaded by localization at startup'
        ),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'use_rviz', default_value='false',
            description='Open RViz with localization_ros2.rviz pre-configured',
        ),
        DeclareLaunchArgument('points_topic', default_value=''),
        DeclareLaunchArgument(
            'enable_vertical_velocity_prediction', default_value=''),
        DeclareLaunchArgument('ndt_score_threshold', default_value=''),
        DeclareLaunchArgument(
            'localization_rejection_timeout_s', default_value=''),
        DeclareLaunchArgument(
            'localization_recovery_valid_frames', default_value=''),
        # Empty default = "not specified": config/config.yaml wins. Pass a value
        # explicitly (e.g. auto_confirm_enabled:=true) only to override it.
        DeclareLaunchArgument('gl_online_shadow_enabled', default_value=''),
        DeclareLaunchArgument('gl_recall_source', default_value=''),
        DeclareLaunchArgument(
            'gl_episode_runtime_enabled', default_value=''),
        # Deprecated compatibility alias. Prefer gl_episode_runtime_enabled.
        DeclareLaunchArgument('gl_auto_init_enabled', default_value=''),
        DeclareLaunchArgument('gl_confirm_frames', default_value=''),
        DeclareLaunchArgument('gl_probe_period_s', default_value=''),
        DeclareLaunchArgument('gl_attempt_timeout_s', default_value=''),
        DeclareLaunchArgument('retrieval_topk', default_value=''),
        DeclareLaunchArgument('retrieval_fallback_to_grid', default_value=''),
        DeclareLaunchArgument('retrieval_nms_xy', default_value=''),
        DeclareLaunchArgument('retrieval_max_distance', default_value=''),
        DeclareLaunchArgument('retrieval_min_points', default_value=''),
        DeclareLaunchArgument('retrieval_yaw_half_range_deg', default_value=''),
        DeclareLaunchArgument('retrieval_yaw_step_deg', default_value=''),
        DeclareLaunchArgument('retrieval_search_radius_xy', default_value=''),
        DeclareLaunchArgument('retrieval_seed_stride_xy', default_value=''),
        DeclareLaunchArgument('auto_confirm_enabled', default_value=''),
        DeclareLaunchArgument('auto_confirm_verify_timeout_s', default_value=''),
        DeclareLaunchArgument('per_call_deadline_ms', default_value=''),
        DeclareLaunchArgument('episode_timeout_s', default_value=''),
        DeclareLaunchArgument('level0_gravity_max_age_ms', default_value=''),
        DeclareLaunchArgument('flat_single_level_map', default_value=''),
        DeclareLaunchArgument('m2b_approval_id', default_value=''),
        DeclareLaunchArgument('dump_candidates_csv', default_value=''),
        DeclareLaunchArgument('candidates_csv_path', default_value=''),
        DeclareLaunchArgument('candidates_csv_episode_id', default_value=''),

        # Localization node (parameters injected on demand in _launch_setup).
        OpaqueFunction(function=_launch_setup),

        # [PATCH -- removed a redundant/dangerous second TF publisher]
        # The original file had its OWN static_transform_publisher here
        # for base_link -> livox_frame, hardcoded to a DIFFERENT robot's
        # mounting values (xyz=[0.211,0,0.145]) -- the exact
        # duplicated-extrinsic problem anubis_ws was built to eliminate.
        # anubis_description already publishes this exact transform from
        # the URDF (the actual source of truth for this robot's real
        # mounting: xyz=[0.13011,-0.02329,0.17598], see
        # anubis_description/urdf/sensors.xacro). Run
        # `ros2 launch anubis_description description.launch.py`
        # alongside this launch file instead of re-publishing it here.

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', _rviz_config_path()],
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
    ])