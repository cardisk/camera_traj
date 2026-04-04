from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Directories
    _pkg_camera_traj = get_package_share_directory('camera_traj')

    # Flags
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='False')

    pure_local_trajectory_arg = DeclareLaunchArgument(
        'pure_local_trajectory', default_value='False')

    debug_output_arg = DeclareLaunchArgument(
        'debug_output', default_value='True')

    # Topics
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic', default_value='/zed/zed_node/depth/depth_registered')

    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic', default_value='/zed/zed_node/depth/camera_info')

    yolo_topic_arg = DeclareLaunchArgument(
        'yolo_topic', default_value='/cone_detection/output')

    rolling_map_debug_topic_arg = DeclareLaunchArgument(
        'rolling_map_debug_topic', default_value='/camera_traj/debug/rolling_map')

    trajectory_debug_topic_arg = DeclareLaunchArgument(
        'trajectory_debug_topic', default_value='/camera_traj/debug/trajectory')

    output_topic_arg = DeclareLaunchArgument(
        'output_topic', default_value='/camera_traj/output')

    # Settings
    rolling_map_safety_threshold_arg = DeclareLaunchArgument(
        'rolling_map_safety_threshold', default_value='1.5')

    rolling_map_ema_filter_alpha_arg = DeclareLaunchArgument(
        'rolling_map_ema_filter_alpha', default_value='0.3')

    rolling_map_hit_count_threshold_arg = DeclareLaunchArgument(
        'rolling_map_hit_count_threshold', default_value='3')

    cull_distance_behind_arg = DeclareLaunchArgument(
        'cull_distance_behind', default_value='-2.0')

    cull_distance_max_arg = DeclareLaunchArgument(
        'cull_distance_max', default_value='15.0')

    delaunay_min_distance_arg = DeclareLaunchArgument(
        'delaunay_min_distance', default_value='2.0')

    delaunay_max_distance_arg = DeclareLaunchArgument(
        'delaunay_max_distance', default_value='8.0')

    spline_smoothing_arg = DeclareLaunchArgument(
        'spline_smoothing', default_value='3.0')

    spline_degree_arg = DeclareLaunchArgument(
        'spline_degree', default_value='3')

    spline_sampling_resolution_arg = DeclareLaunchArgument(
        'spline_sampling_resolution', default_value='0.5')

    world_frame_arg = DeclareLaunchArgument(
        'world_frame', default_value='map')

    car_frame_arg = DeclareLaunchArgument(
        'car_frame', default_value='zed_camera_link')

    # Node
    camera_traj_node = Node(
        package='camera_traj',
        executable='camera_traj',
        name='camera_traj',
        output='screen',
        respawn=True,
        parameters=[{
            'use_sim_time':                    LaunchConfiguration('use_sim_time'),
            'pure_local_trajectory':           LaunchConfiguration('pure_local_trajectory'),
            'debug_output':                    LaunchConfiguration('debug_output'),
            'depth_topic':                     LaunchConfiguration('depth_topic'),
            'camera_info_topic':               LaunchConfiguration('camera_info_topic'),
            'yolo_topic':                      LaunchConfiguration('yolo_topic'),
            'rolling_map_debug_topic':         LaunchConfiguration('rolling_map_debug_topic'),
            'trajectory_debug_topic':          LaunchConfiguration('trajectory_debug_topic'),
            'output_topic':                    LaunchConfiguration('output_topic'),
            'rolling_map_safety_threshold':    LaunchConfiguration('rolling_map_safety_threshold'),
            'rolling_map_ema_filter_alpha':    LaunchConfiguration('rolling_map_ema_filter_alpha'),
            'rolling_map_hit_count_threshold': LaunchConfiguration('rolling_map_hit_count_threshold'),
            'cull_distance_behind':            LaunchConfiguration('cull_distance_behind'),
            'cull_distance_max':               LaunchConfiguration('cull_distance_max'),
            'delaunay_min_distance':           LaunchConfiguration('delaunay_min_distance'),
            'delaunay_max_distance':           LaunchConfiguration('delaunay_max_distance'),
            'spline_smoothing':                LaunchConfiguration('spline_smoothing'),
            'spline_degree':                   LaunchConfiguration('spline_degree'),
            'spline_sampling_resolution':      LaunchConfiguration('spline_sampling_resolution'),
            'world_frame':                     LaunchConfiguration('world_frame'),
            'car_frame':                       LaunchConfiguration('car_frame'),
        }]
    )

    return LaunchDescription([
        use_sim_time_arg,
        pure_local_trajectory_arg,
        debug_output_arg,
        depth_topic_arg,
        camera_info_topic_arg,
        yolo_topic_arg,
        rolling_map_debug_topic_arg,
        trajectory_debug_topic_arg,
        output_topic_arg,
        rolling_map_safety_threshold_arg,
        rolling_map_ema_filter_alpha_arg,
        rolling_map_hit_count_threshold_arg,
        cull_distance_behind_arg,
        cull_distance_max_arg,
        delaunay_min_distance_arg,
        delaunay_max_distance_arg,
        spline_smoothing_arg,
        spline_degree_arg,
        spline_sampling_resolution_arg,
        world_frame_arg,
        car_frame_arg,
        camera_traj_node
    ])
