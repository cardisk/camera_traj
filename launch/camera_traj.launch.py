import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Directories
    pkg_camera_traj = get_package_share_directory('camera_traj')

    # Arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='False')
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic', default_value='/zed/zed_node/depth/depth_registered')
    yolo_topic_arg = DeclareLaunchArgument(
        'yolo_topic', default_value='/cone_detection/output')
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic', default_value='/zed/zed_node/depth/camera_info')
    rolling_map_safety_threshold_arg = DeclareLaunchArgument(
        'rolling_map_safety_threshold', default_value='0.5')
    rolling_map_ema_filter_alpha_arg = DeclareLaunchArgument(
        'rolling_map_ema_filter_alpha', default_value='0.6')
    world_frame_arg = DeclareLaunchArgument(
        'world_frame', default_value='odom')
    car_frame_arg = DeclareLaunchArgument(
        'car_frame', default_value='zed_camera_link')
    cull_distance_behind_arg = DeclareLaunchArgument(
        'cull_distance_behind', default_value='-2.0')
    cull_distance_max_arg = DeclareLaunchArgument(
        'cull_distance_max', default_value='10.0')
    rolling_map_debug_active_arg = DeclareLaunchArgument(
        'rolling_map_debug_active', default_value='True')
    rolling_map_debug_topic_arg = DeclareLaunchArgument(
        'rolling_map_debug_topic', default_value='/camera_traj/debug/rolling_map')

    # Node
    camera_traj_node = Node(
        package='camera_traj',
        executable='camera_traj',
        name='camera_traj',
        output='screen',
        respawn=True,
        parameters=[{
            'use_sim_time':                 LaunchConfiguration('use_sim_time'),
            'depth_topic':                  LaunchConfiguration('depth_topic'),
            'yolo_topic':                   LaunchConfiguration('yolo_topic'),
            'camera_info_topic':            LaunchConfiguration('camera_info_topic'),
            'rolling_map_safety_threshold': LaunchConfiguration('rolling_map_safety_threshold'),
            'rolling_map_ema_filter_alpha': LaunchConfiguration('rolling_map_ema_filter_alpha'),
            'world_frame':                  LaunchConfiguration('world_frame'),
            'car_frame':                    LaunchConfiguration('car_frame'),
            'cull_distance_behind':         LaunchConfiguration('cull_distance_behind'),
            'cull_distance_max':            LaunchConfiguration('cull_distance_max'),
            'rolling_map_debug_active':     LaunchConfiguration('rolling_map_debug_active'),
            'rolling_map_debug_topic':      LaunchConfiguration('rolling_map_debug_topic'),
        }]
    )

    return LaunchDescription([
        use_sim_time_arg,
        depth_topic_arg,
        yolo_topic_arg,
        camera_info_topic_arg,
        rolling_map_safety_threshold_arg,
        rolling_map_ema_filter_alpha_arg,
        world_frame_arg,
        car_frame_arg,
        cull_distance_behind_arg,
        cull_distance_max_arg,
        rolling_map_debug_active_arg,
        rolling_map_debug_topic_arg,
        camera_traj_node
    ])
