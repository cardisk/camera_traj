import os
import sys
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def get_command_overrides(yaml_file_path):
    """
    Dynamically extracting the parameters overrides passed
    through the command line
    """
    valid_keys = set()

    try:
        with open(yaml_file_path, 'r') as f:
            config = yaml.safe_load(f)
            node_params = config.get('camera_traj', {}).get('ros__parameters', {})
            valid_keys = set(node_params.keys())
    except Exception as e:
        print(f"\n[ERROR] Could not read YAML file: {e}\n")
        return {}

    overrides = {}
    for arg in sys.argv:
        if ":=" in arg:
            key, val = arg.split(":=", 1)

            if key not in valid_keys:
                print(f"\n[WARNING] '{key}' not present inside the YAML file, ignoring...\n")
                continue

            if val.lower() == 'true':
                overrides[key] = True

            elif val.lower() == 'false':
                overrides[key] = False

            else:
                try:
                    if '.' in val:
                        overrides[key] = float(val)

                    else:
                        overrides[key] = int(val)

                except ValueError:
                    overrides[key] = val

    return overrides


def generate_launch_description():
    # Directories
    pkg_camera_traj = get_package_share_directory('camera_traj')

    config_file = os.path.join(pkg_camera_traj, 'config', 'camera_traj_params.yaml')
    overrides = get_command_overrides(config_file)

    dyn_launch_args = [
        DeclareLaunchArgument(key, default_value=str(val))
        for key, val in overrides.items()
    ]

    # Node
    camera_traj_node = Node(
        package='camera_traj',
        executable='camera_traj',
        name='camera_traj',
        output='screen',
        respawn=True,
        parameters=[
            config_file,
            overrides
        ]
    )

    # 2026-04-05: because of a known bug inside zed-ros2-wrapper,
    # the depth image will have the misspelled frame_id name.
    # This will create a static identity transformation because
    # the frames are the same but with different names.
    zed_tf_bridge_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='zed_depth_frame_bridge',
        arguments=[
            '0', '0', '0', '0', '0', '0',
            'zed_left_camera_optical_frame',
            'zed_left_camera_frame_optical'
        ],
        output='screen'
    )

    return LaunchDescription(
        dyn_launch_args + [
            camera_traj_node,
            zed_tf_bridge_node
        ]
    )
