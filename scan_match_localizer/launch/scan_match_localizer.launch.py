import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('scan_match_localizer')
    default_config = os.path.join(pkg_share, 'config', 'scan_match_localizer.yaml')

    declare_config_cmd = DeclareLaunchArgument(
        'config',
        default_value=default_config,
        description='Full path to the scan_match_localizer config file.')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true.')

    node = Node(
        package='scan_match_localizer',
        executable='scan_match_localizer_node',
        name='scan_match_localizer',
        output='screen',
        parameters=[
            LaunchConfiguration('config'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
    )

    return LaunchDescription([
        declare_config_cmd,
        declare_use_sim_time_cmd,
        node,
    ])
