"""显式启动一次任务；system 启动不自动让机器人运动。"""

import importlib.util
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    path = Path(get_package_share_directory("g2_chapter12")) / "launch/system.launch.py"
    spec = importlib.util.spec_from_file_location("chapter12_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return LaunchDescription(
        [
            Node(
                package="g2_chapter12",
                executable="mission",
                output="screen",
                parameters=[module.moveit_config().to_dict(), {"use_sim_time": True}],
            )
        ]
    )
