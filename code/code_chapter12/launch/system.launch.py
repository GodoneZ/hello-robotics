"""一个 ROS 系统：AMCL 或 SLAM 二选一 + Nav2 + MoveIt2 + 互锁/感知/RViz。"""

from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from nav2_common.launch import RewrittenYaml


def moveit_config():
    return (
        MoveItConfigsBuilder("g2_right_arm", package_name="g2_chapter12")
        .robot_description(file_path="urdf/g2_right_arm.urdf.xacro")
        .robot_description_semantic(file_path="srdf/g2_right_arm.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"], load_all=False)
        .to_moveit_configs()
    )


def generate_launch_description():
    root = Path(get_package_share_directory("g2_chapter12"))
    nav = Path(get_package_share_directory("nav2_bringup"))
    slam = LaunchConfiguration("slam")
    rviz = LaunchConfiguration("rviz")
    m = moveit_config()
    common = {"use_sim_time": True}
    # 不依赖 shell 当前目录，也不修改 /opt/ros 的默认行为树。
    params = RewrittenYaml(
        source_file=str(root / "config/nav2_params.yaml"),
        root_key="",
        param_rewrites={"default_nav_to_pose_bt_xml": str(root / "config/navigate_to_pose.xml")},
        convert_types=True,
    )
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(nav / "launch/localization_launch.py")),
        condition=UnlessCondition(slam),
        launch_arguments={
            "map": LaunchConfiguration("map"),
            "params_file": params,
            "use_sim_time": "true",
            "autostart": "true",
            "use_composition": "False",
        }.items(),
    )
    mapping = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        condition=IfCondition(slam),
        parameters=[str(root / "config/slam.yaml"), common],
        output="screen",
    )
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(nav / "launch/navigation_launch.py")),
        launch_arguments={
            "params_file": params,
            "use_sim_time": "true",
            "autostart": "true",
            "use_composition": "False",
        }.items(),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("slam", default_value="false", choices=["true", "false"]),
            DeclareLaunchArgument("map", default_value=str(root / "maps/room.yaml")),
            DeclareLaunchArgument("rviz", default_value="true"),
            localization,
            mapping,
            navigation,
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[m.robot_description, common],
                output="screen",
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                parameters=[m.to_dict(), common, {"publish_robot_description_semantic": True}],
                output="screen",
            ),
            Node(
                package="g2_chapter12",
                executable="controller.py",
                parameters=[common],
                output="screen",
            ),
            Node(
                package="g2_chapter12",
                executable="perception.py",
                parameters=[common],
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                condition=IfCondition(rviz),
                arguments=["-d", str(root / "rviz/mobile.rviz")],
                parameters=[m.to_dict(), common],
                output="screen",
            ),
        ]
    )
