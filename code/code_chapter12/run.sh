#!/usr/bin/env bash
# 系统依赖是外部安装；所有课程代码、配置、地图和 USD 均为本章本地文件。
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-12}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$ROOT/config/fastdds.xml"
export ROS_LOG_DIR="$ROOT/log/runtime"
mkdir -p "$ROS_LOG_DIR"
unset PYTHONNOUSERSITE
CMD="${1:-help}"; if [[ $# -gt 0 ]]; then shift; fi
if [[ "$CMD" == sim ]]; then
  ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/robot/isaac-sim}"
  [[ -x "$ISAAC_SIM_ROOT/python.sh" ]] || { echo '请设置 ISAAC_SIM_ROOT'; exit 1; }
  unset PYTHONPATH OLD_PYTHONPATH PYTHONHOME CONDA_PREFIX
  export ROS_DISTRO=humble ROS_VERSION=2 ROS_PYTHON_VERSION=3
  export AMENT_PREFIX_PATH=/opt/ros/humble
  export LD_LIBRARY_PATH="$ISAAC_SIM_ROOT/exts/isaacsim.ros2.bridge/humble/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  cd "$ROOT"
  exec "$ISAAC_SIM_ROOT/python.sh" -m sim.bridge "$@"
fi
# 固定系统 Python，避免 Conda 和 Isaac Python ABI 污染。
unset PYTHONHOME PYTHONPATH
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
[[ -f /opt/ros/humble/setup.bash ]] || { echo '需要 ROS 2 Humble'; exit 1; }
source /opt/ros/humble/setup.bash
cd "$ROOT"
case "$CMD" in
  build)
    /usr/bin/python3 -m g2_mobile.mapping
    colcon --log-base "$ROOT/log/build" build --base-paths "$ROOT" \
      --build-base "$ROOT/build" --install-base "$ROOT/install" \
      --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
    ;;
  test) /usr/bin/python3 -m unittest discover -s tests -v ;;
  system|mission|teleop|estop|reset|save-map|check)
    [[ -f "$ROOT/install/setup.bash" ]] || { echo '先运行 ./run.sh build'; exit 1; }
    if [[ "$CMD" == system || "$CMD" == mission || "$CMD" == teleop ]]; then
      for config in task.yaml nav2_params.yaml navigate_to_pose.xml; do
        cmp -s "$ROOT/config/$config" "$ROOT/install/g2_chapter12/share/g2_chapter12/config/$config" || {
          echo "$config 已修改：请重新 ./run.sh build，避免运行旧场景/导航参数"; exit 1;
        }
      done
    fi
    source "$ROOT/install/setup.bash"
    case "$CMD" in
      system) exec ros2 launch g2_chapter12 system.launch.py "$@" ;;
      mission) exec flock -n "$ROOT/log/command.lock" ros2 launch g2_chapter12 mission.launch.py "$@" ;;
      teleop) exec flock -n "$ROOT/log/command.lock" /usr/bin/python3 -m g2_mobile.teleop ;;
      estop) exec ros2 service call /estop std_srvs/srv/Trigger '{}' ;;
      reset) exec ros2 service call /reset_estop std_srvs/srv/Trigger '{}' ;;
      save-map) exec ros2 run nav2_map_server map_saver_cli -f "$ROOT/maps/slam_room" --ros-args -p use_sim_time:=true ;;
      check) exec /usr/bin/python3 tests/check_runtime.py "$@" ;;
    esac ;;
  *) echo '用法: ./run.sh {build|test|sim [--headless]|system [slam:=true] [rviz:=false]|mission|teleop|estop|reset|save-map|check}' ;;
esac
