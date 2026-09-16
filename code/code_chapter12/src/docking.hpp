#pragma once
// 无 ROS 依赖的近距离停靠几何；路径仅交给 Nav2 DWB 执行，绝不直接驱动底盘。
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace docking
{
struct Pose
{
  double x, y, yaw;
};
inline double wrap(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}
inline double distance(const Pose & a, const Pose & b)
{
  return std::hypot(a.x - b.x, a.y - b.y);
}
inline bool within(const Pose & a, const Pose & b)
{
  return distance(a, b) <= .06 && std::abs(wrap(a.yaw - b.yaw)) <= .05;
}
inline std::vector<Pose> short_path(const Pose & start, const Pose & goal)
{
  for (double v : {start.x, start.y, start.yaw, goal.x, goal.y, goal.yaw}) {
    if (!std::isfinite(v)) {
      throw std::runtime_error("停靠位姿不是有限数值");
    }
  }
  const double length = distance(start, goal);
  const double turn = wrap(goal.yaw - start.yaw);
  // 这里只做已到桌旁的微调，不替代全局规划，不触发桌旁旋转/后退恢复行为。
  if (length > .30 || std::abs(turn) > .35) {
    throw std::runtime_error("超出近距离停靠范围 30 cm / 0.35 rad，拒绝局部直线路径");
  }
  const int segments = std::max(1, static_cast<int>(std::ceil(length / .025)));
  std::vector<Pose> path;
  for (int i = 0; i <= segments; ++i) {
    const double t = static_cast<double>(i) / segments;
    path.push_back({start.x + t * (goal.x - start.x), start.y + t * (goal.y - start.y),
      wrap(start.yaw + t * turn)});
  }
  path.back() = goal;  // 终点就是配置中的 odom 停靠点，不做栅格量化或容差替换。
  return path;
}
}  // namespace docking
