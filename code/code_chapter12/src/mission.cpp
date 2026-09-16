// 编排层：只调用 ROS 标准接口，不直接设置机器人/物体位置。
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <nav2_msgs/action/follow_path.hpp>
#include <nav_msgs/msg/path.hpp>
#include "docking.hpp"
#include <control_msgs/action/gripper_command.hpp>
#include <moveit_msgs/action/move_group.hpp>
#include <moveit_msgs/action/execute_trajectory.hpp>
#include <moveit_msgs/srv/get_cartesian_path.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include "scene_client.hpp"
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include <thread>
using namespace std::chrono_literals;
using Nav = nav2_msgs::action::NavigateToPose;
using Follow = nav2_msgs::action::FollowPath;
using Grip = control_msgs::action::GripperCommand;
using Plan = moveit_msgs::action::MoveGroup;
using Execute = moveit_msgs::action::ExecuteTrajectory;
using Group = moveit::planning_interface::MoveGroupInterface;
using Scene = SceneClient;
class PlanningFailure : public std::runtime_error
{
  using std::runtime_error::runtime_error;
};

class Mission
{
public:
  explicit Mission(rclcpp::Node::SharedPtr node)
  : node_(node), tf_(std::make_shared<tf2_ros::Buffer>(node->get_clock())), listener_(*tf_)
  {
    cfg_ = YAML::LoadFile(
      ament_index_cpp::get_package_share_directory(
        "g2_chapter12") + "/config/task.yaml");
    mode_pub_ = node_->create_publisher<std_msgs::msg::String>("/mode_request", 10);
    status_pub_ = node_->create_publisher<std_msgs::msg::String>(
      "/task_status", rclcpp::QoS(
        1).transient_local());
    mode_sub_ = node_->create_subscription<std_msgs::msg::String>(
      "/motion_mode", 10, [this](std_msgs::msg::String::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(
          mutex_);
        actual_mode_ = msg->data;
        mode_at_ = std::chrono::steady_clock::now();
      });
    poses_sub_ = node_->create_subscription<geometry_msgs::msg::PoseArray>(
      "/grasp_candidates", 10, [this](geometry_msgs::msg::PoseArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        poses_ = *msg;
      });
    center_sub_ = node_->create_subscription<geometry_msgs::msg::PointStamped>(
      "/target_center", 10, [this](geometry_msgs::msg::PointStamped::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        center_ = *msg;
      });
    heartbeat_ = node_->create_wall_timer(
      100ms, [this] {
        std::lock_guard<std::mutex> lock(mutex_);
        std_msgs::msg::String msg;
        msg.data = requested_mode_;
        mode_pub_->publish(msg);
      });
    path_sub_ = node_->create_subscription<nav_msgs::msg::Path>(
      "/plan", 10, [this](nav_msgs::msg::Path::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        global_path_ = *msg;
      });
    dock_path_pub_ = node_->create_publisher<nav_msgs::msg::Path>(
      "/dock_path", rclcpp::QoS(1).transient_local());
    nav_ = rclcpp_action::create_client<Nav>(node_, "/navigate_to_pose");
    follow_ = rclcpp_action::create_client<Follow>(node_, "/follow_path");
    grip_ = rclcpp_action::create_client<Grip>(node_, "/gripper_controller/gripper_cmd");
    plan_ = rclcpp_action::create_client<Plan>(node_, "/move_action");
    execute_ = rclcpp_action::create_client<Execute>(node_, "/execute_trajectory");
    cartesian_ =
      node_->create_client<moveit_msgs::srv::GetCartesianPath>("/compute_cartesian_path");
  }

  ~Mission()
  {
    idle();
  }

  void run()
  {
    state("WAIT_READY");
    if (!plan_->wait_for_action_server(60s) || !nav_->wait_for_action_server(60s) ||
      !follow_->wait_for_action_server(30s) || !execute_->wait_for_action_server(30s) ||
      !grip_->wait_for_action_server(30s)) throw std::runtime_error("Nav2/MoveIt2/夹爪 action 未就绪");
    const auto tf_deadline = std::chrono::steady_clock::now() + 15s;
    bool tf_ready=false;
    while (rclcpp::ok() && std::chrono::steady_clock::now()<tf_deadline) {
      try { auto t=tf_->lookupTransform("odom","arm_base_link",tf2::TimePointZero);
        double age=(node_->now()-rclcpp::Time(t.header.stamp)).seconds();
        if (node_->now().nanoseconds()>0 && age>=-.1 && age<=.3) { tf_ready=true; break; }
      } catch (...) {}
      std::this_thread::sleep_for(50ms);
    }
    if (!tf_ready) throw std::runtime_error("等待新鲜 odom -> arm_base_link TF 超时");
    Group arm(node_, "right_arm", tf_);
    arm.setPoseReferenceFrame("odom");
    arm.setEndEffectorLink("gripper_r_center_link"); arm.setPlanningTime(5.); arm.setNumPlanningAttempts(3);
    arm.setPlannerId("RRTConnectkConfigDefault"); arm.setMaxVelocityScalingFactor(.20); arm.setMaxAccelerationScalingFactor(.15);
    arm.setGoalJointTolerance(.01); arm.setGoalPositionTolerance(.004); arm.setGoalOrientationTolerance(.04);
    if (!arm.getCurrentState(15.)) throw std::runtime_error("没有完整 /joint_states 与移动基座 TF");
    Scene scene(node_); state("STOW"); mode("ARM"); add_scene(scene);
    joints(arm,cfg_["home"].as<std::vector<double>>()); gripper(.785);

    auto pick = [&](const std::string & label, double table_top) {
      state(label + "_PERCEIVE"); auto candidates=fresh_poses();
      auto target=target_object(candidates.poses.front());
      if (!scene.applyCollisionObject(target)) throw std::runtime_error("目标碰撞物体同步失败");
      begin_grasp_debug(scene, arm);
      try {
      geometry_msgs::msg::Pose grasp,pre; bool found=false;
      for (const auto & c:candidates.poses) { grasp=c; pre=c; pre.position.z+=.12;
        try { allow_touch(scene,false); pose(arm,pre); state(label+"_APPROACH"); allow_touch(scene,true); cartesian(arm,grasp); found=true; break; }
        catch (const PlanningFailure & e) { if (!healthy("ARM")) throw; RCLCPP_WARN(node_->get_logger(),"候选不可达：%s",e.what()); }
      }
      if (!found) throw std::runtime_error(label+" 所有预抓取候选不可达");
      state(label+"_CLOSE"); gripper(0.);
      moveit_msgs::msg::AttachedCollisionObject a; a.link_name="gripper_r_center_link"; a.touch_links={"gripper_r_center_link"}; a.object=target;
      if (!scene.applyAttachedCollisionObject(a)) throw std::runtime_error("附着碰撞体失败");
      state(label+"_LIFT"); auto lift=grasp; lift.position.z+=.12; cartesian(arm,lift);
      verify([table_top](const auto & p){return p.z>table_top+.10;},1.,label+" 无法确认苹果抬升：缺少连续有效 RGB-D 观测（也需检查遮挡/漏检）");
      // 持物先回 home，再恢复碰撞检查，之后才允许移动底盘。
      state(label + "_HOME"); joints(arm,cfg_["home"].as<std::vector<double>>());
      end_grasp_debug(scene);
      return std::make_pair(grasp,lift);
      } catch (...) {
        try { end_grasp_debug(scene); }
        catch (const std::exception & e) {
          RCLCPP_ERROR(node_->get_logger(), "恢复碰撞矩阵失败，停止任务并重启 system：%s", e.what());
        }
        throw;
      }
    };
    auto place = [&](const std::vector<double> & slot, const geometry_msgs::msg::Pose & lift, const std::string & label) {
      // 动作前再确认控制器持物互锁；掉物不能借 NAV->ARM 切换绕过。
      mode("ARM");
      auto tray=cfg_["tray"].as<std::vector<double>>(); auto above=lift;
      above.position.x=slot[0]; above.position.y=slot[1]; above.position.z=tray[2]+.22;
      state(label+"_TRANSFER"); pose(arm,above); auto p=above; p.position.z=tray[2]+tray[5]/2+.045;
      state(label+"_PLACE"); cartesian(arm,p); gripper(.785);
      moveit_msgs::msg::AttachedCollisionObject a; a.link_name="gripper_r_center_link"; a.touch_links={"gripper_r_center_link"}; a.object=target_object(p); a.object.operation=moveit_msgs::msg::CollisionObject::REMOVE;
      if (!scene.applyAttachedCollisionObject(a)) throw std::runtime_error("分离碰撞体失败");
      scene.removeCollisionObjects({"target"}); state(label+"_RETREAT"); cartesian(arm,above);
      state(label+"_VERIFY_PLACE"); verify([slot, tray](const auto & x){
        const double board_z = tray[2] + tray[5] / 2.0;
        return std::abs(x.x-slot[0])<.10 && std::abs(x.y-slot[1])<.10 &&
          x.z > board_z && x.z < board_z + .075;
      },1.5,label+" 未稳定留在板子上");
      allow_touch(scene,false);
    };

    state("NAVIGATE_SOURCE"); navigate_to(cfg_["source_dock"].as<std::vector<double>>());
    const auto source_top=cfg_["source_table"].as<std::vector<double>>()[2]+cfg_["source_table"].as<std::vector<double>>()[5]/2;
    auto first=pick("SOURCE",source_top);
    state("CARRY"); joints(arm,cfg_["home"].as<std::vector<double>>());
    navigate();
    place(cfg_["place_slots"][0].as<std::vector<double>>(),first.second,"SOURCE");
    const auto dest_top=cfg_["table"].as<std::vector<double>>()[2]+cfg_["table"].as<std::vector<double>>()[5]/2;
    auto second=pick("DESTINATION",dest_top);
    place(cfg_["place_slots"][1].as<std::vector<double>>(),second.second,"DESTINATION");
    state("STOW"); joints(arm,cfg_["home"].as<std::vector<double>>()); idle(); state("DONE");
  }

  void fail(const std::string & reason)
  {
    if (!rclcpp::ok()) {
      return;
    }
    idle();
    state("FAILED: " + reason);
    // 取消仍在服务端的 action，禁止失败后继续运动；夹爪保持、不盲目松开。
    nav_->async_cancel_all_goals();
    follow_->async_cancel_all_goals();
    plan_->async_cancel_all_goals();
    execute_->async_cancel_all_goals();
    grip_->async_cancel_all_goals();
    if (grasp_debug_active_) {
      try { Scene scene(node_); end_grasp_debug(scene); }
      catch (const std::exception & e) {
        RCLCPP_ERROR(node_->get_logger(), "请重启 system 恢复碰撞矩阵：%s", e.what());
      }
    }
  }

private:
  void idle()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_mode_ = "IDLE";
    }
    if (rclcpp::ok()) {
      std_msgs::msg::String msg;
      msg.data = "IDLE";
      mode_pub_->publish(msg);
    }
  }
  void state(const std::string & value)
  {
    std_msgs::msg::String msg;
    msg.data = value;
    status_pub_->publish(msg);
    RCLCPP_INFO(node_->get_logger(), "[TASK] %s", value.c_str());
  }
  bool healthy(const std::string & mode)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return actual_mode_ == mode && std::chrono::steady_clock::now() - mode_at_ < 1s;
  }
  void mode(const std::string & wanted)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requested_mode_ = wanted;
    }
    const auto deadline = std::chrono::steady_clock::now() + 20s;
    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
      if (healthy(wanted)) {
        return;
      }
      std::this_thread::sleep_for(50ms);
    }
    throw std::runtime_error("模式互锁未满足：" + wanted);
  }
  template<class Action>
  typename rclcpp_action::ClientGoalHandle<Action>::WrappedResult call(
    typename rclcpp_action::Client<Action>::SharedPtr client, const typename Action::Goal & goal,
    double timeout, const std::string & action_name)
  {
    auto accepted = client->async_send_goal(goal);
    if (accepted.wait_for(5s) != std::future_status::ready) {
      client->async_cancel_all_goals();
      throw std::runtime_error(action_name + " 接收超时");
    }
    auto handle = accepted.get();
    if (!handle) {
      throw std::runtime_error(action_name + " 被拒绝");
    }
    auto result = client->async_get_result(handle);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout);
    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
      if (result.wait_for(50ms) == std::future_status::ready) {
        auto value = result.get();
        if (value.code != rclcpp_action::ResultCode::SUCCEEDED) {
          if constexpr (std::is_same_v<Action, Plan>) {
            throw PlanningFailure("MoveIt 规划 action 未成功，error_code=" +
              (value.result ? std::to_string(value.result->error_code.val) : "无结果") +
              "（关闭碰撞检查仍要求有效 IK 和关节限位）");
          }
          const std::string status = value.code == rclcpp_action::ResultCode::ABORTED ?
            "ABORTED" : value.code == rclcpp_action::ResultCode::CANCELED ? "CANCELED" : "UNKNOWN";
          throw std::runtime_error(action_name + " 返回 " + status +
            "；请检查对应服务端日志");
        }
        return value;
      }
      std::string wanted;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        wanted = requested_mode_;
      }
      if (!healthy(wanted)) {
        break;
      }
    }
    auto canceled = client->async_cancel_goal(handle);
    canceled.wait_for(2s);
    throw std::runtime_error(action_name + " 超时/互锁丢失，已请求取消");
  }
  geometry_msgs::msg::TransformStamped fresh_base()
  {
    auto base = tf_->lookupTransform("odom", "base_link", tf2::TimePointZero,
        tf2::durationFromSec(3.));
    const double age = (node_->now() - rclcpp::Time(base.header.stamp)).seconds();
    if (age < -.1 || age > .3) {
      throw std::runtime_error("停靠复核 TF 过期或时钟不一致");
    }
    return base;
  }
  docking::Pose base_pose()
  {
    const auto base = fresh_base();
    const auto & p = base.transform.translation;
    const auto & q = base.transform.rotation;
    return {p.x, p.y, std::atan2(2 * (q.w * q.z + q.x * q.y),
        1 - 2 * (q.y * q.y + q.z * q.z))};
  }
  void navigate()
  {
    navigate_to(cfg_["dock"].as<std::vector<double>>());
  }
  void navigate_to(const std::vector<double> & dock)
  {
    const docking::Pose target{dock.at(0), dock.at(1), dock.at(2)};
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration<double>(cfg_["nav_timeout"].as<double>());
    auto remaining = [&]() {
        double seconds = std::chrono::duration<double>(
          deadline - std::chrono::steady_clock::now()).count();
        if (seconds <= 0.) {
          throw std::runtime_error("停靠总时限耗尽");
        }
        return seconds;
      };
    state("NAVIGATE");
    mode("NAV");
    geometry_msgs::msg::PoseStamped waypoint;
    waypoint.header.frame_id = "odom";
    waypoint.pose.position.x = target.x;
    waypoint.pose.position.y = target.y;
    waypoint.pose.orientation.z = std::sin(target.yaw / 2);
    waypoint.pose.orientation.w = std::cos(target.yaw / 2);
    auto map_tf = tf_->lookupTransform("map", "odom", tf2::TimePointZero,
        tf2::durationFromSec(3.));
    Nav::Goal goal;
    goal.behavior_tree = ament_index_cpp::get_package_share_directory("g2_chapter12") +
      "/config/navigate_to_pose.xml";
    tf2::doTransform(waypoint, goal.pose, map_tf);
    goal.pose.header.stamp = node_->now();
    RCLCPP_INFO(node_->get_logger(), "导航目标 map=(%.3f, %.3f)，停靠目标 odom=(%.3f, %.3f, %.3f)",
      goal.pose.pose.position.x, goal.pose.pose.position.y, target.x, target.y, target.yaw);
    call<Nav>(nav_, goal, remaining(), "/navigate_to_pose");
    // Nav2 的成功是到达路径终点；NavFn 容差回退和 AMCL 修正均可能使它偏离精确停靠点。
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!global_path_.poses.empty()) {
        const auto & end = global_path_.poses.back().pose.position;
        RCLCPP_INFO(node_->get_logger(), "全局路径终点 %s=(%.3f, %.3f)，距原导航目标 %.3f m",
          global_path_.header.frame_id.c_str(), end.x, end.y,
          std::hypot(end.x - goal.pose.pose.position.x, end.y - goal.pose.pose.position.y));
      }
    }
    state("SETTLE");
    mode("ARM");  // 已有互锁要求新鲜 odom 且静止至少 0.5 s。
    auto actual = base_pose();
    RCLCPP_INFO(node_->get_logger(), "导航后复核 odom=(%.3f, %.3f, %.3f)，误差 %.3f m / %.3f rad",
      actual.x, actual.y, actual.yaw, docking::distance(actual, target),
      std::abs(docking::wrap(actual.yaw - target.yaw)));
    if (!docking::within(actual, target)) {
      // 不重复发送可能立即成功的 map 目标。短路径终点固定在 odom；仍由 Nav2 DWB
      // 做动态窗口速度采样和局部障碍检查。受阻即失败，不绕过碰撞检查强行精靠。
      auto points = docking::short_path(actual, target);
      Follow::Goal precise;
      precise.controller_id = "Dock";
      precise.goal_checker_id = "dock_goal_checker";
      precise.path.header.frame_id = "odom";
      precise.path.header.stamp = node_->now();
      for (const auto & p : points) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header = precise.path.header;
        pose.pose.position.x = p.x;
        pose.pose.position.y = p.y;
        pose.pose.orientation.z = std::sin(p.yaw / 2);
        pose.pose.orientation.w = std::cos(p.yaw / 2);
        precise.path.poses.push_back(pose);
      }
      state("DOCK");
      mode("NAV");
      dock_path_pub_->publish(precise.path);
      call<Follow>(follow_, precise, std::min(45., remaining()), "/follow_path");
      state("SETTLE");
      mode("ARM");
    }
    // 使用动作完成之后收到的新 TF 连续复核 0.5 s，不能用旧缓存或单帧合格放行。
    auto previous_stamp = fresh_base().header.stamp;
    const auto stable_until = std::chrono::steady_clock::now() + 500ms;
    while (rclcpp::ok() && std::chrono::steady_clock::now() < stable_until) {
      std::this_thread::sleep_for(50ms);
      if (!healthy("ARM")) {
        throw std::runtime_error("停靠复核时静止互锁丢失");
      }
      actual = base_pose();
      if (!docking::within(actual, target)) {
        RCLCPP_ERROR(node_->get_logger(), "最终停靠偏差 %.3f m / %.3f rad",
          docking::distance(actual, target), std::abs(docking::wrap(actual.yaw - target.yaw)));
        throw std::runtime_error("停靠仍超出 6 cm / 0.05 rad，拒绝伸臂");
      }
    }
    if (!rclcpp::ok() || rclcpp::Time(fresh_base().header.stamp) <= rclcpp::Time(previous_stamp)) {
      throw std::runtime_error("停靠复核期间 TF 未更新");
    }
    RCLCPP_INFO(node_->get_logger(), "停靠通过：%.3f m / %.3f rad（连续静止复核）",
      docking::distance(actual, target), std::abs(docking::wrap(actual.yaw - target.yaw)));
  }
  void gripper(double angle)
  {
    Grip::Goal goal;
    goal.command.position = angle;
    goal.command.max_effort = 0.;
    auto r = call<Grip>(grip_, goal, 25., "/gripper_controller/gripper_cmd");
    RCLCPP_INFO(
      node_->get_logger(), "gripper target=%.3f actual=%.3f stalled=%d", angle, r.result->position,
      r.result->stalled);
    if (!(r.result->reached_goal || (angle < .1 && r.result->stalled))) {
      throw std::runtime_error("夹爪没有到位或检测到阻挡");
    }
  }
  void planned(Group & arm)
  {
    arm.setStartStateToCurrentState();
    Plan::Goal goal;
    arm.constructMotionPlanRequest(goal.request);
    goal.planning_options.plan_only = true;
    auto r = call<Plan>(plan_, goal, 30., "/move_action");
    if (r.result->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
      throw PlanningFailure("MoveIt 规划失败 code=" + std::to_string(r.result->error_code.val));
    }
    execute(r.result->planned_trajectory);
  }
  void execute(const moveit_msgs::msg::RobotTrajectory & trajectory)
  {
    Execute::Goal goal;
    goal.trajectory = trajectory;
    auto r = call<Execute>(execute_, goal, cfg_["motion_timeout"].as<double>(), "/execute_trajectory");
    if (r.result->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
      throw std::runtime_error("轨迹执行失败");
    }
  }
  void joints(Group & arm, const std::vector<double> & q)
  {
    auto current = arm.getCurrentJointValues();
    double error = 0.;
    for (size_t i = 0; i < q.size(); ++i) {
      error = std::max(error, std::abs(current.at(i) - q[i]));
    }
    if (error < .02) {
      return;                  // 已到位，不发送零时长空动作。
    }
    arm.clearPoseTargets();
    if (!arm.setJointValueTarget(q)) {
      throw std::runtime_error("非法收臂角度");
    }
    planned(arm);
  }
  void pose(Group & arm, const geometry_msgs::msg::Pose & target)
  {
    RCLCPP_DEBUG(node_->get_logger(), "预期 TCP odom=(%.3f, %.3f, %.3f)，抓取忽略碰撞=%d",
      target.position.x, target.position.y, target.position.z, grasp_debug_active_);
    arm.clearPoseTargets();
    // 以当前关节作 IK 种子，避免 pose 约束随机选到翻肘/绕背的远端解。
    // 仍由 MoveIt 做关节空间路径规划；不使用近似 IK 或放宽限位。
    arm.setJointValueTarget(arm.getCurrentJointValues());
    geometry_msgs::msg::PoseStamped goal;
    goal.header.frame_id = "odom";
    goal.pose = target;
    if (!arm.setJointValueTarget(goal, "gripper_r_center_link")) {
      throw PlanningFailure("目标没有有效 IK（当前关节种子）");
    }
    planned(arm);
  }
  void cartesian(Group & arm, const geometry_msgs::msg::Pose & target)
  {
    if (!cartesian_->wait_for_service(5s)) {
      throw std::runtime_error("Cartesian 服务不可用");
    }
    auto state = arm.getCurrentState(3.);
    if (!state) {
      throw std::runtime_error("机械臂状态过期");
    }
    auto req = std::make_shared<moveit_msgs::srv::GetCartesianPath::Request>();
    req->header.frame_id = "odom";
    req->header.stamp = node_->now();
    req->group_name = "right_arm";
    req->link_name = "gripper_r_center_link";
    moveit::core::robotStateToRobotStateMsg(*state, req->start_state);
    req->start_state.is_diff = true;  // 保留场景中的持物碰撞体。
    req->waypoints = {target};
    req->max_step = .008;
    req->jump_threshold = 3.0;
    req->revolute_jump_threshold = .25;
    // 只有显式开启的仿真抓取窗口关闭检查；放置阶段恢复检查。
    req->avoid_collisions = !grasp_debug_active_;
    auto future = cartesian_->async_send_request(req);
    if (future.wait_for(10s) != std::future_status::ready) {
      cartesian_->remove_pending_request(future);
      throw std::runtime_error("Cartesian 服务超时");
    }
    auto result = future.get();
    if (result->error_code.val != 1 || result->fraction < .999) {
      throw PlanningFailure(
              "直线路径不完整 fraction=" + std::to_string(result->fraction) +
              "，拒绝部分执行");
    }
    robot_trajectory::RobotTrajectory rt(arm.getRobotModel(), "right_arm");
    rt.setRobotTrajectoryMsg(*state, result->solution);
    // 保留 Cartesian IK 路点；TOTG 的路径拟合可能在限位附近越界。
    trajectory_processing::IterativeParabolicTimeParameterization timing;
    if (!timing.computeTimeStamps(rt, .12, .12)) {
      throw std::runtime_error("轨迹时间参数化失败");
    }
    moveit_msgs::msg::RobotTrajectory trajectory;
    rt.getRobotTrajectoryMsg(trajectory);
    execute(
      trajectory);
  }
  geometry_msgs::msg::PoseArray fresh_poses()
  {
    const auto since = node_->now();
    const auto deadline = std::chrono::steady_clock::now() + 15s;
    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline && healthy("ARM")) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (poses_.header.frame_id == "odom" && !poses_.poses.empty() &&
          rclcpp::Time(poses_.header.stamp) > since &&
          (node_->now() - rclcpp::Time(poses_.header.stamp)).seconds() < .5)
        {
          return poses_;
        }
      }
      std::this_thread::sleep_for(50ms);
    }
    if (node_->count_publishers("/grasp_candidates") == 0) {
      throw std::runtime_error("/grasp_candidates 没有发布者：感知节点未启动或已退出，请检查 system 终端 perception.py 日志");
    }
    throw std::runtime_error("停稳后没有新鲜 RGB-D 抓取候选：检查 system 终端 [perception] 的点云、TF、ROI 和候选数量诊断");
  }
  void verify(
    const std::function<bool(const geometry_msgs::msg::Point &)> & test, double duration,
    const std::string & error)
  {
    const auto since = node_->now();
    auto first = since;
    auto previous_stamp = since;
    const auto deadline = std::chrono::steady_clock::now() + 20s;
    geometry_msgs::msg::Point previous;
    bool tracking = false;
    int samples = 0;
    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline && healthy("ARM")) {
      geometry_msgs::msg::PointStamped obs;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        obs = center_;
      }
      auto stamp = rclcpp::Time(obs.header.stamp);
      if (stamp > previous_stamp && stamp > since && (node_->now() - stamp).seconds() < .5) {
        const double delta =
          std::sqrt(
          std::pow(obs.point.x - previous.x, 2) + std::pow(
            obs.point.y - previous.y,
            2) +
          std::pow(obs.point.z - previous.z, 2));
        if (!test(obs.point) ||
          (tracking && (delta > .015 || (stamp - previous_stamp).seconds() > .5)))
        {
          tracking = false;
          samples = 0;
        } else {
          if (!tracking) {
            first = stamp;
            tracking = true;
          }
          if (++samples >= 5 && (stamp - first).seconds() >= duration) {
            return;
          }
        }
        previous = obs.point;
        previous_stamp = stamp;
      }
      std::this_thread::sleep_for(50ms);
    }
    throw std::runtime_error(error);
  }
  moveit_msgs::msg::CollisionObject box(const std::string & id, const std::vector<double> & b)
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = "odom";
    obj.id = id;
    obj.operation = obj.ADD;
    shape_msgs::msg::SolidPrimitive shape;
    shape.type = shape.BOX;
    shape.dimensions = {b[3], b[4], b[5]};
    geometry_msgs::msg::Pose p;
    p.position.x = b[0];
    p.position.y = b[1];
    p.position.z = b[2];
    p.orientation.w = 1.;
    obj.primitives = {shape};
    obj.primitive_poses = {p};
    return obj;
  }
  void add_scene(Scene & scene)
  {
    std::vector<moveit_msgs::msg::CollisionObject> objects;
    objects.push_back(box("table", cfg_["table"].as<std::vector<double>>()));
    objects.push_back(box("source_table", cfg_["source_table"].as<std::vector<double>>()));
    objects.push_back(box("tray", cfg_["tray"].as<std::vector<double>>()));
    int i = 0;
    for (auto b:cfg_["obstacles"]) {
      objects.push_back(box("obstacle" + std::to_string(i++), b.as<std::vector<double>>()));
    }
    for (auto b:
      std::vector<std::vector<double>>{{0, -4.69, 1, 9.5, .12, 2}, {0, 4.69, 1, 9.5, .12, 2},
        {-4.69, 0, 1, .12, 9.5, 2}, {4.69, 0, 1, .12, 9.5, 2}})
    {
      objects.push_back(box("wall" + std::to_string(i++), b));
    }
    objects.push_back(box("ground", {0, 0, -.03, 9.5, 9.5, .05}));
    if (!scene.applyCollisionObjects(objects)) {
      throw std::runtime_error("规划场景服务不可用");
    }
  }
  moveit_msgs::msg::CollisionObject target_object(const geometry_msgs::msg::Pose & grasp)
  {
    auto table = cfg_["table"].as<std::vector<double>>();
    return box("target", {grasp.position.x, grasp.position.y, grasp.position.z, .05, .05, .05});
  }
  // 仅供 Isaac 仿真调试。保存/恢复整张 ACM，不删除规划场景几何。
  void begin_grasp_debug(Scene & scene, Group & arm)
  {
    if (!cfg_["simulation_grasp_ignore_collisions"].as<bool>(false)) return;
    if (!node_->get_parameter("use_sim_time").as_bool())
      throw std::runtime_error("忽略抓取碰撞仅支持仿真 use_sim_time=true");
    if (grasp_debug_active_) throw std::runtime_error("抓取调试窗口重复开启");
    const auto snapshot = scene.read(
      moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX |
      moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_NAMES);
    saved_acm_ = snapshot.allowed_collision_matrix;
    collision_detection::AllowedCollisionMatrix acm(saved_acm_);
    auto names = arm.getRobotModel()->getLinkModelNames();
    for (const auto & object : snapshot.world.collision_objects) names.push_back(object.id);
    names.push_back("target");
    acm.setEntry(names, names, true);
    acm.setEntry(true);
    for (const auto & name : names) acm.setDefaultEntry(name, true);
    moveit_msgs::msg::PlanningScene diff;
    diff.is_diff = true;
    acm.getMessage(diff.allowed_collision_matrix);
    // 即使服务应答丢失也尝试恢复，不能遗留全局宽松的碰撞矩阵。
    grasp_debug_active_ = true;
    if (!scene.applyPlanningScene(diff)) {
      end_grasp_debug(scene);
      throw std::runtime_error("开启仿真抓取调试失败");
    }
    RCLCPP_WARN(node_->get_logger(),
      "SIM ONLY：抓取/抬升/回 home 暂时忽略 MoveIt 环境及自身碰撞；Isaac 物理碰撞与 Nav2 避障不变");
  }
  void end_grasp_debug(Scene & scene)
  {
    if (!grasp_debug_active_) return;
    moveit_msgs::msg::PlanningScene diff;
    diff.is_diff = true;
    diff.allowed_collision_matrix = saved_acm_;
    if (!scene.applyPlanningScene(diff)) throw std::runtime_error("恢复碰撞矩阵失败");
    grasp_debug_active_ = false;
    RCLCPP_INFO(node_->get_logger(), "抓取调试结束，已恢复 MoveIt 碰撞检查");
  }
  void allow_touch(Scene & scene, bool allowed)
  {
    if (grasp_debug_active_) return;  // 调试窗口不能被目标接触设置覆盖。
    // ACM diff 是整张矩阵替换：必须先读取，保留 SRDF 相邻关节白名单。
    collision_detection::AllowedCollisionMatrix acm(scene.read(
      moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX).allowed_collision_matrix);
    acm.setEntry("target", "gripper_r_center_link", allowed);
    moveit_msgs::msg::PlanningScene diff;
    diff.is_diff = true;
    acm.getMessage(diff.allowed_collision_matrix);
    if (!scene.applyPlanningScene(diff)) {
      throw std::runtime_error("夹爪接触白名单同步失败");
    }
  }
  rclcpp::Node::SharedPtr node_;
  YAML::Node cfg_;
  bool grasp_debug_active_{false};
  moveit_msgs::msg::AllowedCollisionMatrix saved_acm_;
  std::mutex mutex_;
  std::string requested_mode_{"IDLE"}, actual_mode_{"IDLE"};
  std::chrono::steady_clock::time_point mode_at_{};
  nav_msgs::msg::Path global_path_;
  geometry_msgs::msg::PoseArray poses_;
  geometry_msgs::msg::PointStamped center_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  tf2_ros::TransformListener listener_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr mode_pub_, status_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr mode_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr poses_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr center_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr dock_path_pub_;
  rclcpp::TimerBase::SharedPtr heartbeat_;
  rclcpp_action::Client<Nav>::SharedPtr nav_;
  rclcpp_action::Client<Follow>::SharedPtr follow_;
  rclcpp_action::Client<Grip>::SharedPtr grip_;
  rclcpp_action::Client<Plan>::SharedPtr plan_;
  rclcpp_action::Client<Execute>::SharedPtr execute_;
  rclcpp::Client<moveit_msgs::srv::GetCartesianPath>::SharedPtr cartesian_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared(
    "mobile_mission",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&] {
      executor.spin();
    });
  int code = 0;
  try {
    Mission task(node);
    try {
      task.run();
    } catch (const std::exception & e) {
      task.fail(e.what());
      code = 1;
    }
  } catch (const std::exception & e) {
    RCLCPP_ERROR(node->get_logger(), "启动失败：%s", e.what());
    code = 1;
  }
  executor.cancel();
  spinner.join();
  rclcpp::shutdown();
  return code;
}
