#pragma once
// PlanningScene 标准服务的有界适配：服务失联时明确失败，不在内部永久等待。
#include <rclcpp/rclcpp.hpp>
#include <moveit_msgs/srv/apply_planning_scene.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <chrono>
#include <stdexcept>
#include <vector>

class SceneClient
{
public:
  explicit SceneClient(const rclcpp::Node::SharedPtr & node)
  : apply_(node->create_client<moveit_msgs::srv::ApplyPlanningScene>("/apply_planning_scene")),
    get_(node->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene")) {}

  bool applyPlanningScene(moveit_msgs::msg::PlanningScene diff)
  {
    diff.is_diff = true;
    diff.robot_state.is_diff = true; // 不用空 RobotState 清除已有附着物。
    auto request = std::make_shared<moveit_msgs::srv::ApplyPlanningScene::Request>();
    request->scene = std::move(diff);
    return call<moveit_msgs::srv::ApplyPlanningScene>(apply_, request)->success;
  }
  bool applyCollisionObjects(const std::vector<moveit_msgs::msg::CollisionObject> & objects)
  {
    moveit_msgs::msg::PlanningScene diff;
    diff.world.collision_objects = objects;
    return applyPlanningScene(diff);
  }
  bool applyCollisionObject(const moveit_msgs::msg::CollisionObject & object)
  { return applyCollisionObjects({object}); }
  bool applyAttachedCollisionObject(const moveit_msgs::msg::AttachedCollisionObject & object)
  {
    moveit_msgs::msg::PlanningScene diff;
    diff.robot_state.attached_collision_objects = {object};
    return applyPlanningScene(diff);
  }
  void removeCollisionObjects(const std::vector<std::string> & names)
  {
    std::vector<moveit_msgs::msg::CollisionObject> objects;
    for (const auto & name : names) {
      moveit_msgs::msg::CollisionObject object;
      object.id = name;
      object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
      objects.push_back(object);
    }
    if (!applyCollisionObjects(objects)) throw std::runtime_error("移除碰撞物体失败");
  }
  moveit_msgs::msg::PlanningScene read(uint32_t components)
  {
    auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
    request->components.components = components;
    return call<moveit_msgs::srv::GetPlanningScene>(get_, request)->scene;
  }

private:
  template<class Service>
  typename Service::Response::SharedPtr call(
    const typename rclcpp::Client<Service>::SharedPtr & client,
    const typename Service::Request::SharedPtr & request)
  {
    using namespace std::chrono_literals;
    if (!client->wait_for_service(5s))
      throw std::runtime_error(std::string(client->get_service_name()) + " 服务不可用");
    auto future = client->async_send_request(request);
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
      if (future.wait_for(50ms) == std::future_status::ready) return future.get();
    }
    client->remove_pending_request(future);
    throw std::runtime_error(std::string(client->get_service_name()) + " 响应超时/任务中断");
  }
  rclcpp::Client<moveit_msgs::srv::ApplyPlanningScene>::SharedPtr apply_;
  rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr get_;
};
