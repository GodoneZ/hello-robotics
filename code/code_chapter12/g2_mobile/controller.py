#!/usr/bin/env python3
"""标准 ROS action 适配器与互锁，不做运动规划，不访问 Isaac API。"""

import threading
import time
import numpy as np
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory, GripperCommand
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from g2_mobile.common import (
    ARM,
    GRIPPER,
    load_config,
    validate_trajectory,
    sample_trajectory,
    stopped,
)


class Controller(Node):
    def __init__(self):
        super().__init__("mobile_controller")
        self.cfg = load_config()
        self.lock = threading.RLock()
        self.busy = False
        self.estop = False
        self.carry_fault = False
        self.carry_bad_since = None
        self.releasing = False
        self.holding = False  # 成功闭合后锁存；只有成功张开才解除。
        self.state = {}
        self.state_at = 0.0
        self.odom_at = 0.0
        self.stop_since = 0.0
        self.request = "IDLE"
        self.request_at = 0.0
        self.mode = "IDLE"
        self.group = ReentrantCallbackGroup()
        self.command = self.create_publisher(JointState, "/isaac/joint_command", 10)
        self.mode_pub = self.create_publisher(String, "/motion_mode", 10)
        self.create_subscription(JointState, "/isaac/joint_states", self.on_state, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(String, "/mode_request", self.on_request, 10)
        self.create_service(Trigger, "/estop", self.on_estop)
        self.create_service(Trigger, "/reset_estop", self.on_reset)
        self.timer = self.create_timer(
            0.05, self.watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME)
        )
        self.arm_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/right_arm_controller/follow_joint_trajectory",
            self.execute_arm,
            goal_callback=self.arm_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self.group,
        )
        self.grip_server = ActionServer(
            self,
            GripperCommand,
            "/gripper_controller/gripper_cmd",
            self.execute_gripper,
            goal_callback=self.grip_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self.group,
        )

    def on_state(self, msg):
        if len(msg.name) != len(msg.position) or not np.isfinite(msg.position).all():
            return
        with self.lock:
            self.state = dict(zip(msg.name, msg.position))
            self.state_at = time.monotonic()

    def on_odom(self, msg):
        t = msg.twist.twist
        with self.lock:
            self.odom_at = time.monotonic()
            if stopped([t.linear.x, t.linear.y, t.angular.z]):
                if not self.stop_since:
                    self.stop_since = self.odom_at
            else:
                self.stop_since = 0.0

    def on_request(self, msg):
        with self.lock:
            self.request = msg.data if msg.data in ("NAV", "ARM", "IDLE") else "IDLE"
            self.request_at = time.monotonic()

    def on_estop(self, req, res):
        with self.lock:
            self.estop = True
        self.watchdog()
        res.success = True
        res.message = "已锁存急停；保持夹爪，不自动释放物体"
        return res

    def on_reset(self, req, res):
        with self.lock:
            res.success = (
                self.fresh() and self.stop_since > 0 and not self.busy and self.request == "IDLE"
                and not self.carry_fault
            )
            if res.success:
                self.estop = False
        res.message = (
            "急停已复位" if res.success else "需请求 IDLE、反馈新鲜、底盘静止且无执行中 action"
        )
        return res

    def fresh(self):
        now = time.monotonic()
        return (
            now - self.state_at < 0.8
            and now - self.odom_at < 0.8
            and all(n in self.state for n in (*ARM, GRIPPER))
        )

    def watchdog(self):
        with self.lock:
            mode = "IDLE"
            now = time.monotonic()
            # 全阶段监控：NAV -> IDLE -> DOCK -> ARM 不能解除持物要求。
            # 开度只是失持征兆，不宣称等同于接触/力传感器。
            carry_valid = (not self.holding or self.releasing or
                           (self.fresh() and self.cfg['carry_gripper_range'][0] <
                            self.state[GRIPPER] < self.cfg['carry_gripper_range'][1]))
            if self.holding and not self.releasing and self.fresh() and not carry_valid:
                if self.carry_bad_since is None:
                    self.carry_bad_since = now
                if now - self.carry_bad_since >= .20 and not self.carry_fault:
                    self.carry_fault = True
                    self.get_logger().error(
                        f"CARRY_LOST: 持物夹爪开度异常 q={self.state[GRIPPER]:.3f}；"
                        "锁存停止，禁止空手继续放置；检查物体并重启任务/系统")
            else:
                self.carry_bad_since = None
            if not self.carry_fault and carry_valid and not self.estop and self.fresh() and time.monotonic() - self.request_at < 0.75:
                if (
                    self.request == "NAV"
                    and not self.busy
                    and (not self.holding or
                         self.cfg['carry_gripper_range'][0] < self.state[GRIPPER] <
                         self.cfg['carry_gripper_range'][1])
                    and max(abs(np.array([self.state[n] for n in ARM]) - self.cfg["home"])) < 0.10
                ):
                    mode = "NAV"
                elif (
                    self.request == "ARM"
                    and self.stop_since
                    and time.monotonic() - self.stop_since > 0.5
                ):
                    mode = "ARM"
            self.mode = mode
            self.mode_pub.publish(String(data=mode))

    def reserve(self):
        with self.lock:
            if self.busy or self.mode != "ARM" or not self.fresh() or self.estop:
                return GoalResponse.REJECT
            self.busy = True
        return GoalResponse.ACCEPT

    def arm_goal(self, goal):
        try:
            t = goal.trajectory
            validate_trajectory(
                t.joint_names,
                [p.positions for p in t.points],
                [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in t.points],
            )
            for p in t.points:
                for field in [p.velocities, p.accelerations, p.effort]:
                    if len(field) not in (0, 7) or not np.isfinite(field).all():
                        raise ValueError("非法轨迹导数")
            for tol in list(goal.path_tolerance) + list(goal.goal_tolerance):
                if (
                    tol.name not in ARM
                    or not np.isfinite([tol.position, tol.velocity, tol.acceleration]).all()
                ):
                    raise ValueError("非法 tolerance")
                if tol.velocity > 0 or tol.acceleration > 0:
                    raise ValueError("此最小控制器仅支持位置 tolerance")
            stamp = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
            if stamp and not -0.2 <= stamp - self.get_clock().now().nanoseconds * 1e-9 <= 1.0:
                raise ValueError("过期/过远的 header.stamp")
        except (ValueError, TypeError) as exc:
            self.get_logger().error(str(exc))
            return GoalResponse.REJECT
        return self.reserve()

    def grip_goal(self, goal):
        if (
            not np.isfinite([goal.command.position, goal.command.max_effort]).all()
            or not 0 <= goal.command.position <= 0.785
        ):
            return GoalResponse.REJECT
        # max_effort=0 使用 USD 中配置的驱动上限；本桥不宣称实现力矩控制。
        if goal.command.max_effort != 0:
            return GoalResponse.REJECT
        return self.reserve()

    def send(self, names, positions):
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(names)
        m.position = list(map(float, positions))
        self.command.publish(m)

    def actual(self, names):
        with self.lock:
            return np.array([self.state[n] for n in names])

    def safe(self):
        with self.lock:
            return (
                rclpy.ok()
                and self.mode == "ARM"
                and self.fresh()
                and not self.estop
                and time.monotonic() - self.request_at < 0.75
            )

    def finish(self, handle, result, ok=False):
        if handle.is_cancel_requested:
            handle.canceled()
        elif ok:
            handle.succeed()
        else:
            handle.abort()
        return result

    def execute_arm(self, handle):
        completed = False
        result = FollowJointTrajectory.Result()
        try:
            traj = handle.request.trajectory
            p, t = validate_trajectory(
                traj.joint_names,
                [p.positions for p in traj.points],
                [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in traj.points],
            )
            initial = self.actual(ARM)
            if (t[0] == 0 and max(abs(p[0] - initial)) > 0.10) or (
                t[0] > 0 and max(abs(p[0] - initial)) / t[0] > 3.1416
            ):
                raise RuntimeError("轨迹起点与实际关节不一致或首段超速")
            start = max(
                self.get_clock().now().nanoseconds * 1e-9,
                traj.header.stamp.sec + traj.header.stamp.nanosec * 1e-9,
            )
            wall_deadline = time.monotonic() + max(60.0, float(t[-1]) * 8.0)
            path_tol = np.full(7, 0.40)
            goal_tol = np.full(7, 0.015)
            for entries, values in [
                (handle.request.path_tolerance, path_tol),
                (handle.request.goal_tolerance, goal_tol),
            ]:
                for tol in entries:
                    if tol.position > 0:
                        values[ARM.index(tol.name)] = min(values[ARM.index(tol.name)], tol.position)
            margin = (
                handle.request.goal_time_tolerance.sec
                + handle.request.goal_time_tolerance.nanosec * 1e-9
            )
            margin = margin if margin > 0 else 5.0
            while self.safe() and time.monotonic() < wall_deadline:
                if handle.is_cancel_requested:
                    result.error_string = "用户取消，保持当前位置"
                    return self.finish(handle, result)
                elapsed = self.get_clock().now().nanoseconds * 1e-9 - start
                if elapsed < 0:
                    time.sleep(0.01)
                    continue
                desired = sample_trajectory(p, t, initial, elapsed)
                actual = self.actual(ARM)
                error = desired - actual
                self.send(ARM, desired)
                fb = FollowJointTrajectory.Feedback()
                fb.header.stamp = self.get_clock().now().to_msg()
                fb.joint_names = list(ARM)
                fb.desired.positions = desired.tolist()
                fb.actual.positions = actual.tolist()
                fb.error.positions = error.tolist()
                handle.publish_feedback(fb)
                if elapsed < t[-1] and np.any(abs(error) > path_tol):
                    result.error_code = result.PATH_TOLERANCE_VIOLATED
                    raise RuntimeError(f"关节跟踪误差超限 joint={ARM[int(np.argmax(abs(error)))]} "
                                       f"error={np.round(error, 3).tolist()} desired={np.round(desired, 3).tolist()} "
                                       f"actual={np.round(actual, 3).tolist()}")
                if elapsed >= t[-1] and np.all(abs(p[-1] - actual) <= goal_tol):
                    result.error_code = result.SUCCESSFUL
                    completed = True
                    return self.finish(handle, result, True)
                if elapsed > t[-1] + margin:
                    result.error_code = result.GOAL_TOLERANCE_VIOLATED
                    raise RuntimeError("最终关节未收敛")
                time.sleep(0.01)
            raise RuntimeError("互锁/反馈/心跳异常或墙钟超时")
        except Exception as exc:
            if result.error_code == 0:
                result.error_code = result.INVALID_GOAL
            result.error_string = str(exc)
            self.get_logger().error(str(exc))
            return self.finish(handle, result)
        finally:
            with self.lock:
                if all(n in self.state for n in ARM):
                    # 成功保持最终目标，不把容差内尚未收敛的反馈当成新目标。
                    self.send(ARM, p[-1] if completed else self.actual(ARM))
                self.busy = False

    def execute_gripper(self, handle):
        result = GripperCommand.Result()
        target = handle.request.command.position
        previous = float(self.actual([GRIPPER])[0])
        initial = previous
        stable = time.monotonic()
        begin = self.get_clock().now().nanoseconds * 1e-9
        deadline = time.monotonic() + 20
        completed = False
        with self.lock:
            self.releasing = target > .7  # 仅显式成功接受的张开 action 暂停持物开度检查。
        # 接触后的物理振荡需要收敛；最多 8s 仿真/20s 墙钟，稳定阈值不变。
        try:
            while (
                self.safe()
                and time.monotonic() < deadline
                and self.get_clock().now().nanoseconds * 1e-9 - begin < 8.0
            ):
                if handle.is_cancel_requested:
                    return self.finish(handle, result)
                elapsed = self.get_clock().now().nanoseconds * 1e-9 - begin
                ratio = min(max(elapsed / 0.6, 0.0), 1.0)
                # 第九章平滑开合：避免指令阶跃把轻小物体弹走。
                command = initial + (target - initial) * (3 * ratio**2 - 2 * ratio**3)
                self.send([GRIPPER], [command])
                actual = float(self.actual([GRIPPER])[0])
                if abs(actual - previous) > 0.003:
                    stable = time.monotonic()
                    previous = actual
                result.position = actual
                result.reached_goal = ratio >= 1.0 and abs(actual - target) < 0.025
                result.stalled = (
                    ratio >= 1.0
                    and target < 0.1
                    and actual > 0.03
                    and time.monotonic() - stable > 0.8
                )
                if result.reached_goal or result.stalled:
                    with self.lock:
                        if target < 0.1:
                            self.holding = True
                        elif target > 0.7:
                            self.holding = False
                    completed = True
                    return self.finish(handle, result, True)
                time.sleep(0.02)
            elapsed = self.get_clock().now().nanoseconds * 1e-9 - begin
            self.get_logger().error(
                f"夹爪未完成：target={target:.3f}, actual={result.position:.3f}, "
                f"sim_elapsed={elapsed:.2f}s, stable_wall={time.monotonic() - stable:.2f}s, "
                f"safe={self.safe()}；需要到位或持续 0.8s 停滞，未放行抬升"
            )
            return self.finish(handle, result)
        finally:
            with self.lock:
                # 成功闭合保持驱动力；取消/失败保持当前位置，不继续挤压或自动张开。
                if not completed or handle.is_cancel_requested:
                    if not self.holding:
                        self.send([GRIPPER], self.actual([GRIPPER]))
                    elif self.releasing:
                        # 中断张开保持当时开度；监控将对失持锁存故障。
                        self.send([GRIPPER], self.actual([GRIPPER]))
                    else:
                        self.send([GRIPPER], [0.0])
                self.releasing = False
                self.busy = False


def main():
    rclpy.init()
    node = Controller()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.estop = True
        if rclpy.ok():
            node.mode_pub.publish(String(data="IDLE"))
        executor.shutdown(timeout_sec=3.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
