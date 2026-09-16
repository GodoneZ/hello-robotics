"""Isaac Python 仅处理物理和基础消息；标准 action 在系统 ROS Python 内实现。"""

import argparse
import csv
from pathlib import Path
import math
import time
import numpy as np
from g2_mobile.common import ARM, GRIPPER, LOWER, UPPER, rotation, backproject
from g2_mobile.mapping import grid, raycast
from sim.simulation import Simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=0, help="有限步数冒烟测试，0 为持续运行")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="只读物理轨迹/相机快照写入 log/integration，不参与控制",
    )
    args = parser.parse_args()
    sim = Simulation(args.headless)
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import TransformStamped, Twist
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState, LaserScan, PointCloud2, PointField, Image, CameraInfo
    from std_msgs.msg import String
    from tf2_ros import TransformBroadcaster

    rclpy.init()
    node = Node("isaac_mobile_bridge")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    tf = TransformBroadcaster(node)
    pubs = {
        name: node.create_publisher(
            typ,
            topic,
            qos_profile_sensor_data if name in ("rgb", "depth", "info", "cloud", "scan") else 10,
        )
        for name, typ, topic in [
            ("clock", Clock, "/clock"),
            ("odom", Odometry, "/odom"),
            ("scan", LaserScan, "/scan"),
            ("joints", JointState, "/joint_states"),
            ("all_joints", JointState, "/isaac/joint_states"),
            ("cloud", PointCloud2, "/camera/points"),
            ("rgb", Image, "/camera/color/image_raw"),
            ("depth", Image, "/camera/depth/image_raw"),
            ("info", CameraInfo, "/camera/camera_info"),
        ]
    }
    diagnostic = {"state": "START", "saved_at": 0.0}
    trace = None
    if args.diagnostics:
        from PIL import Image as PILImage

        log = Path(__file__).resolve().parents[1] / "log/integration"
        log.mkdir(parents=True, exist_ok=True)
        trace = (log / "physics.csv").open("w", buffering=1)
        writer = csv.writer(trace)
        writer.writerow(
            [
                "sim_time",
                "state",
                "tcp_x",
                "tcp_y",
                "tcp_z",
                "gripper",
                "base_x",
                "base_y",
                "base_yaw",
                "cmd_x",
                "cmd_y",
                "cmd_yaw",
                "source_apple_x", "source_apple_y", "source_apple_z",
                "destination_apple_x", "destination_apple_y", "destination_apple_z",
                "arm_x", "arm_y", "arm_z", "arm_qw", "arm_qx", "arm_qy", "arm_qz",
                "gripper_target", "motion_mode",
            ]
        )

    def task_status(msg):
        diagnostic["state"] = msg.data.split(":")[0]
        diagnostic["saved_at"] = 0.0

    data = {"mode": "IDLE", "mode_time": 0.0, "vel": np.zeros(3), "vel_time": 0.0}

    def mode(msg):
        if msg.data != data["mode"] and msg.data not in ("NAV", "ARM"):
            sim.hold()
        data.update(mode=msg.data, mode_time=time.monotonic())

    def velocity(msg):
        v = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        if np.isfinite(v).all():
            data.update(
                vel=np.clip(v, [-0.55, -0.55, -1.0], [0.55, 0.55, 1.0]), vel_time=time.monotonic()
            )

    def joints(msg):
        if data["mode"] != "ARM" or time.monotonic() - data["mode_time"] > 0.6:
            return
        p = np.asarray(msg.position)
        if not np.isfinite(p).all():
            return
        if tuple(msg.name) == ARM and p.shape == (7,) and np.all(p >= LOWER) and np.all(p <= UPPER):
            sim.arm_target = p.copy()
        if msg.name == [GRIPPER] and p.shape == (1,) and 0 <= p[0] <= 0.785:
            sim.grip_target = float(p[0])

    subscriptions = [
        node.create_subscription(String, "/motion_mode", mode, 10),
        node.create_subscription(String, "/task_status", task_status, 10),
        node.create_subscription(Twist, "/cmd_vel", velocity, 10),
        node.create_subscription(JointState, "/isaac/joint_command", joints, 10),
    ]
    raw = grid(sim.cfg)
    last_pose, last_t = sim.pose(), sim.time
    count, stale = 0, True

    def publish_tf(parent, child, p, q, stamp):
        t = TransformStamped()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.header.stamp = stamp
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(
            float, p
        )
        (
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        ) = map(float, q)
        tf.sendTransform(t)

    try:
        while sim.app.is_running() and (not args.steps or count < args.steps):
            for _ in range(8):
                executor.spin_once(timeout_sec=0.0)
            expired = time.monotonic() - data["mode_time"] > 0.6
            if expired and not stale:
                sim.hold()
            stale = expired
            vel = (
                data["vel"]
                if not expired
                and data["mode"] == "NAV"
                and time.monotonic() - data["vel_time"] < 0.4
                else np.zeros(3)
            )
            sim.step(vel)
            count += 1
            now = sim.time
            stamp = Clock().clock
            stamp.sec = int(now)
            stamp.nanosec = int((now - int(now)) * 1e9)
            c = Clock()
            c.clock = stamp
            pubs["clock"].publish(c)
            p = sim.pose()
            delta = p - last_pose
            delta[2] = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
            twist = delta / max(now - last_t, 1e-6)
            cs, sn = math.cos(p[2]), math.sin(p[2])
            body_v = [cs * twist[0] + sn * twist[1], -sn * twist[0] + cs * twist[1], twist[2]]
            last_pose, last_t = p, now
            q = [0.0, 0.0, math.sin(p[2] / 2), math.cos(p[2] / 2)]
            publish_tf("odom", "base_link", [p[0], p[1], 0.0], q, stamp)
            R = rotation(q)
            ap, aq = sim.arm_prim.get_world_pose()
            # base_link 为地面投影 frame；所有高度保留在 base→arm 中。
            rel = R.T @ (ap - np.array([p[0], p[1], 0.0]))
            from sim.arm.kinematics import matrix_to_quaternion

            arm_R = rotation([aq[1], aq[2], aq[3], aq[0]])
            ar = matrix_to_quaternion(R.T @ arm_R)
            publish_tf("base_link", "arm_base_link", rel, [ar[1], ar[2], ar[3], ar[0]], stamp)
            publish_tf("base_link", "base_scan", [0, 0, 0.3], [0, 0, 0, 1], stamp)
            cp, cq = sim.camera.get_world_pose(camera_axes="ros")
            cr = matrix_to_quaternion(R.T @ rotation([cq[1], cq[2], cq[3], cq[0]]))
            publish_tf(
                "base_link",
                "camera_optical_frame",
                R.T @ (cp - [p[0], p[1], 0]),
                [cr[1], cr[2], cr[3], cr[0]],
                stamp,
            )
            od = Odometry()
            od.header.stamp = stamp
            od.header.frame_id = "odom"
            od.child_frame_id = "base_link"
            od.pose.pose.position.x, od.pose.pose.position.y = map(float, p[:2])
            od.pose.pose.orientation.z = q[2]
            od.pose.pose.orientation.w = q[3]
            od.twist.twist.linear.x, od.twist.twist.linear.y, od.twist.twist.angular.z = map(
                float, body_v
            )
            od.pose.covariance[0] = 0.0025
            od.pose.covariance[7] = 0.0025
            od.pose.covariance[35] = 0.004
            pubs["odom"].publish(od)
            js = JointState()
            js.header.stamp = stamp
            js.name = sim.names
            js.position = np.asarray(sim.robot.get_joint_positions(), dtype=float).tolist()
            js.velocity = np.asarray(sim.robot.get_joint_velocities(), dtype=float).tolist()
            pubs["all_joints"].publish(js)
            js.name = list(ARM)
            js.position = [js.position[i] for i in sim.arm_ids]
            js.velocity = [js.velocity[i] for i in sim.arm_ids]
            pubs["joints"].publish(js)
            if count % 3:
                continue
            scan = LaserScan()
            scan.header.stamp = stamp
            scan.header.frame_id = "base_scan"
            scan.angle_min = -math.pi
            scan.angle_max = math.pi
            scan.angle_increment = 2 * math.pi / 359
            scan.range_min = 0.12
            scan.range_max = 8.0
            scan.scan_time = 0.1
            scan.time_increment = 0.0
            scan.ranges = raycast(raw, p)
            pubs["scan"].publish(scan)
            if count % 6:  # RGB-D 5 Hz，避免大图像挤占控制消息。
                continue
            frame = sim.camera.get_current_frame()
            rgb = frame.get("rgb", frame.get("rgba"))
            depth = frame.get("distance_to_image_plane")
            if rgb is None or depth is None:
                continue
            rgb = np.asarray(rgb)[..., :3]
            depth = np.asarray(depth).squeeze().astype("<f4")
            if depth.shape != rgb.shape[:2]:
                continue
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            if trace is not None:
                tcp, _ = sim.tcp_prim.get_world_pose()
                writer.writerow(
                    [
                        now,
                        diagnostic["state"],
                        *tcp,
                        sim.robot.get_joint_positions()[sim.grip_id],
                        *p,
                        *vel,
                        *sim.objects["source_apple"].get_world_pose()[0],
                        *sim.objects["destination_apple"].get_world_pose()[0],
                        *ap, *aq,
                        sim.grip_target, data["mode"],
                    ]
                )
                if time.monotonic() - diagnostic["saved_at"] > 1.0:
                    PILImage.fromarray(rgb).save(log / (diagnostic["state"] + ".png"))
                    diagnostic["saved_at"] = time.monotonic()
            K = np.asarray(sim.camera.get_intrinsics_matrix())
            for key, image, enc in [("rgb", rgb, "rgb8"), ("depth", depth, "32FC1")]:
                if pubs[key].get_subscription_count() == 0:
                    continue  # 无消费者时不发送 MB 级图像，避免 UDP 分片挤占点云。
                msg = Image()
                msg.header.stamp = stamp
                msg.header.frame_id = "camera_optical_frame"
                msg.height, msg.width = image.shape[:2]
                msg.encoding = enc
                msg.step = msg.width * (3 if key == "rgb" else 4)
                msg.data = np.ascontiguousarray(image).tobytes()
                pubs[key].publish(msg)
            info = CameraInfo()
            info.header.stamp = stamp
            info.header.frame_id = "camera_optical_frame"
            info.height, info.width = depth.shape
            info.k = K.flatten().tolist()
            info.distortion_model = "plumb_bob"
            info.d = [0.0] * 5
            info.r = np.eye(3).flatten().tolist()
            info.p = np.c_[K, np.zeros(3)].flatten().tolist()
            pubs["info"].publish(info)
            xyz = backproject(depth, K).reshape(-1, 3)
            colors = rgb[::3, ::3].reshape(-1, 3).astype(np.uint32)
            valid = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0.05) & (xyz[:, 2] < 5.0)
            packed = np.empty((int(valid.sum()), 4), dtype="<f4")
            packed[:, :3] = xyz[valid]
            packed[:, 3].view("<u4")[:] = (
                (colors[valid, 0] << 16) | (colors[valid, 1] << 8) | colors[valid, 2]
            )
            cloud = PointCloud2()
            cloud.header = info.header
            cloud.height = 1
            cloud.width = len(packed)
            cloud.fields = [
                PointField(
                    name=n,
                    offset=i * 4,
                    datatype=PointField.FLOAT32 if i < 3 else PointField.UINT32,
                    count=1,
                )
                for i, n in enumerate(["x", "y", "z", "rgb"])
            ]
            cloud.point_step = 16
            cloud.row_step = 16 * len(packed)
            cloud.is_dense = True
            cloud.data = packed.tobytes()
            pubs["cloud"].publish(cloud)
        print(f"[chapter12] bridge finished: {count} steps", flush=True)
    finally:
        sim.base.stop()
        if trace is not None:
            trace.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        sim.close()


if __name__ == "__main__":
    main()
