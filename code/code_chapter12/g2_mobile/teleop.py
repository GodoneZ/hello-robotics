#!/usr/bin/env python3
"""建图时的低速键盘遥控。与 mission 二选一；松键 0.25 秒自动归零。"""

import select
import sys
import termios
import time
import tty
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


def main():
    if not sys.stdin.isatty():
        raise RuntimeError("teleop 需要交互终端")
    rclpy.init()
    node = Node("mapping_teleop")
    mode = node.create_publisher(String, "/mode_request", 10)
    cmd = node.create_publisher(Twist, "/cmd_vel", 10)
    current = {"mode": "IDLE"}
    sub = node.create_subscription(
        String, "/motion_mode", lambda msg: current.update(mode=msg.data), 10
    )
    settings = termios.tcgetattr(sys.stdin)
    velocity = (0.0, 0.0, 0.0)
    last = 0.0
    keys = {
        "w": (0.18, 0.0, 0.0),
        "s": (-0.18, 0.0, 0.0),
        "a": (0.0, 0.18, 0.0),
        "d": (0.0, -0.18, 0.0),
        "q": (0.0, 0.0, 0.3),
        "e": (0.0, 0.0, -0.3),
        " ": (0.0, 0.0, 0.0),
    }
    print("建图：W/S 前后，A/D 横移，Q/E 旋转，空格停，X 退出。需要机械臂已在 home。")
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            mode.publish(String(data="NAV"))
            rclpy.spin_once(node, timeout_sec=0.02)
            if select.select([sys.stdin], [], [], 0.02)[0]:
                key = sys.stdin.read(1).lower()
                if key in ("x", "\x03"):
                    break
                velocity = keys.get(key, (0.0, 0.0, 0.0))
                last = time.monotonic()
            value = (
                velocity
                if time.monotonic() - last < 0.25 and current["mode"] == "NAV"
                else (0.0, 0.0, 0.0)
            )
            msg = Twist()
            msg.linear.x, msg.linear.y, msg.angular.z = value
            cmd.publish(msg)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        if rclpy.ok():
            cmd.publish(Twist())
            mode.publish(String(data="IDLE"))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
