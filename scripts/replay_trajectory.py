"""Replay a resolved-rate trajectory in RViz through /joint_states.

This node is visualization-only and never connects to the physical arm.
"""

import argparse

import numpy as np

from robot.widowx import JOINT_NAMES
from scripts.straight_line import simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", nargs=6, type=float, required=True, metavar="DEG")
    parser.add_argument("--target", nargs=3, type=float, required=True, metavar="M")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.01)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    _, _, trajectory, _, deviation = simulate(
        args.start, args.target, args.duration, args.dt
    )
    # The arm trajectory has six revolute joints.  The description also
    # contains two prismatic carriage joints; publishing them at zero gives
    # robot_state_publisher the complete transform tree for the gripper.
    published_names = list(JOINT_NAMES) + [
        "left_carriage_joint",
        "right_carriage_joint",
    ]

    class Replay(Node):
        def __init__(self):
            super().__init__("replay_trajectory")
            self.publisher = self.create_publisher(JointState, "/joint_states", 10)
            self.index = 0
            self.timer = self.create_timer(args.dt, self.publish_next)

        def publish_next(self):
            if self.index >= len(trajectory):
                self.get_logger().info(
                    f"trajectory complete; maximum deviation {np.max(deviation) * 1000:.3f} mm"
                )
                self.timer.cancel()
                rclpy.shutdown()
                return
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = published_names
            message.position = trajectory[self.index].tolist() + [0.0, 0.0]
            self.publisher.publish(message)
            self.index += 1

    rclpy.init()
    node = Replay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
