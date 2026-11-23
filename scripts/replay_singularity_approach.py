"""Animate a damped Cartesian approach toward the singular pose in RViz only."""

import numpy as np

from kinematics.control import follow
from kinematics.jacobian import jacobian
from robot.widowx import JOINT_NAMES


def main():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    q_sing = np.array([-0.3998, 1.2352, 2.2944, 0.5220, np.pi / 2, -2.3648])
    _, _, Vh = np.linalg.svd(jacobian(q_sing))
    q0 = q_sing + 0.1 * Vh[0]
    duration = 3.0
    dt = 0.01
    steps = int(round(duration / dt))
    zetas = np.zeros((steps, 6))
    zetas[:] = jacobian(q_sing) @ (-0.1 * Vh[0] / duration)
    trajectory, _ = follow(q0, zetas, dt, method="dls", lam=1e-2)
    names = list(JOINT_NAMES) + ["left_carriage_joint", "right_carriage_joint"]

    class Replay(Node):
        def __init__(self):
            super().__init__("replay_singularity_approach")
            self.publisher = self.create_publisher(JointState, "/joint_states", 10)
            self.index = 0
            self.timer = self.create_timer(dt, self.publish_next)

        def publish_next(self):
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = names
            msg.position = trajectory[self.index].tolist() + [0.0, 0.0]
            self.publisher.publish(msg)
            self.index += 1
            if self.index >= len(trajectory):
                self.get_logger().info("damped singularity approach complete")
                self.timer.cancel()

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
