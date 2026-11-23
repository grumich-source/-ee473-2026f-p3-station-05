"""Forward kinematics for the WidowX AI."""

import argparse
import numpy as np

from kinematics.transforms import dh_transform, pose_to_xyz_rpy
from robot import widowx


def _angles(thetas):
    """Validate and convert joint angles to a one-dimensional NumPy array."""

    thetas = np.asarray(thetas, dtype=float).reshape(-1)

    if len(thetas) != widowx.N_JOINTS:
        raise ValueError(
            f"Expected {widowx.N_JOINTS} joint angles, got {len(thetas)}"
        )

    return thetas


def _six_joint_transform(thetas):
    """Calculate the transform through the six revolute joints."""

    thetas = _angles(thetas)
    T = np.eye(4)

    for i, q in enumerate(thetas):
        theta_offset, d, a, alpha = widowx.DH_PARAMS[i]

        T = T @ dh_transform(
            theta_offset + q,
            d,
            a,
            alpha,
        )

    return T


def fk(thetas):
    """Return the Tool pose expressed in the base frame.

    Joint angles are in radians.
    Lengths are in metres.
    """

    T = _six_joint_transform(thetas)

    # Frame 6 to Flange
    T = T @ widowx.FLANGE_OFFSET

    # Flange to Tool
    T = T @ widowx.TOOL_OFFSET

    return T


def fk_all_frames(thetas):
    """Return all six joint frames, the Flange frame, and the Tool frame."""

    thetas = _angles(thetas)
    T = np.eye(4)
    frames = []

    # Six joint frames
    for i, q in enumerate(thetas):
        theta_offset, d, a, alpha = widowx.DH_PARAMS[i]

        T = T @ dh_transform(
            theta_offset + q,
            d,
            a,
            alpha,
        )

        frames.append(T.copy())

    # Flange frame
    T = T @ widowx.FLANGE_OFFSET
    frames.append(T.copy())

    # Tool frame
    T = T @ widowx.TOOL_OFFSET
    frames.append(T.copy())

    return frames


def _skew(vector):
    """Return the skew-symmetric matrix of a 3D vector."""

    x, y, z = vector

    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def _screw_exp(omega, velocity, theta):
    """Return the exponential of a revolute screw axis."""

    omega = np.asarray(omega, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    W = _skew(omega)

    # Rodrigues' formula for the rotational part
    R = (
        np.eye(3)
        + np.sin(theta) * W
        + (1.0 - np.cos(theta)) * (W @ W)
    )

    # Translational part of the screw exponential
    G = (
        theta * np.eye(3)
        + (1.0 - np.cos(theta)) * W
        + (theta - np.sin(theta)) * (W @ W)
    )

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = G @ velocity

    return T


def fk_poe(thetas):
    """Return the Tool pose using the product-of-exponentials method.

    The screw axes are defined independently from the URDF in
    robot.widowx.SCREW_AXES.
    """

    thetas = _angles(thetas)

    # Home pose of the Tool frame.
    M = fk(np.zeros(widowx.N_JOINTS))

    T = np.eye(4)

    for screw, theta in zip(widowx.SCREW_AXES, thetas):
        velocity = screw[:3]
        omega = screw[3:]

        T = T @ _screw_exp(
            omega,
            velocity,
            theta,
        )

    return T @ M


def print_pose(T):
    """Print a homogeneous transform as position and RPY orientation."""

    x, y, z, roll, pitch, yaw = pose_to_xyz_rpy(T)

    print("\nTransformation matrix:")
    print(T)

    print("\nTool position:")
    print(f"x = {x * 1000:.2f} mm")
    print(f"y = {y * 1000:.2f} mm")
    print(f"z = {z * 1000:.2f} mm")

    print("\nTool orientation:")
    print(f"roll  = {np.degrees(roll):.2f} degrees")
    print(f"pitch = {np.degrees(pitch):.2f} degrees")
    print(f"yaw   = {np.degrees(yaw):.2f} degrees")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate WidowX AI forward kinematics"
    )

    parser.add_argument(
        "angles",
        nargs=6,
        type=float,
        metavar=("J0", "J1", "J2", "J3", "J4", "J5"),
        help="Six joint angles; degrees by default",
    )

    parser.add_argument(
        "--radians",
        action="store_true",
        help="Interpret the input angles as radians",
    )

    args = parser.parse_args()

    q = np.asarray(args.angles, dtype=float)

    if not args.radians:
        q = np.radians(q)

    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    print_pose(fk(q))
