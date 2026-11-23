"""Frame algebra. Nothing in this file knows which robot it is operating on.

Any dimension particular to the WidowX AI belongs in robot/widowx.py.
"""

import numpy as np


def dh_transform(theta, d, a, alpha):
    """One Denavit-Hartenberg link transform, as a 4x4 homogeneous matrix.

    Which four elementary operations this is a product of, and in what order, depends on whether
    the classical or the modified convention is in use. Implement the one declared in
    robot/widowx.py, and implement it once: every joint passes through this function.

    Angles in radians, lengths in metres.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def rot_x(angle):
    """Rotation about x, as a 4x4."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def rot_y(angle):
    """Rotation about y, as a 4x4."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def rot_z(angle):
    """Rotation about z, as a 4x4."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def translation(x, y, z):
    """Pure translation, as a 4x4."""
    H = np.eye(4, dtype=float)
    H[:3, 3] = [x, y, z]
    return H


def inverse(H):
    """Inverse of a homogeneous transform.

    A homogeneous transformation inverts in closed form, by way of the orthogonality of its
    rotation block. Derive that expression and implement it rather than calling np.linalg.inv: it
    is both faster and more numerically stable, and the derivation repays doing once.
    """
    H = np.asarray(H, dtype=float)
    R = H[:3, :3]
    p = H[:3, 3]

    H_inv = np.eye(4, dtype=float)
    H_inv[:3, :3] = R.T
    H_inv[:3, 3] = -R.T @ p
    return H_inv


def pose_to_xyz_rpy(H):
    """Convenience: pull (x, y, z, roll, pitch, yaw) out of a 4x4 for printing and comparison.

    Observe the convention. Fixed-axis roll-pitch-yaw and Euler angles are distinct
    parameterizations, and whichever is adopted here governs the measurements as well.
    """
    H = np.asarray(H, dtype=float)
    x, y, z = H[:3, 3]
    R = H[:3, :3]

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    pitch = np.arctan2(
        -R[2, 0],
        np.hypot(R[0, 0], R[1, 0]),
    )

    if np.hypot(R[0, 0], R[1, 0]) > 1e-8:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # At gimbal lock, roll is not unique; choose roll = 0.
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])

    return x, y, z, roll, pitch, yaw


def pose_from_xyz_rpy(x, y, z, roll, pitch, yaw):
    """Build a 4x4 pose from a position and three orientation angles.

    This is how a target is specified. `ik()` accepts a 4x4 transformation, and Part C begins
    here: select a position and an approach orientation, construct the matrix, and solve it.

    Compose it from the helpers above rather than writing out the entries directly.

    Fix the convention here and state it in your report. Fixed-axis roll-pitch-yaw and Euler
    angles are different parameterizations, and the order in which the three rotations are
    applied is part of the convention. Whichever is adopted, pose_to_xyz_rpy must be its exact
    inverse, and measurements must be taken under the same one.
    """
    values = np.asarray([x, y, z, roll, pitch, yaw], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Pose position and orientation must be finite numbers")

    x, y, z, roll, pitch, yaw = values
    H = translation(x, y, z)
    H[:3, :3] = (
        rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    )[:3, :3]
    return H


def poses_close(H1, H2, pos_tol=1e-6, rot_tol=1e-6):
    """True if two poses agree to within tolerance.

    Used by the round-trip test. An element-by-element comparison of 4x4 matrices mixes metres
    with dimensionless rotation entries, so separate the two: compare the translation columns in
    metres and the rotation blocks by angle.
    """
    H1 = np.asarray(H1, dtype=float)
    H2 = np.asarray(H2, dtype=float)

    position_error = np.linalg.norm(H1[:3, 3] - H2[:3, 3])
    R_diff = H1[:3, :3].T @ H2[:3, :3]
    cos_angle = (np.trace(R_diff) - 1.0) / 2.0
    rotation_error = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    return bool(position_error <= pos_tol and rotation_error <= rot_tol)
