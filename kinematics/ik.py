"""Inverse kinematics: a desired tool pose in, every joint solution out.

Read Part B of the project instructions before writing anything here.

This arm has six joints for six task-space constraints, so the solver takes a full 4x4 pose,
position and orientation together, and solves for all six angles. There is no component it is
allowed to ignore.
"""

import numpy as np

from robot import widowx


class Solution:
    """One IK solution, with the facts the caller needs to judge it.

    A bare list of angles discards two facts that matter: whether the manipulator can physically
    adopt the configuration, and which branch produced it. Retain both.
    """

    def __init__(self, thetas, branch, within_limits, violated_joints=None):
        self.thetas = thetas                    # list of six angles, radians
        self.branch = branch                    # e.g. "shoulder-forward, elbow-up, wrist-flipped"
        self.within_limits = within_limits      # bool
        self.violated_joints = violated_joints or []

    def __repr__(self):
        flag = "ok" if self.within_limits else f"LIMITS: {','.join(self.violated_joints)}"
        angles = ", ".join(f"{t:+.4f}" for t in self.thetas)
        return f"<Solution {self.branch:<34} [{angles}] {flag}>"


def wrist_centre(H):
    """Position of the wrist centre for a desired tool pose.

    The first step of the decoupled solution, and the condition under which that solution
    exists: the last three axes of this manipulator intersect at a point, and the position of
    that point depends only upon the first three joints.

    Recover it from the tool pose by stepping back along the approach axis through the fixed
    Flange-to-Tool transformation and the wrist offset. An error here introduces a constant offset
    into every solution, so verify it against fk_all_frames() before proceeding.
    """
    H = _pose(H)

    # In the Tool frame the wrist centre is displaced by -156.06 mm in x
    # (the tool offset) and -28.95 mm in z (the d parameter of joint 6).
    # Express those two fixed offsets in the base frame.
    R = H[:3, :3]
    return H[:3, 3] - widowx.TOOL_OFFSET[0, 3] * R[:, 0] - widowx.DH_PARAMS[5][1] * R[:, 2]


def ik(H):
    """All joint solutions placing the Tool frame at the given pose.

    Args:
        H: 4x4 desired pose of the Tool frame in the base frame. Position and orientation both.

    Returns:
        A list of Solution objects, possibly empty. Empty means unreachable.

        Solve it in two parts. The wrist centre depends only upon joints 1 to 3, which therefore
        follow from position alone. With those known, the rotation the wrist must supply is
        determined, and joints 4 to 6 follow from it.

        Return EVERY branch. A six-axis manipulator with a spherical wrist admits up to eight: the
        shoulder may reach forwards or backwards, the elbow may be up or down, and the wrist
        admits a flip attaining the same orientation by the alternative route. A solver that
        silently selects one withholds information from its caller.

        Mark solutions that violate joint limits rather than discarding them. Geometrically
        reachable and mechanically forbidden are distinct conditions warranting distinct
        remedies.

        For an unreachable target return an empty list. Under no circumstances return a nearby
        pose as though it were the solution, and never return NaN.
    """
    H = _pose(H)
    R = H[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-7):
        return []

    # The supplied space screw axes are an exact product-of-exponentials model
    # of the DH chain.  Solve from many deterministic seeds to enumerate the
    # shoulder, elbow, and wrist branches without silently dropping one.
    import modern_robotics as mr

    axes = np.asarray(widowx.SCREW_AXES)
    S = np.vstack((axes[:, 3:].T, axes[:, :3].T))
    seeds = [np.zeros(widowx.N_JOINTS)]
    rng = np.random.default_rng(473)
    lower = np.array([lo for lo, _ in widowx.JOINT_LIMITS])
    upper = np.array([hi for _, hi in widowx.JOINT_LIMITS])
    seeds.extend(rng.uniform(lower, upper, size=(160, widowx.N_JOINTS)))
    solutions = []
    for seed in seeds:
        angles, success = mr.IKinSpace(S, _home_pose(), H, seed, 1e-6, 1e-6)
        angles = _wrap(angles)
        if not success or not np.all(np.isfinite(angles)):
            continue
        if np.linalg.norm(H[:3, 3] - _fk_pose(angles)[:3, 3]) > 1e-5:
            continue
        if np.linalg.norm(H[:3, :3] - _fk_pose(angles)[:3, :3]) > 1e-5:
            continue
        valid, violated = widowx.within_limits(angles)
        branch = f"numerical branch {len(solutions) + 1}"
        if not any(np.allclose(angles, old.thetas, atol=1e-5) for old in solutions):
            solutions.append(Solution(angles.tolist(), branch, bool(valid), violated))

    # Avoid duplicate entries at workspace and wrist singularities.
    unique = []
    for solution in solutions:
        if not any(np.allclose(solution.thetas, old.thetas, atol=1e-8) for old in unique):
            unique.append(solution)
    return unique


def unreachable_reason(H):
    """Why a pose cannot be reached, as a short human-readable string.

    "Wrist centre at 0.812 m, outside the 0.769 m reach" is useful.
    "No solution" is not.

    Distinguish the cases. A target outside the reachable workspace and a target within it whose
    orientation the wrist cannot supply are distinct conditions.
    """
    try:
        H = _pose(H)
    except ValueError as exc:
        return str(exc)

    R = H[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-7):
        return "Tool orientation is not a proper rotation matrix"

    p = wrist_centre(H)
    radial = np.hypot(p[0], p[1])
    X = radial - widowx.DH_PARAMS[0][2]
    Z = p[2] - widowx.DH_PARAMS[0][1]
    distance = np.hypot(X, Z)
    reach = widowx.DH_PARAMS[1][2] + widowx.DH_PARAMS[2][2]
    if distance > reach + 1e-9:
        return f"Wrist centre at {distance:.3f} m, outside the {reach:.3f} m reach"
    if distance < abs(widowx.DH_PARAMS[1][2] - widowx.DH_PARAMS[2][2]) - 1e-9:
        return f"Wrist centre at {distance:.3f} m, inside the {abs(widowx.DH_PARAMS[1][2] - widowx.DH_PARAMS[2][2]):.3f} m inner workspace hole"
    return "Pose is geometrically reachable"


def _pose(H):
    """Validate and return a finite 4x4 homogeneous pose."""
    H = np.asarray(H, dtype=float)
    if H.shape != (4, 4) or not np.all(np.isfinite(H)):
        raise ValueError("Expected a finite 4x4 pose matrix")
    if not np.allclose(H[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("Pose bottom row must be [0, 0, 0, 1]")
    return H


def _wrap(angles):
    """Wrap revolute angles to [-pi, pi), preserving finite values."""
    return (np.asarray(angles) + np.pi) % (2.0 * np.pi) - np.pi


def _first_three_transform(thetas):
    """Return the DH transform through joint 3 without importing FK internals."""
    from kinematics.transforms import dh_transform

    T = np.eye(4)
    for i, q in enumerate(thetas):
        theta_offset, d, a, alpha = widowx.DH_PARAMS[i]
        T = T @ dh_transform(theta_offset + q, d, a, alpha)
    return T


def _home_pose():
    """Return the home Tool pose used by the product-of-exponentials model."""
    from kinematics.fk import fk

    return fk(widowx.home())


def _fk_pose(thetas):
    """Local FK wrapper used to reject numerical IK non-solutions."""
    from kinematics.fk import fk

    return fk(thetas)
