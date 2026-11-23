"""Resolved-rate motion control: a task-space velocity in, a joint trajectory out.

The loop is fixed. What changes near a singularity is how the Jacobian is inverted, and that
choice is the substance of Part C.
"""

import numpy as np

from kinematics.jacobian import jacobian

METHODS = ("inverse", "dls")


def joint_rates(J, zeta, method="inverse", lam=1e-2):
    """Joint velocities producing the commanded end-effector velocity.

        "inverse"   solve(J, zeta)                          exact, undefined at a singularity
        "dls"       J.T @ inv(J @ J.T + lam^2 I) @ zeta     damped, defined everywhere

    The arm has six joints and the task six components, so J is square and the direct inverse is
    available away from singular configurations. Prefer numpy.linalg.solve to forming an explicit
    inverse: fewer operations and better conditioned.

    lam**2 I keeps the matrix invertible where J J^T is not, so the result stays bounded through a
    singularity. lam = 0 recovers the undamped solution.
    """
    J = np.asarray(J, dtype=float)
    zeta = np.asarray(zeta, dtype=float).reshape(-1)
    if J.shape != (6, 6):
        raise ValueError(f"Expected J with shape (6, 6), got {J.shape}")
    if zeta.size != 6:
        raise ValueError(f"Expected zeta with length 6, got {zeta.size}")
    if not np.all(np.isfinite(J)) or not np.all(np.isfinite(zeta)):
        raise ValueError("J and zeta must be finite")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")

    if method == "inverse":
        return np.linalg.solve(J, zeta)

    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be a nonnegative finite number")
    regularized = J @ J.T + lam * lam * np.eye(6)
    return J.T @ np.linalg.solve(regularized, zeta)


def straight_line(start_pose, end_pose, steps):
    """A straight line in task space between two poses, as a sequence of commanded velocities.

    Part B holds orientation constant, so only the position interpolates. Return whatever the
    loop below needs: the commanded velocity at each step, or the sequence of desired poses.
    """
    start_pose = np.asarray(start_pose, dtype=float).reshape(-1)
    end_pose = np.asarray(end_pose, dtype=float).reshape(-1)
    if start_pose.size not in (3, 6) or end_pose.size != start_pose.size:
        raise ValueError("start_pose and end_pose must both contain 3 or 6 values")
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not np.all(np.isfinite(start_pose)) or not np.all(np.isfinite(end_pose)):
        raise ValueError("poses must be finite")

    poses = np.linspace(start_pose, end_pose, int(steps))
    if start_pose.size == 6:
        # Part B keeps orientation fixed; preserve the supplied start orientation.
        poses[:, 3:] = start_pose[3:]
    return poses


def follow(thetas0, zetas, dt, method="inverse", lam=1e-2):
    """Integrate joint velocities into a trajectory.

    At each step: build J at the current configuration, map the commanded velocity to joint
    rates, and take theta <- theta + theta_dot * dt.

    Return the joint trajectory and, alongside it, the commanded joint rates at every step. Part C
    plots those rates for the direct inverse and for damped least squares along the same path, and
    the comparison is the point: without the record there is nothing to plot.
    """
    thetas0 = np.asarray(thetas0, dtype=float).reshape(-1)
    zetas = np.asarray(zetas, dtype=float)
    if thetas0.size != 6:
        raise ValueError(f"Expected 6 initial joint angles, got {thetas0.size}")
    if zetas.ndim != 2 or zetas.shape[1] != 6:
        raise ValueError(f"Expected zetas with shape (N, 6), got {zetas.shape}")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite number")
    if not np.all(np.isfinite(thetas0)) or not np.all(np.isfinite(zetas)):
        raise ValueError("thetas0 and zetas must be finite")

    trajectory = np.empty((len(zetas) + 1, 6), dtype=float)
    rates = np.empty((len(zetas), 6), dtype=float)
    trajectory[0] = thetas0
    for k, zeta in enumerate(zetas):
        rates[k] = joint_rates(
            jacobian(trajectory[k]),
            zeta,
            method=method,
            lam=lam,
        )
        trajectory[k + 1] = trajectory[k] + dt * rates[k]

    return trajectory, rates
