"""Resolved-rate motion control: a task-space velocity in, a joint trajectory out.

The loop is fixed. What changes near a singularity is how the Jacobian is inverted, and that
choice is the substance of Part C.
"""

import numpy as np

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
    raise NotImplementedError


def straight_line(start_pose, end_pose, steps):
    """A straight line in task space between two poses, as a sequence of commanded velocities.

    Part B holds orientation constant, so only the position interpolates. Return whatever the
    loop below needs: the commanded velocity at each step, or the sequence of desired poses.
    """
    raise NotImplementedError


def follow(thetas0, zetas, dt, method="inverse", lam=1e-2):
    """Integrate joint velocities into a trajectory.

    At each step: build J at the current configuration, map the commanded velocity to joint
    rates, and take theta <- theta + theta_dot * dt.

    Return the joint trajectory and, alongside it, the commanded joint rates at every step. Part C
    plots those rates for the direct inverse and for damped least squares along the same path, and
    the comparison is the point: without the record there is nothing to plot.
    """
    raise NotImplementedError
