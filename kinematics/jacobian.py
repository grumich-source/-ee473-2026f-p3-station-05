"""The geometric Jacobian, and the check that it is right.

Conventions, from §3 of the instructions. The Jacobian maps joint rates to the linear velocity of
the end effector over its angular velocity, both expressed in the base frame. Record that choice
wherever it matters: it governs how the velocity command is formed in Part B.

Angles in radians, lengths in metres, joint order always robot.widowx.JOINT_NAMES.
"""

import numpy as np


def jacobian(thetas):
    """The 6 x n geometric Jacobian at a configuration.

    For a revolute joint the i-th column is

        [ z_(i-1) x (o_n - o_(i-1)) ; z_(i-1) ]

    Every quantity is already in your forward kinematics: z_(i-1) is the third column and
    o_(i-1) the fourth column of the transformation from the base frame to frame i-1, which is a
    partial product you compute on the way to the tool pose. Returning those intermediate frames
    from fk_all_frames() is the cheapest way to get them.

    Returns a 6 x 6 array for this arm. Raise ValueError on the wrong number of angles.
    """
    raise NotImplementedError


def jacobian_fd(thetas, h=1e-6):
    """The same Jacobian by central differences, for the verification in §3.

        J_fd[:, i] = ( f(theta + h e_i) - f(theta - h e_i) ) / (2h)

    The position rows follow from a difference of position vectors. The orientation rows do not:
    the difference of two rotation matrices is not a rotation. Form R_pert R^T instead, take its
    axis and angle by the matrix logarithm on SO(3), and divide by h.

    The error does not fall monotonically as h shrinks. Two effects compete, truncation at large
    h and floating-point cancellation at small h, and the h you report should reflect that.
    """
    raise NotImplementedError


def manipulability(J):
    """sqrt(det(J J^T)), a measure of how near a configuration is to singular.

    Zero exactly at a singularity and small near one. det J = 0 is a yes/no test that is never
    satisfied in floating point, so this, or the smallest singular value of J, is what can be
    plotted along a path. Part C item 2 wants that plot.
    """
    raise NotImplementedError
