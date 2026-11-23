"""Everything that knows this is a WidowX AI.

This is the **only** file in the project that may contain hardware-specific numbers. Joint
limits, link lengths, DH parameters, the tool frame definition, and the network interface: all of
it lives here, and nothing outside this file may hard-code a dimension.

The boundary has a payoff in Project 3, which extends your kinematics/ package to velocity
control on this same arm. Where the boundary has held, that is an addition. Where link lengths
have leaked into kinematics/, it is a rewrite.

The values come from the Part A derivation. The structure is given; the numbers are the
assignment.
"""

import numpy as np

# Joint order. Every list of angles in this project is in this order, without exception.
# These are the names Trossen's description uses; keep them so your code and the URDF agree.
JOINT_NAMES = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
]

N_JOINTS = len(JOINT_NAMES)


# ----------------------------------------------------------------------------
# DH parameters
# ----------------------------------------------------------------------------
# One row per joint, in JOINT_NAMES order. The instructions specify the CLASSICAL convention.
# Trossen publishes both classical and modified tables; their rows do not correspond, so make
# sure you are reading the right one before you compare.
#
# Each row is the CONSTANT part of the row. The joint variable is added at evaluation time.
# This arm is all revolute.
#
#   (theta_offset, d, a, alpha)      angles in radians, lengths in metres

DH_CONVENTION = "classic"

DH_PARAMS = [
    # (theta_offset, d, a, alpha)
    (0.0, 0.1035, 0.02, -(np.pi)/2),                                               # joint_0
    (np.pi, 0.0, 0.264, np.pi),                                                  # joint_1
    (-(np.pi/2)-np.arctan(0.245/0.06), 0.0, np.sqrt((0.245**2)+(0.06**2)), 0.0),   # joint_2
    (-np.arctan(0.06/0.245), 0.0, 0.06775, np.pi/2),                             # joint_3
    (np.pi/2, 0.0, 0.0, np.pi/2),                                                # joint_4
    (0.0, 0.02895, 0.0, np.pi/2),                                                # joint_5
]


# ----------------------------------------------------------------------------
# Tool frame
# ----------------------------------------------------------------------------
# Trossen's chain does not end at joint 6. It continues through a Flange frame and then to Tool,
# which is the frame the instructions prescribe. That is two fixed transformations: a rotation
# carrying frame 6 into the flange, and a translation of 156.06 mm from the flange to the tool
# point. A table that stops at joint 6 is short by that whole segment, and it shows up as a large
# constant position error at every configuration.

# 4x4 numpy array: frame 6 -> Tool

FLANGE_OFFSET = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [1.0,  0.0, 0.0, 0.0],
    [0.0,  0.0, 1.0, 0.0],
    [0.0,  0.0, 0.0, 1.0],
])

TOOL_OFFSET = np.array([
    [1.0, 0.0, 0.0, 0.15606],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

# ----------------------------------------------------------------------------
# Joint limits
# ----------------------------------------------------------------------------
# From the manufacturer, in radians. Do not guess these: read them from wxai_follower.urdf in the
# trossen_arm_description package, or from Trossen's documentation. Part B requires you to flag
# solutions that violate them.

JOINT_LIMITS = [
    # (lower, upper)
    (-3.0543261909900767, 3.0543261909900767),  # joint_0
    (0.0, np.pi),                              # joint_1
    (0.0, 2.356194490192345),                  # joint_2
    (-np.pi / 2, np.pi / 2),                   # joint_3
    (-np.pi / 2, np.pi / 2),                   # joint_4
    (-np.pi, np.pi),                           # joint_5
]


def within_limits(thetas):
    """True if every joint angle is inside its limit.

    Returns a (bool, list_of_offending_joint_names) pair so the caller can report *which* joint
    is out of range rather than just that something is.
    """
    if len(thetas) != N_JOINTS:
        raise ValueError(
            f"Expected {N_JOINTS} joint angles, got {len(thetas)}"
        )

    offending = []

    for name, angle, limits in zip(JOINT_NAMES, thetas, JOINT_LIMITS):
        lower, upper = limits

        if lower is None or upper is None:
            raise ValueError(
                f"Joint limits for {name} have not been defined"
            )

        if not np.isfinite(angle) or angle < lower or angle > upper:
            offending.append(name)

    return len(offending) == 0, offending


def home():
    """The all-zeros configuration, as a list of N_JOINTS angles.

    Establish what this configuration looks like physically before commanding it.
    """
    return [0.0] * N_JOINTS


# ----------------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------------
# Part C only. Everything below talks to the physical arm, and this is the only file in the
# project allowed to. Nothing in kinematics/ may import any of it.
#
# The controller resides on the laboratory network and the trossen_arm driver connects to it
# directly over Ethernet, in contrast to the ROS stack used for the RX150 in Project 1. No launch
# file is involved.
#
# Import trossen_arm WITHIN the methods rather than at module level. The test suite must run on a
# machine with no manipulator attached, and a module-level import would cause `pytest` to fail at
# collection with an ImportError unrelated to the code under test.
#
# The SDK version must match the controller firmware. See requirements.txt.

ARM_IP = "192.168.1.2"


class Arm:
    """A connection to the physical WidowX AI.

    Employ it as a context manager, so that the manipulator is left in a safe state even where a
    script raises partway through:

        with Arm() as arm:
            arm.send(solution.thetas)
            measured = arm.read()

    Implement `__enter__` by constructing a TrossenArmDriver and calling configure() with the
    model, the follower end effector, ARM_IP, and the clear-error flag. Implement `__exit__` by
    bringing the arm somewhere safe and calling cleanup().

    Review the safety note in Part C before connecting. This manipulator is heavier and faster
    than the RX150, and a sign error in a commanded pose carries correspondingly greater
    consequences.
    """

    def __init__(self, goal_time=3.0):
        self.goal_time = goal_time
        self.driver = None

    def __enter__(self):
        import trossen_arm

        if self.driver is not None:
            raise RuntimeError("Arm is already connected")

        self.driver = trossen_arm.TrossenArmDriver()

        try:
            self.driver.configure(
                trossen_arm.Model.wxai_v0,
                trossen_arm.StandardEndEffector.wxai_v0_follower,
                ARM_IP,
                False,
            )

            # Put the six arm joints in position mode.
            self.driver.set_arm_modes(trossen_arm.Mode.position)
        except Exception:
            self.driver.cleanup()
            self.driver = None
            raise

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.driver is not None:
            self.driver.cleanup()

        self.driver = None

    def send(self, thetas):
        thetas = np.asarray(thetas, dtype=float)

        if thetas.shape != (N_JOINTS,):
            raise ValueError(
                f"Expected {N_JOINTS} joint angles, got shape {thetas.shape}"
            )

        if self.driver is None:
            raise RuntimeError("Arm is not connected; use it inside 'with Arm()'")

        valid, offending = within_limits(thetas)

        if not valid:
            raise ValueError(
                f"Joint angles outside limits: {offending}"
            )

        self.driver.set_arm_positions(
            thetas.tolist(),
            self.goal_time,
            True,
        )

    def driver_ee_pose(self):
        if self.driver is None:
            raise RuntimeError("Arm is not connected; use it inside 'with Arm()'")

        values = self.driver.get_cartesian_positions()

        # Returns x, y, z, rx, ry, rz
        values = np.asarray(values, dtype=float)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("Driver returned an invalid Cartesian pose")
        return values

    def read(self):
        # Returns only the six arm joints, excluding the gripper.
        if self.driver is None:
            raise RuntimeError("Arm is not connected; use it inside 'with Arm()'")

        positions = self.driver.get_arm_positions()
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (N_JOINTS,) or not np.all(np.isfinite(positions)):
            raise ValueError("Driver returned an invalid joint configuration")
        return positions.tolist()


# Space-frame screw axes from wxai_follower.urdf at the zero configuration.
# Each row is [vx, vy, vz, wx, wy, wz].

SCREW_AXES = np.array([
    [0.0000,  0.0000,  0.0000,  0.0,  0.0,  1.0],
    [-0.1035, 0.0000,  0.0200,  0.0,  1.0,  0.0],
    [0.1035,  0.0000,  0.2440,  0.0, -1.0,  0.0],
    [0.1635,  0.0000, -0.0010, 0.0, -1.0, 0.0],
    [0.0000,  0.06875, 0.0000, 0.0, 0.0, -1.0],
    [0.0000,  0.1635, 0.0000, 1.0, 0.0, 0.0],
])
