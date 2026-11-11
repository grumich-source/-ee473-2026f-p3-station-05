"""A verified forward kinematics for the WidowX AI, for teams whose own does not reproduce the
reference poses in section 2.1 of the project description.

Use this only if you need it, and say in the report that you did. It costs nothing in Project 3,
which grades the Jacobian and the control loops rather than the kinematics underneath them. It is
not a better answer than your own table, only a checked one.

The numbers were read from wxai_follower.urdf and verified two ways: against the raw URDF joint
origin chain, agreeing to 1.1e-15 m over 4000 random configurations inside the joint limits, and
against the eight poses tabulated in the project description.

This module is deliberately standalone. It imports math and numpy and nothing from the rest of the
package, so it works whatever state your own kinematics is in. The frames it returns are its own
Denavit-Hartenberg frames, which are not the URDF link frames and are not your frames either. That
does not matter: a Jacobian column needs the direction of its joint axis and any one point on that
axis, so every valid frame assignment gives the same Jacobian.

Interface, matching what jacobian.py expects:

    fk(thetas)              -> 4x4 pose of the Tool frame in the base frame
    fk_all_frames(thetas)   -> frames 1..6 then Tool, each a 4x4 in the base frame
"""

import math

import numpy as np

JOINT_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
N_JOINTS = len(JOINT_NAMES)

DH_CONVENTION = "classic"

# (theta_offset, d, a, alpha);  theta_i = q_i + theta_offset.  Radians and metres.
# Row 3 carries the 60 mm elbow offset folded into the link length and the angle offset, which is
# why its a is a hypotenuse and its offset is not a round number.
_ELBOW = math.atan2(0.060, 0.245)          # 0.2401706 rad
DH_PARAMS = [
    (0.0,                 0.1035, 0.020,                    -math.pi / 2),
    (math.pi,             0.0,    0.264,                     math.pi),
    (-(math.pi - _ELBOW), 0.0,    math.hypot(0.245, 0.060),  0.0),
    (-_ELBOW,             0.0,    0.06775,                   math.pi / 2),
    (math.pi / 2,         0.0,    0.0,                       math.pi / 2),
    (0.0,                 0.0,    0.0,                       0.0),
]

# Frame 6 to Tool. Carries the flange rotation and the 156.062 mm tool extension together, so the
# 28.95 mm that Trossen's published table puts in d6 appears here instead. Both reach the same
# tool pose; only the orientation of the intermediate frame 6 differs, and nothing uses it.
TOOL_OFFSET = np.array([
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [1.0,  0.0,  0.0, 0.185012],
    [0.0,  0.0,  0.0, 1.0],
])

# From the URDF. Note joint_0: the URDF says +-175 degrees, the specification page says +-180.
# The URDF is what the driver enforces, so it is what this uses.
JOINT_LIMITS = [
    (-3.0543261909900767, 3.0543261909900767),
    (0.0, math.pi),
    (0.0, 2.356194490192345),
    (-math.pi / 2, math.pi / 2),
    (-math.pi / 2, math.pi / 2),
    (-math.pi, math.pi),
]

REST = [0.0, math.pi / 2, math.pi / 2, 0.0, 0.0, 0.0]


def within_limits(thetas):
    """(ok, names_of_violating_joints)."""
    _check(thetas)
    bad = [n for n, t, (lo, hi) in zip(JOINT_NAMES, thetas, JOINT_LIMITS) if not lo <= t <= hi]
    return (not bad), bad


def home():
    return [0.0] * N_JOINTS


def rest():
    return list(REST)


def _check(thetas):
    if len(thetas) != N_JOINTS:
        raise ValueError(f"expected {N_JOINTS} angles, got {len(thetas)}")


def dh_transform(theta, d, a, alpha):
    """One classical link transform: Rot_z(theta) Trans_z(d) Trans_x(a) Rot_x(alpha)."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def fk(thetas):
    """Pose of the Tool frame in the base frame, as a 4x4."""
    _check(thetas)
    H = np.eye(4)
    for q, (off, d, a, alpha) in zip(thetas, DH_PARAMS):
        H = H @ dh_transform(q + off, d, a, alpha)
    return H @ TOOL_OFFSET


def fk_all_frames(thetas):
    """Frames 1..6 then the Tool frame, each expressed in the base frame.

    Part A needs the z axis and the origin of each of these. Frame 0 is the base frame itself, so
    it is the identity and is not in this list; build the Jacobian's first column from it directly.
    """
    _check(thetas)
    H = np.eye(4)
    frames = []
    for q, (off, d, a, alpha) in zip(thetas, DH_PARAMS):
        H = H @ dh_transform(q + off, d, a, alpha)
        frames.append(H.copy())
    frames.append(H @ TOOL_OFFSET)
    return frames


# The eight poses from section 2.1, as (name, q_degrees, xyz_mm, rpy_rad).
REFERENCE_POSES = [
    ("home",               [0, 0, 0, 0, 0, 0],
     (253.762, 0.000, 163.500), (0.0000, 0.0000, 0.0000)),
    ("rest",               [0, 90, 90, 0, 0, 0],
     (517.762, 0.000, 427.500), (0.0000, 0.0000, 0.0000)),
    ("waist only",         [45, 0, 0, 0, 0, 0],
     (179.437, 179.437, 163.500), (0.0000, 0.0000, 0.7854)),
    ("shoulder and elbow", [0, 60, 45, 0, 0, 0],
     (384.330, 0.000, 261.256), (0.0000, 0.2618, 0.0000)),
    ("wrist exercised",    [0, 45, 90, 30, -40, 60],
     (18.355, 118.923, 708.183), (-0.1286, -0.8330, 1.2716)),
    ("general A",          [30, 75, 60, -45, 35, -90],
     (324.562, 64.851, 163.127), (-2.3529, 0.7887, -0.4271)),
    ("general B",          [-120, 20, 120, 80, -80, 150],
     (372.585, 280.934, 424.652), (-0.5236, 0.0000, -0.3491)),
    ("near limits",        [-175, 0, 135, -90, 90, 180],
     (394.069, 220.195, 282.221), (-2.3562, 0.0000, 1.6581)),
]


def check(fk_function=fk, tol_p=1e-5, tol_r=2e-4):
    """Compare a forward kinematics against the eight reference poses.

    Pass your own fk to test it:

        python3 -c "from robot.widowx_reference import check; from kinematics.fk import fk; check(fk)"

    The tolerances are loose because REFERENCE_POSES holds the values as tabulated in the project
    description, rounded to 0.001 mm and 0.0001 rad. They are still a sharp test: a sign error or a
    dropped offset misses by millimetres, not by hundredths of one.
    """
    worst_p = worst_r = 0.0
    ok = True
    for name, q_deg, xyz_mm, rpy_rad in REFERENCE_POSES:
        q = [math.radians(v) for v in q_deg]
        H = np.asarray(fk_function(q), dtype=float)
        dp = np.abs(H[:3, 3] - np.array(xyz_mm) / 1000.0).max()
        dr = max(abs(_wrap(a - b)) for a, b in zip(_rpy(H[:3, :3]), rpy_rad))
        worst_p, worst_r = max(worst_p, dp), max(worst_r, dr)
        good = dp <= tol_p and dr <= tol_r
        ok = ok and good
        print(f"{'ok  ' if good else 'FAIL'}  {name:19s} "
              f"position {dp * 1000:9.6f} mm   orientation {dr:9.6f} rad")
    print(f"\nworst: {worst_p * 1000:.6f} mm, {worst_r:.6f} rad   "
          f"=> {'PASS' if ok else 'does not match the reference'}")
    return ok


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _rpy(R):
    """Fixed-axis XYZ: roll about x, then pitch about y, then yaw about z."""
    if abs(R[2, 0]) < 1.0 - 1e-12:
        return (math.atan2(R[2, 1], R[2, 2]),
                math.asin(-R[2, 0]),
                math.atan2(R[1, 0], R[0, 0]))
    return (math.atan2(-R[1, 2], R[1, 1]),
            math.pi / 2 * (-1.0 if R[2, 0] > 0 else 1.0),
            0.0)


if __name__ == "__main__":
    check()
