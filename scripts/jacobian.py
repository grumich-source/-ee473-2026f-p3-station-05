"""Print the WidowX Jacobian for six joint angles given in radians."""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make direct execution from the repository root work:
# python3 scripts/jacobian.py ...
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kinematics.jacobian import jacobian, jacobian_fd, manipulability


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the WidowX Jacobian from six joint angles in radians."
    )
    parser.add_argument(
        "angles",
        nargs=6,
        type=float,
        metavar="RAD",
        help="six joint angles in radians, in robot joint order",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=1e-6,
        help="finite-difference step size in radians (default: 1e-6)",
    )
    args = parser.parse_args()

    angles = np.array(args.angles, dtype=float)
    analytic = jacobian(angles)
    finite_difference = jacobian_fd(angles, h=args.h)

    np.set_printoptions(precision=3, suppress=True)
    print("Joint angles (rad):")
    print(angles)
    print("\nAnalytic Jacobian:")
    print(analytic)
    print("\nFinite-difference Jacobian:")
    print(finite_difference)
    print("\nMaximum absolute difference:")
    print(f"{np.max(np.abs(analytic - finite_difference)):.3f}")
    print("\nManipulability:")
    print(f"{manipulability(analytic):.3f}")


if __name__ == "__main__":
    main()
