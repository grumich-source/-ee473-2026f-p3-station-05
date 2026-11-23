"""Simulate resolved-rate motion along a straight Cartesian path."""

import argparse

import numpy as np

from kinematics.control import follow
from kinematics.fk import fk
from robot.widowx import N_JOINTS


def simulate(start_degrees, target, duration=2.0, dt=0.01):
    """Return commanded positions, realized positions, joint trajectory, and rates."""

    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive")
    start = np.radians(np.asarray(start_degrees, dtype=float))
    target = np.asarray(target, dtype=float)
    if start.shape != (N_JOINTS,) or target.shape != (3,):
        raise ValueError("start must contain 6 angles and target must contain 3 coordinates")

    steps = max(1, int(round(duration / dt)))
    dt_actual = duration / steps
    start_position = fk(start)[:3, 3]
    velocity = (target - start_position) / duration
    zetas = np.zeros((steps, 6), dtype=float)
    zetas[:, :3] = velocity
    trajectory, rates = follow(start, zetas, dt_actual, method="inverse")
    realized = np.array([fk(q)[:3, 3] for q in trajectory])
    commanded = np.linspace(start_position, target, steps + 1)
    deviation = np.linalg.norm(realized - commanded, axis=1)
    return commanded, realized, trajectory, rates, deviation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", nargs=6, type=float, required=True, metavar="DEG")
    parser.add_argument("--target", nargs=3, type=float, required=True, metavar="M")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--plot", type=str, help="save a 3-D commanded/realized path plot")
    parser.add_argument("--save", type=str, help="save the trajectory and results as a NumPy .npz file")
    args = parser.parse_args()

    commanded, realized, trajectory, rates, deviation = simulate(
        args.start, args.target, args.duration, args.dt
    )
    print(f"steps: {len(realized) - 1}")
    print(f"start position (m): {commanded[0]}")
    print(f"target position (m): {commanded[-1]}")
    print(f"maximum path deviation: {np.max(deviation) * 1000:.6f} mm")

    if args.save:
        np.savez(
            args.save,
            commanded=commanded,
            realized=realized,
            trajectory=trajectory,
            rates=rates,
            deviation=deviation,
            start_degrees=np.asarray(args.start),
            target=np.asarray(args.target),
            duration=args.duration,
            dt=args.duration / max(1, len(realized) - 1),
        )
        print(f"saved trajectory: {args.save}")

    if args.plot:
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axis = figure.add_subplot(111, projection="3d")
        axis.plot(*commanded.T, "k--", label="commanded")
        axis.plot(*realized.T, "b", label="realized")
        axis.scatter(*commanded[0], color="green", label="start")
        axis.scatter(*commanded[-1], color="red", label="target")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")
        # Use equal physical scaling so the 3-D perspective does not visually
        # exaggerate the small tracking error.  Disable offset notation, which
        # is especially confusing here because y is numerically near zero.
        points = np.vstack((commanded, realized))
        centre = points.mean(axis=0)
        span = np.ptp(points, axis=0)
        half_width = max(float(np.max(span)) / 2.0, 1e-3)
        axis.set_xlim(centre[0] - half_width, centre[0] + half_width)
        axis.set_ylim(centre[1] - half_width, centre[1] + half_width)
        axis.set_zlim(centre[2] - half_width, centre[2] + half_width)
        axis.set_box_aspect((1, 1, 1))
        axis.ticklabel_format(style="plain", axis="both", useOffset=False)
        axis.ticklabel_format(style="plain", axis="z", useOffset=False)
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.plot, dpi=150)
        print(f"saved plot: {args.plot}")


if __name__ == "__main__":
    main()
