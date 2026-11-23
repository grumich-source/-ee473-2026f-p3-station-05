"""Run the saved, validated Part B trajectory on the physical WidowX AI.

This script is intentionally dry-run by default.  Hardware motion requires the
explicit --execute flag after the team has completed the safety checks.
"""

import argparse

import numpy as np

from robot.widowx import Arm, JOINT_LIMITS, N_JOINTS


REST = np.array([0.0, np.pi / 2.0, np.pi / 2.0, 0.0, 0.0, 0.0])


def load_trajectory(path, max_velocity):
    data = np.load(path)
    trajectory = np.asarray(data["trajectory"], dtype=float)
    rates = np.asarray(data["rates"], dtype=float)
    dt = float(data["dt"])

    if trajectory.ndim != 2 or trajectory.shape[1] != N_JOINTS:
        raise ValueError("trajectory must have shape (N, 6)")
    if rates.shape != (len(trajectory) - 1, N_JOINTS):
        raise ValueError("rates do not match trajectory length")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("saved timestep must be positive and finite")
    if not np.all(np.isfinite(trajectory)) or not np.all(np.isfinite(rates)):
        raise ValueError("saved trajectory contains non-finite values")
    if np.max(np.abs(rates)) > max_velocity:
        raise ValueError(
            f"trajectory requires {np.max(np.abs(rates)):.4f} rad/s, "
            f"above configured limit {max_velocity:.4f} rad/s"
        )

    for joint, (lower, upper) in enumerate(JOINT_LIMITS):
        if np.any(trajectory[:, joint] < lower) or np.any(trajectory[:, joint] > upper):
            raise ValueError("trajectory contains a joint-limit violation")

    return trajectory, rates, dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="saved Part B .npz trajectory")
    parser.add_argument("--initial-goal-time", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=1,
                        help="send every Nth saved waypoint (default: 1)")
    parser.add_argument("--speed-scale", type=float, default=1.0,
                        help="multiply saved trajectory speed; default 1.0")
    parser.add_argument("--max-velocity", type=float, default=0.25)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="enable physical motion; without this flag only checks are performed",
    )
    args = parser.parse_args()

    if args.initial_goal_time <= 0 or args.max_velocity <= 0 or args.speed_scale <= 0:
        raise ValueError("initial-goal-time, max-velocity, and speed-scale must be positive")
    if args.stride < 1:
        raise ValueError("stride must be a positive integer")

    trajectory, rates, dt = load_trajectory(args.trajectory, args.max_velocity)
    segment_time = dt * args.stride / args.speed_scale
    command_points = trajectory[args.stride::args.stride]
    if len(command_points) == 0 or not np.array_equal(command_points[-1], trajectory[-1]):
        command_points = np.vstack((command_points, trajectory[-1]))
    command_rates = np.diff(np.vstack((trajectory[0], command_points)), axis=0) / segment_time
    effective_max_velocity = float(np.max(np.abs(command_rates)))
    if effective_max_velocity > args.max_velocity:
        raise ValueError(
            f"decimated commands require {effective_max_velocity:.4f} rad/s, "
            f"above configured limit {args.max_velocity:.4f} rad/s; "
            "reduce --stride or increase --goal-time"
        )
    print(f"trajectory: {args.trajectory}")
    print(f"saved points: {len(trajectory)}, command points: {len(command_points)}, saved dt: {dt:.4f} s")
    print(f"maximum saved joint speed: {np.max(np.abs(rates)):.4f} rad/s")
    print(f"segment goal-time: {segment_time:.4f} s")
    print(f"speed scale: {args.speed_scale:.3f}")
    print(f"maximum hardware command speed: {effective_max_velocity:.4f} rad/s")
    print("joint limits: PASS")

    if not args.execute:
        print("DRY RUN: no hardware connection or motion requested")
        print("Re-run with --execute only after completing the team safety checks.")
        return

    print("WARNING: physical arm motion is enabled.")
    print("Confirm the workspace is clear, the power disconnect operator is ready,")
    print("and the team has verified this is the Part B trajectory.")
    if input("Type RUN to continue: ").strip() != "RUN":
        print("Cancelled; no hardware motion sent.")
        return

    # Keep the driver connected while the finally block sends the required
    # rest pose.  Using ``with Arm(...)`` here would call __exit__ first on an
    # interrupt, disconnecting before the safety command could be sent.
    arm = Arm(goal_time=segment_time)
    connected = False
    try:
        arm.__enter__()
        connected = True
        # Repositioning from an arbitrary current pose is separate from the
        # Part B trajectory and must be much slower.
        arm.goal_time = args.initial_goal_time
        arm.send(REST)
        arm.goal_time = segment_time
        for point in command_points:
            arm.send(point)
    except KeyboardInterrupt:
        print("Interrupted; sending the rest pose before disconnecting.")
    finally:
        if connected and arm.driver is not None:
            try:
                arm.goal_time = args.initial_goal_time
                arm.send(REST)
            except Exception as exc:
                print(f"WARNING: could not send rest pose during cleanup: {exc}")
            finally:
                arm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
