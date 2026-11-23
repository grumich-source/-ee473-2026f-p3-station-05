"""Plot manipulability while following a safe Cartesian line near a singular pose."""

import numpy as np

from kinematics.control import follow
from kinematics.fk import fk
from kinematics.jacobian import jacobian, manipulability


def main():
    import matplotlib.pyplot as plt

    # Approach the singular solution from a nearby, valid configuration.
    q_sing = np.array([-0.3998, 1.2352, 2.2944, 0.5220, np.pi / 2, -2.3648])
    _, _, Vh = np.linalg.svd(jacobian(q_sing))
    well_conditioned_joint_direction = Vh[0]
    q0 = q_sing + 0.1 * well_conditioned_joint_direction
    duration = 3.0
    dt = 0.01
    steps = int(round(duration / dt))

    zetas = np.zeros((steps, 6))
    # Use the corresponding well-conditioned task-space direction and aim
    # toward the singular pose. This is a straight line in task/twist space.
    zeta = jacobian(q_sing) @ (-0.1 * well_conditioned_joint_direction / duration)
    zetas[:] = zeta
    trajectory, _ = follow(q0, zetas, dt, method="dls", lam=1e-2)

    positions = np.array([fk(q)[:3, 3] for q in trajectory])
    sigma_min = np.array([np.linalg.svd(jacobian(q), compute_uv=False)[-1]
                          for q in trajectory])
    w = np.array([manipulability(jacobian(q)) for q in trajectory])
    time = np.arange(len(trajectory)) * dt

    print(f"start position (m): {positions[0]}")
    print(f"goal position (m):  {positions[-1]}")
    print(f"minimum sigma_min:  {sigma_min.min():.6e}")
    print(f"minimum manipulability: {w.min():.6e}")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].semilogy(time, sigma_min, color="tab:blue")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel(r"$\sigma_{\min}(J)$")
    axes[0].set_title("Smallest singular value")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(time, w, color="tab:orange")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("manipulability")
    axes[1].set_title(r"$\sqrt{\det(JJ^T)}$")
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig("figures/singularity-path.png", dpi=150)
    print("saved plot: figures/singularity-path.png")


if __name__ == "__main__":
    main()
