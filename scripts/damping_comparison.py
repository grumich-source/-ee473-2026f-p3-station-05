"""Compare direct-inverse and damped resolved-rate control near a singularity."""

import numpy as np

from kinematics.control import follow
from kinematics.fk import fk


def main():
    import matplotlib.pyplot as plt

    q_sing = np.array([-0.3998, 1.2352, 2.2944, 0.5220, 1.5708, -2.3648])
    q0 = q_sing.copy()
    q0[4] = 1.47
    duration = 3.0
    dt = 0.01
    steps = int(round(duration / dt))

    p0 = fk(q0)[:3, 3]
    target = fk(q_sing)[:3, 3]
    zetas = np.zeros((steps, 6))
    zetas[:, :3] = (target - p0) / duration

    direct, direct_rates = follow(q0, zetas, dt, method="inverse")
    damped, damped_rates = follow(q0, zetas, dt, method="dls", lam=1e-2)
    time = np.arange(steps) * dt

    print(f"direct maximum |joint rate|: {np.max(np.abs(direct_rates)):.6e} rad/s")
    print(f"DLS maximum |joint rate|:    {np.max(np.abs(damped_rates)):.6e} rad/s")

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for j in range(6):
        axes[0].semilogy(time, np.abs(direct_rates[:, j]), label=f"joint_{j}")
        axes[1].plot(time, damped_rates[:, j], label=f"joint_{j}")
    axes[0].set_title("Direct inverse: near-singular velocity growth")
    axes[0].set_ylabel("|joint velocity| (rad/s)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_title(r"Damped least squares ($\lambda=0.01$)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("joint velocity (rad/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3)
    figure.tight_layout()
    figure.savefig("figures/damping-comparison.png", dpi=150)
    print("saved plot: figures/damping-comparison.png")


if __name__ == "__main__":
    main()
