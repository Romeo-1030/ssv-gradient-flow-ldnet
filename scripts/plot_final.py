#!/usr/bin/env python3
"""Final two-panel figure: interpolation RelRMSE and fixed-initial-scale extrapolation error."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("outputs")
DATA = np.load("data/single_gaussian_heat.npz")
t, x, u = DATA["t"], DATA["x"], DATA["u"]
test = np.random.default_rng(123).permutation(len(u))[128:160]
fair = np.abs(x[None, :]) <= 4.5 * np.sqrt(1 + t[:, None])
initial_rms = np.sqrt(((u[test, 0] ** 2) * fair[0]).sum() / (fair[0].sum() * len(test)))

styles = {
    "physical": ("Physical LDNet", "#8c8c8c", "--"),
    "physical_energy": ("Physical LDNet + energy", "#4d4d4d", "-"),
    "learned_ssv": ("Learned SSV LDNet", "#e17c4f", "--"),
    "learned_ssv_energy": ("Learned SSV LDNet + energy", "#a63d20", "-"),
}
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), dpi=220)
pre, post = t <= .5 + 1e-8, t >= .5 - 1e-8
for mode, (label, color, ls) in styles.items():
    dirs = sorted((ROOT / mode / "n32").glob("seed*"))
    rel = 100 * np.stack([np.load(d / "relrmse_vs_time.npy") for d in dirs])
    fixed = 100 * np.stack([np.load(d / "rmse_vs_time.npy") for d in dirs]) / initial_rms
    for ax, values, mask in ((axes[0], rel, pre), (axes[1], fixed, post)):
        mean, std = values.mean(0), values.std(0)
        ax.plot(t[mask], mean[mask], color=color, ls=ls, lw=2.2, label=label)
        ax.fill_between(t[mask], np.maximum(mean[mask]-std[mask], 1e-4), mean[mask]+std[mask], color=color, alpha=.12)

axes[0].set(title="Observed window", xlabel="time", ylabel="RelRMSE (%)", yscale="log", xlim=(0,.5))
axes[1].set(title="Long-time extrapolation", xlabel="time", ylabel=r"RMSE / RMS$(u_0)$ (%)", yscale="log", xlim=(.5,10))
for ax in axes: ax.grid(alpha=.22, which="both")
axes[1].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Heat equation: 32 training trajectories, 3 observed frames, mean $\pm$ std over 5 seeds", fontsize=12)
fig.tight_layout(); out=ROOT/"final_two_panel_log.png"; fig.savefig(out,bbox_inches="tight"); print(out)
