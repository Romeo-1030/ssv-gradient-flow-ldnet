#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("outputs/ssv_gamma_frozen_ldnet_v5_final3000_cosine")
OUT = Path("figures/final3000_training_diagnostics.png")
METHODS = {
    "physical": ("Physical LDNet", "0.45"),
    "learned_ssv": ("Learned SSV", "#e97842"),
    "learned_ssv_energy": ("Learned SSV + energy", "#a73516"),
    "oracle_ssv": ("Oracle SSV", "#3975a8"),
    "oracle_ssv_energy": ("Oracle SSV + energy", "#134b73"),
}


def histories(method):
    return [json.loads((ROOT / method / "n32" / f"seed{s}" / "history.json").read_text())
            for s in range(5)]


def plot_band(ax, hs, key, color, label):
    epochs = np.asarray([row["epoch"] for row in hs[0]])
    values = np.asarray([[row[key] for row in h] for h in hs], dtype=float)
    mean, std = values.mean(0), values.std(0)
    ax.plot(epochs, mean, color=color, lw=1.8, label=label)
    ax.fill_between(epochs, np.maximum(mean - std, 1e-12), mean + std,
                    color=color, alpha=.14, linewidth=0)


fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))
for method, (label, color) in METHODS.items():
    hs = histories(method)
    plot_band(axes[0], hs, "data_loss", color, label)
    plot_band(axes[1], hs, "valid", color, label)
    if "energy" in method:
        plot_band(axes[2], hs, "raw_gf", color, label)

for ax, title, ylabel in zip(
        axes,
        ["Supervised data loss", "Three-frame validation", "Raw GF residual"],
        ["loss", "NError$_{full}$", "GF loss"]):
    ax.set_title(title)
    ax.set_xlabel("optimization step")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(alpha=.25)
axes[0].legend(fontsize=7, frameon=False)
axes[2].legend(fontsize=8, frameon=False)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print(OUT)
