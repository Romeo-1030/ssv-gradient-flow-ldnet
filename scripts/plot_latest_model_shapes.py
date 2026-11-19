#!/usr/bin/env python3
"""Physical and learned-SSV shape evolution for the latest frozen-SSV model."""
import argparse
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_fourier_longtime_gf import ShiftedSSVLDNet  # noqa: E402

DATA = ROOT / "data/single_gaussian_heat_fixed_domain_v1.npz"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path,
                    default=ROOT / "artifacts/pretrained/learned_ssv_energy_seed0.pt")
parser.add_argument("--out", type=Path,
                    default=ROOT / "outputs/figures/final3000_solution_shapes")
args = parser.parse_args()
CKPT = args.checkpoint
OUT = args.out
OUT.parent.mkdir(parents=True, exist_ok=True)

d = np.load(DATA)
x_np, t_np, u_np, params_np = d["x"], d["t"], d["u"], d["params"]
test_ids = np.random.default_rng(123).permutation(len(u_np))[128:160]
sample_id = int(test_ids[0])  # deterministic; no post-hoc trajectory selection

device = "cpu"
model = ShiftedSSVLDNet(
    nx=len(x_np), mode="learned_ssv_energy", rep_dim=24, latent=16,
    width=128, modes=33, physical_extent=float(np.abs(x_np).max()),
    ssv_extent=4.5, use_gamma=True, oracle_gamma=False,
).to(device)
state = torch.load(CKPT, map_location=device, weights_only=True)
model.load_state_dict(state["model"])
model.eval()

t = torch.tensor(t_np, device=device)
x = torch.tensor(x_np, device=device)
p = torch.tensor(params_np[sample_id:sample_id + 1], device=device)
with torch.no_grad():
    prediction, latent, _ = model.rollout(p, t, x[None])
    alpha, beta = model.scales()
    gamma = model.gamma(p)[0]

target_times = np.array([0.0, 0.5, 2.0, 5.0, 10.0])
time_ids = [int(np.argmin(np.abs(t_np - value))) for value in target_times]
xi_np = np.linspace(-3.25, 3.25, 256, dtype=np.float32)
xi = torch.tensor(xi_np, device=device)[None]

fig, axes = plt.subplots(2, len(time_ids), figsize=(13.8, 5.2), dpi=220)
amp, center, width = params_np[sample_id]
nu = float(d["diffusivity"])
for col, j in enumerate(time_ids):
    tj = float(t_np[j])
    truth = u_np[sample_id, j]
    pred = prediction[0, j].cpu().numpy()
    rel = np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), 1e-12)

    ax = axes[0, col]
    ax.plot(x_np, truth, "k--", lw=1.6, label="truth")
    ax.plot(x_np, pred, color="#9b2f2f", lw=1.8, label="prediction")
    ax.set_title(rf"$t={tj:g}$")
    ax.text(0.04, 0.91, rf"relRMSE $={100*rel:.2f}\%$", transform=ax.transAxes,
            fontsize=8, va="top")
    ax.set_xlim(-12, 12)
    ax.grid(alpha=0.18)

    scale = 1.0 + float(gamma) * tj
    x_query = xi_np * scale ** float(alpha)
    variance = width ** 2 + 2.0 * nu * tj
    physical_truth = amp * width / np.sqrt(variance) * np.exp(
        -0.5 * (x_query - center) ** 2 / variance)
    transformed_truth = scale ** float(beta) * physical_truth
    with torch.no_grad():
        transformed_pred = model.decoder(latent[:, j], xi, p)[0].cpu().numpy()
    ax = axes[1, col]
    ax.plot(xi_np, transformed_truth, "k--", lw=1.6)
    ax.plot(xi_np, transformed_pred, color="#235789", lw=1.8)
    ax.set_xlim(xi_np[0], xi_np[-1])
    ax.grid(alpha=0.18)

# Keep limits fixed across time; do not let autoscaling hide amplitude decay.
for row in range(2):
    values = np.concatenate([line.get_ydata() for ax in axes[row] for line in ax.get_lines()])
    ymin, ymax = float(values.min()), float(values.max())
    padding = 0.05 * max(ymax - ymin, 1e-6)
    for ax in axes[row]:
        ax.set_ylim(ymin - padding, ymax + padding)

axes[0, 0].set_ylabel(r"physical field $u(x,t)$")
axes[1, 0].set_ylabel(r"transformed field $U(\xi,\tau)$")
for ax in axes[0]: ax.set_xlabel(r"$x$")
for ax in axes[1]: ax.set_xlabel(r"$\xi$")
axes[0, -1].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
print(OUT.with_suffix(".png"))
