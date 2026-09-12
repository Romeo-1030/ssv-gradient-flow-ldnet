"""CPU inference for the supplied final learned-SSV + energy seed-0 model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from train import ShiftedSSVLDNet

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"data/single_gaussian_heat_fixed_domain_v1.npz")
    parser.add_argument("--out_dir", type=Path, default=ROOT/"outputs/inference")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = ROOT/"artifacts/pretrained/learned_ssv_energy_seed0.pt"
    metadata = json.loads((ROOT/"artifacts/pretrained/inference.json").read_text())
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Checkpoint checksum mismatch.")
    if not args.data.exists():
        raise FileNotFoundError("First run: python scripts/generate_data.py --out data/single_gaussian_heat_fixed_domain_v1.npz")
    with np.load(args.data) as data:
        x, t, u, params = (data[k] for k in ("x", "t", "u", "params"))
    if u.shape != (160, 201, 256) or params.shape != (160, 3):
        raise ValueError("Use the released generator with its default settings.")
    np.testing.assert_allclose(x, np.linspace(-16, 16, 256), atol=1e-6)
    np.testing.assert_allclose(t, np.linspace(0, 10, 201), atol=1e-6)
    model = ShiftedSSVLDNet(**metadata["model"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    test_ids = np.random.default_rng(123).permutation(len(u))[128:160]
    with torch.inference_mode():
        prediction, _, _ = model.rollout(
            torch.tensor(params[test_ids]), torch.tensor(t),
            torch.tensor(x)[None].expand(len(test_ids), -1))
    pred = prediction.numpy()
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite predictions.")
    # Independently evaluate the standard full-domain metric in float64.
    truth = u[test_ids].astype(np.float64)
    curve = np.sqrt(((pred.astype(np.float64)-truth)**2).sum(axis=(0,2)) /
                    (truth**2).sum(axis=(0,2)))
    reference = json.loads((ROOT/"results/summary/aggregate.json").read_text())
    run = next(v for v in reference["methods"]["learned_ssv_energy"]["runs"] if v["seed"] == 0)
    np.testing.assert_allclose(curve, run["curve"], rtol=1e-4, atol=1e-6)
    observed = t <= .5+1e-7
    report = {
        "method": "learned_ssv_energy", "seed": 0, "device": "cpu",
        "test_trajectories": len(test_ids), "prediction_shape": list(pred.shape),
        "observed_percent": float(100*curve[observed].mean()),
        "extrapolation_percent": float(100*curve[~observed].mean()),
        "final_percent": float(100*curve[-1]),
        "max_curve_absolute_difference": float(np.max(abs(curve-np.array(run["curve"])))),
        "reference_check": "passed", "training_performed": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir/"predictions.npz", prediction=pred,
                        truth=u[test_ids], x=x, t=t, test_ids=test_ids,
                        params=params[test_ids], full_relrmse=curve)
    (args.out_dir/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.8))
    for ax, time in zip(axes, [0, .5, 2, 5, 10]):
        j = int(np.argmin(abs(t-time)))
        ax.plot(x, u[test_ids[0], j], "k--", label="Reference")
        ax.plot(x, pred[0, j], label="Prediction")
        ax.set(title=f"t={time:g}", xlabel="x", xlim=(-12, 12), ylim=(-.05, 1.3))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("u(x,t)"); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out_dir/"prediction.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
