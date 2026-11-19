"""Read-only result aggregation and scientific figures for per-IC gamma runs.

The runner owns all training, calibration and source-of-truth records. This
module writes only reports/figures and accepts incomplete runs while a queue
is active. Examples are prespecified test positions 0, 1 and 2, never selected
by the size of their error. All images show an explicitly named optimizer seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


METHODS = ("latent", "latent_gf", "ssv_gf")
LABELS = {
    "latent": "LDNet",
    "latent_gf": "LDNet + GF",
    "ssv_gf": "Per-IC SSV + LDNet + GF",
}
COLORS = {"latent": "#c86440", "latent_gf": "#3579b1", "ssv_gf": "#25866b"}
OBSERVED = np.asarray([0., .25, .5, .75, 1., 1.25])


def _read(path, default):
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, PermissionError):
        # A report is allowed to lag an atomic result update while running.
        return default


def _rows(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("records", "results", "selections", "rows"):
            if isinstance(value.get(key), list):
                return value[key]
    return []


def _metric(row):
    return row.get("metrics", row.get("test", {}))


def _method(row):
    return row.get("method", row.get("arm"))


def _stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {
        "count": int(len(a)),
        "mean": float(a.mean()) if len(a) else None,
        "sample_sd": float(a.std(ddof=1)) if len(a) > 1 else None,
        "values": a.tolist(),
    }


def _format_percent(stats):
    if not stats["count"]:
        return "pending"
    mean = 100 * stats["mean"]
    if stats["sample_sd"] is None:
        return f"{mean:.3f} (one seed)"
    return f"{mean:.3f} +/- {100 * stats['sample_sd']:.3f}"


def _load_data(out):
    path = out / "data.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _prediction(out, row, data):
    name = row.get("prediction", row.get("prediction_file"))
    if not isinstance(name, str):
        name = f"{_method(row)}_{row['seed']}_test.npz"
    path = Path(name)
    if not path.is_absolute():
        path = out / path
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    pred = arrays.get("pred", arrays.get("prediction"))
    if pred is None or np.ndim(pred) != 3:
        return None
    truth = arrays.get("truth", arrays.get("u"))
    if truth is None and "u" in data and "split" in data:
        truth = data["u"][np.asarray(data["split"]).astype(str) == "test"]
    x, t = arrays.get("x", data.get("x")), arrays.get("t", data.get("t"))
    if truth is None or x is None or t is None or pred.shape != truth.shape:
        return None
    if pred.shape[1:] != (len(t), len(x)):
        return None
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        return None
    return {**arrays, "pred": pred, "truth": truth, "x": x, "t": t}


def _save(fig, path, outputs, pdf=False):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    outputs.append(path)
    if pdf:
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        outputs.append(pdf_path)


def _plot_curves(out, records, data, plt, outputs):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), layout="constrained")
    plotted = False
    for method in METHODS:
        rows = [r for r in records if _method(r) == method and "curve" in _metric(r)]
        if not rows:
            continue
        curves = np.asarray([_metric(r)["curve"] for r in rows], dtype=float) * 100
        if curves.ndim != 2:
            continue
        t = data.get("t", np.linspace(0., 6., curves.shape[1]))
        if len(t) != curves.shape[1]:
            continue
        mean = curves.mean(0)
        sd = curves.std(0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
        label = f"{LABELS[method]} (n={len(rows)})"
        for index, ax in enumerate(axes):
            mask = np.ones(len(t), dtype=bool) if index == 0 else (t <= 1.25)
            ax.plot(t[mask], mean[mask], color=COLORS[method], label=label, lw=1.8)
            if len(curves) > 1:
                ax.fill_between(t[mask], np.maximum(mean[mask] - sd[mask], 1e-6), mean[mask] + sd[mask],
                                color=COLORS[method], alpha=.13)
        plotted = True
    if plotted:
        axes[0].axvspan(0, 1.25, color="#808080", alpha=.08)
        axes[0].axvline(1.25, color="gray", ls=":", lw=1)
        axes[0].axhline(10, color="gray", ls="--", lw=.8, label="10% reference")
        axes[0].set(xlabel="Physical time t", ylabel="Population relative L2 error (%)",
                    yscale="log", title="Test forecasts: mean and sample SD")
        axes[1].set(xlabel="Physical time t", ylabel="Population relative L2 error (%)",
                    xlim=(0, 1.25), title="Observed window (six labeled frames)")
        for time in OBSERVED:
            axes[1].axvline(time, color="gray", alpha=.18, lw=.6)
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        _save(fig, out / "figures" / "error_curves.png", outputs, pdf=True)
    plt.close(fig)


def _plot_examples(out, records, data, plt, outputs):
    chosen = {}
    for method in METHODS:
        for row in sorted([r for r in records if _method(r) == method], key=lambda r: r["seed"]):
            arrays = _prediction(out, row, data)
            if arrays is not None:
                chosen[method] = (row, arrays)
                break
    if not chosen:
        return
    available = min(a["pred"].shape[0] for _, a in chosen.values())
    for example in range(min(3, available)):
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.1), layout="constrained")
        reference = next(iter(chosen.values()))[1]
        for ax, time in zip(axes.flat, [0., .5, 1.25, 2., 4., 6.]):
            j = int(np.abs(reference["t"] - time).argmin())
            ax.plot(reference["x"], reference["truth"][example, j], color="black", lw=2,
                    label="Heat-equation reference")
            for method, (row, arrays) in chosen.items():
                jj = int(np.abs(arrays["t"] - time).argmin())
                ax.plot(arrays["x"], arrays["pred"][example, jj], color=COLORS[method],
                        lw=1.4, ls="--", label=f"{LABELS[method]}, seed {row['seed']}")
            ax.set(title=f"t = {reference['t'][j]:g}", xlabel="x", ylabel="u(x,t)")
            ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", fontsize=8, ncol=2)
        fig.suptitle(f"Prespecified test trajectory {example}: spatial profiles", fontsize=13)
        _save(fig, out / "figures" / f"profiles_test_{example:02d}.png", outputs)
        plt.close(fig)

        fig, axes = plt.subplots(len(chosen), 3, figsize=(12, 3.15 * len(chosen)),
                                 squeeze=False, layout="constrained")
        for row_axes, (method, (record, arrays)) in zip(axes, chosen.items()):
            truth, pred = arrays["truth"][example], arrays["pred"][example]
            vmin, vmax = min(float(truth.min()), float(pred.min())), max(float(truth.max()), float(pred.max()))
            extent = [float(arrays["x"][0]), float(arrays["x"][-1]),
                      float(arrays["t"][0]), float(arrays["t"][-1])]
            for ax, field, title in zip(row_axes, [truth, pred, np.abs(pred - truth)],
                                        ["Reference", "Prediction", "Absolute error"]):
                kwargs = {"cmap": "magma", "vmin": 0} if title == "Absolute error" else {
                    "cmap": "viridis", "vmin": vmin, "vmax": vmax}
                im = ax.imshow(field, origin="lower", aspect="auto", extent=extent, **kwargs)
                ax.axhline(1.25, color="white", ls=":", lw=.9)
                ax.set(title=title, xlabel="x", ylabel="t")
                fig.colorbar(im, ax=ax, shrink=.85)
            row_axes[0].set_ylabel(f"{LABELS[method]}\nseed {record['seed']}\nt")
        fig.suptitle(f"Prespecified test trajectory {example}; reference/prediction share scale within each row",
                     fontsize=11)
        _save(fig, out / "figures" / f"heatmaps_test_{example:02d}.png", outputs)
        plt.close(fig)


def _selection_rows(out, data):
    rows = []
    for path in (out / "gamma_selection.json", out / "calibration.json",
                 out / "selector" / "calibration.json", out / "calibration" / "calibration.json"):
        rows = _rows(_read(path, []))
        if rows:
            break
    references = {}
    splits = {}
    if "gamma_reference" in data:
        ids = data.get("sample_id", np.arange(len(data["gamma_reference"])))
        references = {str(i): float(g) for i, g in zip(ids, data["gamma_reference"])}
        if "split" in data:
            splits = {str(i): str(s) for i, s in zip(ids, data["split"])}
    complete = []
    for row in rows:
        gamma = row.get("gamma_selected", row.get("selected_gamma", row.get("gamma")))
        if gamma is None or not np.isfinite(gamma) or gamma <= 0:
            continue
        row = dict(row)
        row["gamma_selected"] = float(gamma)
        if "gamma_reference" not in row:
            row["gamma_reference"] = references.get(str(row.get("sample_id")))
        if "split" not in row:
            row["split"] = splits.get(str(row.get("sample_id")), "unspecified")
        complete.append(row)
    return complete


def _plot_gamma(out, selections, plt, outputs):
    if not selections:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), layout="constrained")
    gamma = np.array([r["gamma_selected"] for r in selections])
    reference = np.array([r.get("gamma_reference") or np.nan for r in selections], dtype=float)
    split = np.array([r.get("split", "unspecified") for r in selections])
    good = np.isfinite(reference) & (reference > 0)
    for name, color in [("train", "#3579b1"), ("valid", "#c86440"), ("test", "#25866b"),
                        ("unspecified", "#777777")]:
        mask = (split == name) & good
        if mask.any():
            axes[0].scatter(reference[mask], gamma[mask], c=color, s=25, alpha=.75, label=name)
    if good.any():
        low, high = min(reference[good].min(), gamma.min()), max(reference[good].max(), gamma.max())
        axes[0].plot([low, high], [low, high], "k:", lw=1, label="Equality (not an optimum guarantee)")
    axes[0].set(xscale="log", yscale="log", xlabel="Generative variance-based gamma reference",
                ylabel="Six-frame selected gamma", title="Individual assignments")
    if good.any():
        axes[0].legend(fontsize=7)
    values = np.r_[gamma, reference[good]]
    low, high = values.min(), values.max()
    if low == high:
        low, high = low / 1.5, high * 1.5
    bins = np.geomspace(low * .95, high * 1.05, 14)
    axes[1].hist(gamma, bins=bins, alpha=.55, label="Selected gamma", color="#25866b")
    if good.any():
        axes[1].hist(reference[good], bins=bins, histtype="step", linewidth=1.8,
                     label="Reference gamma", color="black")
    axes[1].set(xscale="log", xlabel="gamma", ylabel="Trajectory count", title="Calibration coverage")
    axes[1].legend(fontsize=8)
    _save(fig, out / "figures" / "gamma_assignments.png", outputs)
    plt.close(fig)

    chosen = selections[:min(6, len(selections))]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7), layout="constrained")
    plotted = False
    for ax, row in zip(axes.flat, chosen):
        candidates = row.get("candidates", row.get("dictionary"))
        scores = row.get("scores", row.get("candidate_scores"))
        if candidates is None or scores is None:
            ax.set_visible(False)
            continue
        if len(scores) and isinstance(scores[0], dict):
            scores = [r.get("mean", r.get("score", np.nan)) for r in scores]
        x, y = np.asarray(candidates, dtype=float), np.asarray(scores, dtype=float)
        if x.ndim != 1 or y.shape != x.shape:
            ax.set_visible(False)
            continue
        order = np.argsort(x)
        ax.plot(x[order], 100 * y[order], marker="o", ms=3, lw=1, color="#3579b1")
        ax.axvline(row["gamma_selected"], color="#25866b", ls="--", label="Selected")
        ax.set(xscale="log", xlabel="Candidate gamma", ylabel="Weighted calibration error (%)",
               title=f"{row.get('split', '')} sample {row.get('sample_id', '?')}")
        ax.grid(alpha=.2)
        plotted = True
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_visible(False)
    if plotted:
        fig.suptitle("First six calibration records; ordinary-network temporal validation", fontsize=12)
        _save(fig, out / "figures" / "gamma_selection_scores.png", outputs)
    plt.close(fig)


def _plot_energy(out, plt, outputs):
    for path in sorted(out.glob("energy_*.npz")):
        with np.load(path, allow_pickle=False) as grid:
            if not all(key in grid for key in ("wgrid", "wxgrid", "e")):
                continue
            w, wx, energy = [np.asarray(grid[key]) for key in ("wgrid", "wxgrid", "e")]
        if w.ndim != 1 or wx.ndim != 1 or energy.shape != (len(w), len(wx)):
            continue
        offset = energy[np.abs(w).argmin(), np.abs(wx).argmin()]
        fig, ax = plt.subplots(figsize=(6.5, 5), layout="constrained")
        im = ax.contourf(w, wx, (energy - offset).T, levels=30, cmap="coolwarm")
        fig.colorbar(im, ax=ax, label="Energy density minus value nearest (0,0)")
        ax.set(xlabel="U", ylabel="U_xi", title=path.stem.replace("_", " ") + "\nLearned regularizer; no convexity guarantee")
        _save(fig, out / "figures" / (path.stem + ".png"), outputs)
        plt.close(fig)


def _plot_ordinary(out, data, plt, outputs):
    path = out / "ordinary_selected.npz"
    if not path.exists():
        return
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    pred = arrays.get("mean", arrays.get("pred", arrays.get("prediction")))
    x, t = arrays.get("x", data.get("x")), arrays.get("t", data.get("t"))
    if pred is None or pred.ndim != 3 or x is None or t is None:
        return
    sample_ids = arrays.get("sample_ids", arrays.get("sample_id", np.arange(len(pred))))
    ids = data.get("sample_id", np.arange(len(data.get("u", []))))
    positions = {str(v): i for i, v in enumerate(ids)}
    truth = arrays.get("truth")
    if truth is None and "u" in data and "t" in data:
        if not all(str(v) in positions for v in sample_ids):
            return
        observed_ids = [int(np.abs(data["t"] - value).argmin()) for value in t]
        truth = data["u"][[positions[str(v)] for v in sample_ids]][:, observed_ids]
    if truth is None or truth.shape != pred.shape or pred.shape[1:] != (len(t), len(x)):
        return
    test_ids = set(str(i) for i, s in zip(ids, data.get("split", [])) if str(s) == "test")
    examples = [i for i, sample_id in enumerate(sample_ids) if str(sample_id) in test_ids][:3]
    if not examples:
        examples = list(range(min(3, len(pred))))
    fig, axes = plt.subplots(len(examples), 3, figsize=(11.8, 2.7 * len(examples)),
                             squeeze=False, layout="constrained")
    by_seed = arrays.get("by_seed")
    for row_axes, example in zip(axes, examples):
        for ax, time in zip(row_axes, [0., .5, 1.25]):
            j = int(np.abs(t - time).argmin())
            ax.plot(x, truth[example, j], color="black", lw=1.8, label="Reference")
            ax.plot(x, pred[example, j], color="#25866b", lw=1.5, ls="--",
                    label="Selected ordinary-network mean")
            if by_seed is not None and by_seed.ndim == 4 and by_seed.shape[1:] == pred.shape and len(by_seed) > 1:
                sd = by_seed[:, example, j].std(0, ddof=1)
                ax.fill_between(x, pred[example, j] - sd, pred[example, j] + sd,
                                alpha=.15, color="#25866b")
            ax.set(title=f"Sample {sample_ids[example]}, t={t[j]:g}", xlabel="x", ylabel="u(x,t)")
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", fontsize=8, ncol=2)
    fig.suptitle("Auxiliary ordinary functions: fitted frames (0, 0.5), held-out frame (1.25)", fontsize=12)
    _save(fig, out / "figures" / "ordinary_functions.png", outputs)
    plt.close(fig)


def build_report(out: Path) -> list[Path]:
    """Create reviewable reports and figures from all currently completed fits."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    config = _read(out / "config.json", {})
    records = [r for r in _rows(_read(out / "results.json", []))
               if _method(r) in METHODS and "seed" in r and _metric(r)]
    data = _load_data(out)
    selections = _selection_rows(out, data)
    planned = int(config.get("seeds", 4))
    summary = {"completed_runs": len(records), "planned_seeds_per_method": planned,
               "calibrated_trajectories": len(selections), "methods": {}, "paired_long_differences": {}}
    for method in METHODS:
        rows = [r for r in records if _method(r) == method]
        summary["methods"][method] = {
            "seeds": sorted(int(r["seed"]) for r in rows),
            "count": len(rows),
            **{key: _stats([_metric(r)[key] for r in rows if key in _metric(r)])
               for key in ("observed", "label_times_error", "short", "long", "final")},
            "seconds": _stats([r["seconds"] for r in rows if "seconds" in r]),
        }
    lookup = {(_method(r), int(r["seed"])): r for r in records}
    for comparator in ("latent", "latent_gf"):
        seeds = sorted(set(summary["methods"]["ssv_gf"]["seeds"])
                       & set(summary["methods"][comparator]["seeds"]))
        diffs = [_metric(lookup["ssv_gf", s])["long"] - _metric(lookup[comparator, s])["long"]
                 for s in seeds if "long" in _metric(lookup["ssv_gf", s])
                 and "long" in _metric(lookup[comparator, s])]
        summary["paired_long_differences"][comparator] = {
            "seeds": seeds, **_stats(diffs), "ssv_gf_wins": int(np.sum(np.array(diffs) < 0))}
    complete = all(summary["methods"][m]["count"] >= planned for m in METHODS)
    summary["complete"] = complete
    outputs = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    _plot_curves(out, records, data, plt, outputs)
    _plot_examples(out, records, data, plt, outputs)
    _plot_gamma(out, selections, plt, outputs)
    _plot_ordinary(out, data, plt, outputs)
    _plot_energy(out, plt, outputs)

    status = "COMPLETE" if complete else "INCOMPLETE: only completed fits are summarized"
    lines = ["# Per-initial-condition gamma experiment", "", status, "",
             f"Completed final fits: {len(records)} / {len(METHODS) * planned}. "
             f"Calibrated trajectories: {len(selections)}. Planned optimizer seeds per arm: {planned}.", "",
             "Each initial condition is calibrated with an ordinary network, without latent dynamics. "
             "The first three frames fit candidate networks; the next three score their forecasts with weights 1:2:3. "
             "Selected per-trajectory gamma values are frozen before freshly initialized shared LDNet training. "
             "The auxiliary ordinary-network weights are not transferred to LDNet.", "",
             "All final arms are supervised at t = 0, 0.25, 0.5, 0.75, 1, 1.25. "
             "The SSV arm fixes alpha = beta = 1/2. GF uses model-generated rollout states, not future reference fields. "
             "No reference fields after t = 1.25 enter calibration or training.", "",
             "The SSV method uses each unseen trajectory's early observations to calibrate gamma. "
             "The two original-style baselines condition predictions on the initial field alone. "
             "This is a comparison of these complete workflows, including calibration information; "
             "it does not isolate the effect of coordinates under identical inference inputs.", "",
             "Errors below are percentages: population relative L2 at each time, averaged over the indicated window. "
             "Observed = [0,1.25] (11 grid times, of which six are supervised); "
             "short = (1.25,2]; long = (2,6]; final = t=6. "
             "Uncertainty is sample SD across optimizer seeds on the same dataset, not across independent dataset draws.", "",
             "| Method | Completed seeds | Observed % | Short % | Long % | Final % |",
             "|---|---|---:|---:|---:|---:|"]
    for method in METHODS:
        stats = summary["methods"][method]
        lines.append("| " + " | ".join([LABELS[method], str(stats["seeds"]),
                     *[_format_percent(stats[key]) for key in ("observed", "short", "long", "final")]]) + " |")
    if records:
        lines.extend(["", "| Paired long-error difference: SSV + GF minus comparator | Seeds | Difference (percentage points) | SSV + GF wins |",
                      "|---|---|---:|---:|"])
        for comparator, values in summary["paired_long_differences"].items():
            lines.append(f"| {LABELS[comparator]} | {values['seeds']} | {_format_percent(values)} | "
                         f"{values['ssv_gf_wins']} / {values['count']} |")
    target = summary["methods"]["ssv_gf"]["long"]
    if target["count"]:
        lines.extend(["", f"Current SSV + GF mean long-window error is {100 * target['mean']:.3f}%. "
                      + ("It is at or below the 10% reference target." if target["mean"] <= .1
                         else "It is above the 10% reference target.")
                      + (" The comparison is still incomplete." if not complete else "")])
    lines.extend(["", "The generative variance-based gamma is an audit reference, not a guarantee of the "
                  "forecast-loss optimum. A selected gamma can legitimately differ from it. "
                  "A learned local energy is a soft regularizer, not a demonstrated energy-dissipation or stability theorem.", "",
                  "Figures use prespecified test positions 0, 1 and 2 and the smallest completed optimizer seed "
                  "for each method. Profiles and heatmaps show those individual predictions, not a favorable example "
                  "selected after observing errors. Error-curve bands show sample SD across available optimizer seeds. "
                  "If exported, ordinary-function plots use the first three test trajectories and show the mean "
                  "and sample SD across calibration seeds at their common selected gamma.", "",
                  "## Generated figures", ""])
    for path in outputs:
        relative = path.relative_to(out).as_posix()
        lines.append(f"- [{relative}]({relative})")
    lines.extend(["", "Config, selection scores, per-seed metrics, checkpoints and prediction arrays remain in this run directory."])
    for filename in ("REPORT.md", "REPORT.txt"):
        path = out / filename
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        outputs.append(path)
    path = out / "report_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    outputs.append(path)
    return outputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    for generated in build_report(args.out):
        print(generated)
