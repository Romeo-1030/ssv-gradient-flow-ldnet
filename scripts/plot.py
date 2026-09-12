"""Plot published compact curves without data or model checkpoints."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--summary', type=Path, default=Path('results/summary/aggregate.json'))
p.add_argument('--out', type=Path, default=Path('figures/fixed_domain_two_panel_log'))
a = p.parse_args(); d = json.loads(a.summary.read_text()); t = np.array(d['time'])
fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.25), dpi=220)
for name, result in d['methods'].items():
    values = np.array([s['curve'] for s in result['runs']])*100
    mean, std = values.mean(0), values.std(0)
    for ax, mask in zip(axes, (t<=.5+1e-7, t>=.5-1e-7)):
        line, = ax.plot(t[mask], mean[mask], label=name)
        ax.fill_between(t[mask], np.maximum(mean[mask]-std[mask], 1e-4), mean[mask]+std[mask], color=line.get_color(), alpha=.12)
for ax, title in zip(axes, ('Observed window', 'Long-time extrapolation')):
    ax.set(title=title, xlabel='time', ylabel='Full-domain RelRMSE (%)', yscale='log'); ax.grid(alpha=.2)
axes[1].legend(fontsize=7); fig.tight_layout(); a.out.parent.mkdir(parents=True, exist_ok=True)
for suffix in ('.png', '.pdf'): fig.savefig(a.out.with_suffix(suffix), bbox_inches='tight')
