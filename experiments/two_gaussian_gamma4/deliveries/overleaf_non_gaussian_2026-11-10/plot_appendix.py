"""Render appendix figures from saved predictions only; no training/inference."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'experiments/non_gaussian_heat/runs/ordinary_full_comparison_domain16_t10_cuda'
OUT = HERE / 'project/figures'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'axes.linewidth': .65, 'lines.linewidth': 1.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 250,
})

def finish(ax):
    ax.grid(True, color='#d8d8d8', alpha=.55, linewidth=.5)
    ax.set_axisbelow(True)
    ax.tick_params(width=.65, length=3)

def save(fig, stem):
    fig.savefig(OUT / f'{stem}.pdf')
    fig.savefig(OUT / f'{stem}.png')
    plt.close(fig)

summary = json.loads((RUN / 'report_summary.json').read_text())
seeds = list(map(str, summary['paired_seeds']))
assert seeds == ['0', '1', '2', '3']
times = np.array(summary['curves']['t'])
styles = {
    'latent': ('LDNet', '#cc6677', '-'),
    'latent_gf': ('LDNet + GF', '#4477aa', '--'),
    'ssv': ('LDNet + shared SSV', '#aa3377', '-'),
    'ssv_gf': ('LDNet + shared SSV + GF', '#228833', '--'),
}

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
fig.subplots_adjust(left=.085, right=.973, bottom=.17, top=.91, wspace=.30)
for ax, methods in zip(axes, [list(styles), ['ssv', 'ssv_gf']]):
    ax.axvspan(0, 1.25, color='#eeeeee', alpha=.75, zorder=0)
    ax.axvline(1.25, color='#888888', linestyle=':', linewidth=.9)
    for method in methods:
        values = 100*np.array([summary['curves'][method][s] for s in seeds])
        mean, std = values.mean(0), values.std(0, ddof=0)
        label, color, linestyle = styles[method]
        ax.fill_between(times, mean-std, mean+std, color=color, alpha=.14, linewidth=0)
        ax.plot(times, mean, label=label, color=color, linestyle=linestyle)
    ax.set_xlim(0, 10)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_xlabel('Physical time $t$')
    ax.set_ylabel('RelRMSE (%)')
    finish(ax)
axes[0].set_yscale('log')
axes[0].set_ylim(3, 800)
axes[0].yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
axes[0].yaxis.set_minor_formatter(NullFormatter())
axes[0].set_title('(a) Four-method comparison')
axes[0].legend(loc='upper left', bbox_to_anchor=(.015, .98), frameon=True,
               framealpha=.93, edgecolor='#dddddd', handlelength=2.3,
               borderpad=.35, labelspacing=.35)
axes[1].set_ylim(3, 11)
axes[1].set_yticks([3, 5, 7, 9, 11])
axes[1].set_title('(b) Additional effect of GF with SSV')
axes[1].legend(loc='upper right', frameon=True, framealpha=.93,
               edgecolor='#dddddd', handlelength=2.3,
               borderpad=.35, labelspacing=.35)
save(fig, 'non_gaussian_error_curves')

with np.load(RUN / 'ssv_gf_0_test.npz') as data:
    sample = int(np.flatnonzero(data['sample_ids'] == 'test_001')[0])
    x = data['x'].copy()
    time_grid = data['t'].copy()
    truth = data['truth'][sample].copy()
    pred = data['pred'][sample].copy()
    assert float(data['gamma'][sample]) == 4.

fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.45))
fig.subplots_adjust(left=.085, right=.973, bottom=.13, top=.87, hspace=.52, wspace=.35)
for ax, time in zip(axes.flat, [0., .5, 1.25, 2., 6., 10.]):
    k = int(np.flatnonzero(np.isclose(time_grid, time))[0])
    ax.plot(x, truth[k], color='#222222', linewidth=1.45, label='Exact solution')
    ax.plot(x, pred[k], color='#228833', linestyle='--', linewidth=1.4,
            label='LDNet + shared SSV + GF')
    ax.set_xlim(-16, 16)
    ax.set_xticks([-16, 0, 16])
    low = min(float(truth[k].min()), float(pred[k].min()))
    high = max(float(truth[k].max()), float(pred[k].max()))
    span = high-low
    ax.set_ylim(min(-.04*span, low-.03*span), high+.10*span)
    ax.set_title(f'$t={time:g}$', pad=4)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    finish(ax)
for ax in axes[1]:
    ax.set_xlabel('Physical coordinate $x$')
for ax in axes[:, 0]:
    ax.set_ylabel('$u(x,t)$')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.54, 1.0),
           ncol=2, frameon=False, fontsize=8.5, handlelength=2.7,
           columnspacing=1.8)
save(fig, 'non_gaussian_profiles')

print('Saved error curves and profiles as PDF and PNG:', OUT)
