"""Saved-prediction comparison of shared and per-IC gamma on domain 16.

This module never trains or evaluates a network. It writes only to the new
shared-gamma run, and derives every reported error from saved test arrays.
Incomplete runs use the intersection of completed optimizer seeds in both
arms; neither the example seed nor the example trajectory is error-selected.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ic_gamma_report import _load_data, _method, _prediction, _read, _rows, _save


ARMS = ('shared_gamma_gf', 'per_ic_gamma_gf')
LABELS = {
    'shared_gamma_gf': 'Shared gamma + SSV + LDNet + GF',
    'per_ic_gamma_gf': 'Per-IC gamma + SSV + LDNet + GF',
}
COLORS = {'shared_gamma_gf': '#376ab1', 'per_ic_gamma_gf': '#25866b'}
METRICS = (
    'observed_0_1p25', 'long_2_6', 'final_6',
    'long_2_10', 'extrapolation_1p25_10', 'final_10',
)
METRIC_LABELS = {
    'observed_0_1p25': 'Observed [0,1.25]',
    'long_2_6': 'Long (2,6]',
    'final_6': 'Final t=6',
    'long_2_10': 'Long (2,10]',
    'extrapolation_1p25_10': 'Extrapolation (1.25,10]',
    'final_10': 'Final t=10',
}
EXAMPLE_ID = 'test_014'
EXAMPLE_TIMES = (0., .5, 1.25, 2., 6., 10.)
OBSERVED = np.asarray([0., .25, .5, .75, 1., 1.25])


def _stats(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError('Report statistics require a finite vector')
    return {
        'count': len(a), 'mean': float(a.mean()) if len(a) else None,
        'std': float(a.std(ddof=0)) if len(a) else None,
        'std_ddof': 0, 'values': a.tolist(),
    }


def _format(stats):
    if not stats['count']:
        return 'pending'
    suffix = ' (one seed)' if stats['count'] == 1 else ''
    return f"{100*stats['mean']:.3f} +/- {100*stats['std']:.3f}{suffix}"


def _time_index(t, value):
    indices = np.flatnonzero(np.isclose(t, value, rtol=0., atol=1e-9))
    if len(indices) != 1:
        raise ValueError(f'Expected exactly one sample at t={value}')
    return int(indices[0])


def _summarize_curve(curve, t):
    """Errors are fractions; windows average sampled times, not squared errors."""
    return {
        'observed_0_1p25': float(curve[(t >= 0) & (t <= 1.25)].mean()),
        'long_2_6': float(curve[(t > 2) & (t <= 6)].mean()),
        'final_6': float(curve[_time_index(t, 6.)]),
        'long_2_10': float(curve[(t > 2) & (t <= 10)].mean()),
        'extrapolation_1p25_10': float(curve[(t > 1.25) & (t <= 10)].mean()),
        'final_10': float(curve[_time_index(t, 10.)]),
    }


def _same_data(shared, parent):
    if not shared or not parent:
        raise ValueError('Both runs must contain data.npz before reporting')
    if set(shared) != set(parent):
        raise ValueError('The two data.npz files have different array keys')
    for key in sorted(shared):
        if shared[key].dtype != parent[key].dtype:
            raise ValueError(f'Data dtype differs for {key}')
        np.testing.assert_array_equal(shared[key], parent[key], err_msg=f'Data differs: {key}')
    x, t = shared['x'], shared['t']
    np.testing.assert_allclose(x, np.linspace(-16., 16., 192, endpoint=False), rtol=0., atol=1e-12)
    np.testing.assert_allclose(t, np.linspace(0., 10., 81), rtol=0., atol=1e-12)
    return {'passed': True, 'checked_keys': sorted(shared),
            'comparison': 'Exact array values, shapes and dtypes; no resampling or interpolation.'}


def _collect(run, data):
    test = np.asarray(data['split']).astype(str) == 'test'
    expected_ids = np.asarray(data['sample_id'][test]).astype(str)
    records, omitted = {}, []
    for row in _rows(_read(run/'results.json', [])):
        if _method(row) != 'ssv_gf' or 'seed' not in row:
            continue
        seed = int(row['seed'])
        if seed in records:
            raise ValueError(f'Duplicate completed ssv_gf seed {seed} in {run}')
        arrays = _prediction(run, row, data)
        if arrays is None:
            omitted.append({'seed': seed, 'reason': 'Saved prediction unavailable or invalid'})
            continue
        for key in ('x', 't'):
            np.testing.assert_array_equal(arrays[key], data[key], err_msg=f'{run.name}: {key}')
        np.testing.assert_array_equal(arrays['truth'], data['u'][test], err_msg=f'{run.name}: truth')
        ids = arrays.get('sample_ids', arrays.get('sample_id'))
        if ids is None:
            raise ValueError(f'Prediction sample IDs are required: {run.name}, seed {seed}')
        np.testing.assert_array_equal(np.asarray(ids).astype(str), expected_ids,
                                      err_msg=f'{run.name}: prediction ordering')
        truth = arrays['truth'].astype(np.float64)
        error = arrays['pred'].astype(np.float64)-truth
        denominator = np.square(truth).sum(axis=(0, 2))
        if not np.all(denominator > 0):
            raise ValueError('Zero truth norm in test metric')
        curve = np.sqrt(np.square(error).sum(axis=(0, 2))/denominator)
        records[seed] = {'row': row, 'arrays': arrays, 'curve': curve,
                         'metrics': _summarize_curve(curve, data['t'])}
    return records, omitted


def _pair_audit(shared, per_ic, seeds, smoke_only=False):
    # Normalization of the GF residual may differ with gamma; it is not a
    # shared hyperparameter and intentionally is not asserted equal here.
    keys = (
        'steps', 'backbone_parameters', 'energy_parameters', 'physical', 'gf',
        'gamma_frozen', 'scales', 'decoder_extent', 'fourier_frequencies',
        'label_times', 'gf_times', 'gf_collocation', 'gf_spatial_extent',
        'gf_max_weight', 'initial_parameters_sha256', 'initial_energy_sha256',
    )
    skipped_keys = ('steps', 'gf_max_weight') if smoke_only else ()
    checked, unavailable, skipped = [], [], []
    for seed in seeds:
        left, right = shared[seed]['row'], per_ic[seed]['row']
        for key in keys:
            if key in skipped_keys:
                skipped.append({'seed': seed, 'key': key, 'reason': 'Explicit smoke-only run'})
            elif key not in left or key not in right:
                unavailable.append({'seed': seed, 'key': key})
            elif left[key] != right[key]:
                raise ValueError(f'Final-training matching failed for seed {seed}: {key}')
            else:
                checked.append({'seed': seed, 'key': key})
        for row in (left, right):
            if row.get('gamma_frozen') is False:
                raise ValueError('Comparison requires frozen gamma during final LDNet training')
    return {'passed': not unavailable if seeds else None, 'checked': checked,
            'unavailable': unavailable, 'skipped_smoke_only': skipped,
            'status': 'checked' if seeds else 'pending_completed_pairs',
            'note': 'Gamma assignments and data-dependent GF normalization may differ.'}


def _plot_curves(out, data, records, seeds, outputs, plt):
    if not seeds:
        return
    t = data['t']
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), layout='constrained')
    for arm in ARMS:
        curves = np.asarray([records[arm][seed]['curve'] for seed in seeds])*100
        mean, sd = curves.mean(0), curves.std(0, ddof=0)
        for axis, endpoint in zip(axes, (10., 6., 1.25)):
            mask = t <= endpoint
            axis.plot(t[mask], mean[mask], color=COLORS[arm], lw=1.8, label=LABELS[arm])
            if len(seeds) > 1:
                axis.fill_between(t[mask], np.maximum(0., mean[mask]-sd[mask]),
                                  mean[mask]+sd[mask], color=COLORS[arm], alpha=.15)
    for axis, endpoint in zip(axes, (10., 6., 1.25)):
        axis.set(xlim=(0, endpoint), xlabel='Physical time t', ylabel='Full-domain RelRMSE (%)',
                 title=f't <= {endpoint:g}')
        axis.axvspan(0, 1.25, color='gray', alpha=.07)
        axis.axvline(1.25, color='gray', ls=':', lw=.8)
        axis.grid(alpha=.2)
    for time in OBSERVED:
        axes[-1].axvline(time, color='gray', alpha=.18, lw=.6)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Identical completed seed subset: {seeds}; mean +/- population SD', fontsize=12)
    _save(fig, out/'figures'/'paired_error_curves.png', outputs, pdf=True)
    plt.close(fig)

    differences = np.asarray([records[ARMS[0]][seed]['curve']-
                              records[ARMS[1]][seed]['curve'] for seed in seeds])*100
    fig, axis = plt.subplots(figsize=(9.2, 4.5), layout='constrained')
    for seed, curve in zip(seeds, differences):
        axis.plot(t, curve, lw=.8, alpha=.45, label=f'Seed {seed}')
    mean, sd = differences.mean(0), differences.std(0, ddof=0)
    axis.plot(t, mean, color='#6d3d89', lw=2, label='Paired mean')
    if len(seeds) > 1:
        axis.fill_between(t, mean-sd, mean+sd, color='#6d3d89', alpha=.13)
    axis.axhline(0, color='black', lw=.8)
    axis.axvline(1.25, color='gray', ls=':', lw=.8)
    axis.set(xlim=(0, 10), xlabel='Physical time t', ylabel='Shared minus per-IC (percentage points)',
             title='Paired error differences: negative values favor shared gamma')
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
    _save(fig, out/'figures'/'paired_error_difference.png', outputs, pdf=True)
    plt.close(fig)


def _plot_example(out, data, records, seeds, outputs, plt):
    metadata = {'sample_id': EXAMPLE_ID, 'status': 'pending', 'seed': None,
                'times': list(EXAMPLE_TIMES),
                'selection': 'Prespecified trajectory; smallest completed common seed, never best error.'}
    if not seeds:
        return metadata
    test_ids = np.asarray(data['sample_id'][np.asarray(data['split']).astype(str) == 'test']).astype(str)
    indices = np.flatnonzero(test_ids == EXAMPLE_ID)
    if not len(indices):
        metadata['status'] = 'sample_unavailable'
        return metadata
    if len(indices) != 1:
        raise ValueError(f'Duplicate {EXAMPLE_ID}')
    index, seed = int(indices[0]), min(seeds)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.1), layout='constrained')
    reference = records[ARMS[0]][seed]['arrays']
    for axis, time in zip(axes.flat, EXAMPLE_TIMES):
        j = _time_index(data['t'], time)
        axis.plot(data['x'], reference['truth'][index, j], color='black', lw=2,
                  label='Heat-equation reference')
        for arm in ARMS:
            axis.plot(data['x'], records[arm][seed]['arrays']['pred'][index, j],
                      color=COLORS[arm], lw=1.5, ls='--', label=LABELS[arm])
        axis.set(xlim=(-16., 16.), xlabel='x', ylabel='u(x,t)', title=f't = {time:g}')
        axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', fontsize=8, ncol=2)
    fig.suptitle(f'Prespecified {EXAMPLE_ID}; same optimizer seed {seed}; domain [-16,16]', fontsize=12)
    name = 'profiles_test_014_evolution.png'
    _save(fig, out/'figures'/name, outputs, pdf=True)
    plt.close(fig)
    metadata.update(status='available', seed=seed, figure='figures/'+name)
    return metadata


def build_report(out: Path, parent: Path | None = None) -> list[Path]:
    """Write comparison artifacts to ``out``; never change the parent run."""
    out = Path(out).resolve()
    config = _read(out/'config.json', {})
    smoke_only = config.get('smoke_only') is True
    parent_value = parent if parent is not None else config.get('parent', config.get('parent_run'))
    if not parent_value:
        raise ValueError('Provide parent= or an absolute config["parent"] path')
    parent = Path(parent_value)
    if not parent.is_absolute():
        raise ValueError('Parent run path must be absolute')
    parent = parent.resolve()
    if parent == out:
        raise ValueError('Shared and parent runs must be different directories')
    data, parent_data = _load_data(out), _load_data(parent)
    data_audit = _same_data(data, parent_data)
    shared, omitted_shared = _collect(out, data)
    per_ic, omitted_parent = _collect(parent, parent_data)
    records = dict(zip(ARMS, (shared, per_ic)))
    seeds = sorted(set(shared) & set(per_ic))
    training_audit = _pair_audit(shared, per_ic, seeds, smoke_only=smoke_only)
    selection = _read(out/'shared_gamma_selection.json', {})
    selected_gamma = selection.get('selected_gamma')
    if selected_gamma is not None:
        selected_gamma = float(selected_gamma)
        if not np.isfinite(selected_gamma) or selected_gamma <= 0:
            raise ValueError('Selected shared gamma must be positive and finite')
        for seed, record in shared.items():
            if 'gamma' not in record['arrays']:
                raise ValueError(f'Shared seed {seed} has no saved test gamma')
            np.testing.assert_allclose(record['arrays']['gamma'], selected_gamma, rtol=1e-6, atol=1e-8,
                                       err_msg=f'Shared gamma assignment differs for seed {seed}')
    elif shared:
        raise ValueError('Completed shared predictions require a selected_gamma record')
    expected_value = config.get('seeds', 4)
    expected = list(range(int(expected_value))) if isinstance(expected_value, int) else [int(s) for s in expected_value]
    status = ('pending_gamma_selection' if selected_gamma is None else
              'pending_final_predictions' if not seeds else
              'complete' if set(expected).issubset(seeds) else 'partial')
    summary = {
        'status': status, 'smoke_only': smoke_only,
        'formal_comparison': not smoke_only,
        'shared_run': str(out), 'parent_run': str(parent), 'expected_seeds': expected,
        'paired_seeds': seeds,
        'completed_seeds': {arm: sorted(records[arm]) for arm in ARMS},
        'omitted_records': {ARMS[0]: omitted_shared, ARMS[1]: omitted_parent},
        'data_identity': data_audit, 'final_training_matching': training_audit,
        'metric': {
            'formula': 'sqrt(sum_test_IC,x((prediction(t)-truth(t))**2) / sum_test_IC,x(truth(t)**2))',
            'units_in_json': 'fraction; multiply by 100 for percent or paired percentage points',
            'window_average': 'Arithmetic mean of per-time pooled errors over every saved time in the interval.',
            'observed_window': 'All grid times in [0,1.25]; six supervised frames are a subset.',
            'seed_std_ddof': 0, 'spatial_domain': [-16., 16.], 'nx': len(data['x']),
        },
        'shared_gamma_selection': {'file': 'shared_gamma_selection.json', 'selected_gamma': selected_gamma,
                                   'candidates': selection.get('candidates'), 'scores': selection.get('scores')},
        'comparison_scope': [
            'Complete pipeline comparison: gamma selection procedure and assignments differ.',
            'Shared gamma is chosen using validation trajectories only and then fixed for every trajectory.',
            'Per-IC gamma uses each trajectory\'s six early frames, including six-frame calibration for test trajectories.',
            'Both final models share the LDNet/GF architecture and freeze gamma before final training.',
            'Only completed same-seed pairs enter aggregate numbers, paired differences and figures.',
            'These results do not isolate shared versus individual gamma from the choice of selector.',
        ],
        'arms': {}, 'paired_difference_shared_minus_per_ic': {},
    }
    for arm in ARMS:
        summary['arms'][arm] = {
            'label': LABELS[arm],
            'metrics': {metric: _stats([records[arm][seed]['metrics'][metric] for seed in seeds]) for metric in METRICS},
            'per_seed': [{'seed': seed, **records[arm][seed]['metrics']} for seed in seeds],
        }
    for metric in METRICS:
        values = [shared[seed]['metrics'][metric]-per_ic[seed]['metrics'][metric] for seed in seeds]
        summary['paired_difference_shared_minus_per_ic'][metric] = {
            **_stats(values), 'seeds': seeds,
            'shared_better_count': sum(value < 0 for value in values),
        }
    outputs = []
    (out/'figures').mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _plot_curves(out, data, records, seeds, outputs, plt)
    summary['prespecified_example'] = _plot_example(out, data, records, seeds, outputs, plt)
    summary['curves'] = {
        't': data['t'].tolist(),
        **{arm: {str(seed): records[arm][seed]['curve'].tolist() for seed in seeds} for arm in ARMS},
    }
    summary['figures'] = [str(path.relative_to(out)).replace('\\', '/') for path in outputs]
    report = [
        'Shared gamma versus per-IC gamma: domain [-16,16], six frames, t=10',
        ('SMOKE TEST ONLY: not a formal accuracy comparison. Final-training steps and GF maximum weight may differ.'
         if smoke_only else 'Formal experiment report; final-training matching checks are enforced.'),
        f"Status: {summary['status']}; paired optimizer seeds: {seeds}; expected: {expected}",
        f'Completed shared seeds: {sorted(shared)}; completed per-IC seeds: {sorted(per_ic)}',
        f'Shared selected gamma: {selected_gamma}',
        f'Parent per-IC run (read only): {parent}',
        'All data.npz arrays match exactly. Errors recomputed from saved predictions; no training or inference.',
        'Metric: original-paper pooled full-domain current-time RelRMSE, arithmetic time-window mean.',
        'Values below are percentages, mean +/- population SD across the same completed seeds (ddof=0).',
        'Observed averages all saved times [0,1.25], not only the six labeled frames.',
        '',
        f"{'Window':<30} {'Shared gamma + GF':<26} {'Per-IC gamma + GF':<26} Paired shared - per-IC (pp)",
    ]
    for metric in METRICS:
        values = [_format(summary['arms'][arm]['metrics'][metric]) for arm in ARMS]
        difference = _format(summary['paired_difference_shared_minus_per_ic'][metric])
        report.append(f'{METRIC_LABELS[metric]:<30} {values[0]:<26} {values[1]:<26} {difference}')
    report.extend(['', 'Negative paired differences favor shared gamma. SD uses paired differences, not independent-error propagation.',
                   'One seed has SD zero by definition; this is not evidence of reproducibility.', '',
                   'Pipeline interpretation:'] + summary['comparison_scope'])
    if training_audit['unavailable']:
        report.append('Training-matching metadata incomplete: '+str(training_audit['unavailable']))
    report.extend(['', 'Prespecified example: '+json.dumps(summary['prespecified_example']),
                   '', 'Generated figures:'] + summary['figures'])
    report_path, summary_path = out/'REPORT.txt', out/'report_summary.json'
    report_path.write_text('\n'.join(report)+'\n', encoding='utf-8')
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    return [report_path, summary_path, *outputs]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--parent', type=Path)
    args = parser.parse_args()
    for generated in build_report(args.out, args.parent):
        print(generated)
