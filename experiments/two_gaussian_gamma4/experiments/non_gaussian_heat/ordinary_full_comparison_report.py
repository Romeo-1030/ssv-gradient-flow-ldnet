"""Four-arm ordinary two-Gaussian comparison, using saved predictions only.

No models, optimizers or training runners are imported. The source dataset
and shared gamma are checked, and all aggregate comparisons use the same
completed seed intersection across the four arms. Existing source reports
and predictions are never changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ic_gamma_report import _load_data, _method, _prediction, _read, _rows, _save


METHODS = ('latent', 'latent_gf', 'ssv', 'ssv_gf')
LABELS = {'latent': 'LDNet', 'latent_gf': 'LDNet + GF',
          'ssv': 'LDNet + shared SSV', 'ssv_gf': 'LDNet + shared SSV + GF'}
COLORS = {'latent': '#cc6677', 'latent_gf': '#4477aa',
          'ssv': '#aa3377', 'ssv_gf': '#228833'}
METRICS = ('observed_0_1p25', 'long_2_6', 'final_6', 'long_2_10',
           'extrapolation_1p25_10', 'final_10')
WINDOWS = ('Obs [0,1.25]', 'Long (2,6]', 'Final 6', 'Long (2,10]',
           'Ext (1.25,10]', 'Final 10')
CONTRASTS = {
    'gf_without_ssv': ('latent', 'latent_gf'),
    'gf_with_ssv': ('ssv', 'ssv_gf'),
    'ssv_without_gf': ('latent', 'ssv'),
    'ssv_with_gf': ('latent_gf', 'ssv_gf'),
}
OBSERVED = (0., .25, .5, .75, 1., 1.25)
EXAMPLE_ID = 'test_001'
EXAMPLE_SEED = 0
EXAMPLE_TIMES = (0., .5, 1.25, 2., 6., 10.)


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Expected a finite metric vector')
    return {'count': len(values), 'mean': float(values.mean()) if len(values) else None,
            'std': float(values.std(ddof=0)) if len(values) else None,
            'std_ddof': 0, 'values': values.tolist()}


def _percent(stats):
    if not stats['count']:
        return 'pending'
    return f"{100*stats['mean']:.3f} +/- {100*stats['std']:.3f}"


def _time_index(t, value):
    indices = np.flatnonzero(np.isclose(t, value, rtol=0., atol=1e-9))
    if len(indices) != 1:
        raise ValueError(f'Expected one saved time at t={value}')
    return int(indices[0])


def _metrics(curve, t):
    return dict(zip(METRICS, (
        float(curve[(t >= 0) & (t <= 1.25)].mean()),
        float(curve[(t > 2) & (t <= 6)].mean()),
        float(curve[_time_index(t, 6.)]),
        float(curve[(t > 2) & (t <= 10)].mean()),
        float(curve[(t > 1.25) & (t <= 10)].mean()),
        float(curve[_time_index(t, 10.)]),
    )))


def _data_audit(data, source_data):
    if not data or not source_data:
        raise ValueError('New and source runs must both contain data.npz')
    if set(data) != set(source_data):
        raise ValueError('Source and new data.npz keys differ')
    for key in sorted(data):
        if data[key].dtype != source_data[key].dtype:
            raise ValueError(f'Data dtype differs: {key}')
        np.testing.assert_array_equal(data[key], source_data[key], err_msg=f'Data differs: {key}')
    np.testing.assert_allclose(data['x'], np.linspace(-16., 16., 192, endpoint=False),
                               rtol=0., atol=1e-12)
    np.testing.assert_allclose(data['t'], np.linspace(0., 10., 81), rtol=0., atol=1e-12)
    return {'passed': True, 'keys_checked': sorted(data),
            'comparison': 'Exact values, shapes and dtypes of every data.npz array.'}


def _collect(out, data):
    records = {method: {} for method in METHODS}
    omitted = []
    test = np.asarray(data['split']).astype(str) == 'test'
    expected_ids = np.asarray(data['sample_id'][test]).astype(str)
    for row in _rows(_read(out/'results.json', [])):
        method = _method(row)
        if method not in METHODS or 'seed' not in row:
            continue
        seed = int(row['seed'])
        if seed in records[method]:
            raise ValueError(f'Duplicate completed record: {method}, seed {seed}')
        arrays = _prediction(out, row, data)
        if arrays is None:
            omitted.append({'method': method, 'seed': seed, 'reason': 'Prediction unavailable or invalid'})
            continue
        for key in ('x', 't'):
            np.testing.assert_array_equal(arrays[key], data[key], err_msg=f'{method}: {key}')
        np.testing.assert_array_equal(arrays['truth'], data['u'][test], err_msg=f'{method}: truth')
        ids = arrays.get('sample_ids', arrays.get('sample_id'))
        if ids is None:
            raise ValueError(f'{method}, seed {seed}: missing sample IDs')
        np.testing.assert_array_equal(np.asarray(ids).astype(str), expected_ids,
                                      err_msg=f'{method}: prediction ordering')
        truth = arrays['truth'].astype(np.float64)
        error = arrays['pred'].astype(np.float64)-truth
        denominator = np.square(truth).sum(axis=(0, 2))
        if np.any(denominator <= 0):
            raise ValueError('Zero pooled truth norm')
        curve = np.sqrt(np.square(error).sum(axis=(0, 2))/denominator)
        records[method][seed] = {'row': row, 'arrays': arrays, 'curve': curve,
                                'metrics': _metrics(curve, data['t'])}
    return records, omitted


def _training_audit(records, shared_gamma, smoke):
    """Validate arms as they complete, including before a four-way pair exists."""
    common_keys = (
        'backbone_parameters', 'decoder_extent', 'fourier_frequencies', 'label_times',
        'gamma_frozen', 'initial_parameters_sha256',
    )
    gf_keys = ('energy_parameters', 'initial_energy_sha256', 'gf_times',
               'gf_collocation', 'gf_spatial_extent')
    checked, skipped, cross_arm_seeds = [], [], []
    seeds = sorted(set().union(*(set(records[method]) for method in METHODS)))
    for seed in seeds:
        available = [(method, records[method][seed]) for method in METHODS if seed in records[method]]
        if len(available) > 1:
            cross_arm_seeds.append(seed)
        anchor = available[0][1]['row']
        for method, entry in available:
            row = entry['row']
            ssv, gf = method in ('ssv', 'ssv_gf'), method.endswith('_gf')
            required = dict(physical=not ssv, gf=gf, gamma_frozen=True,
                            scales=[[.5, .5]] if ssv else [[0., 0.]],
                            energy_parameters=4417 if gf else 0,
                            backbone_parameters=91329, decoder_extent=16., fourier_frequencies=16,
                            label_times=list(OBSERVED))
            for key, value in required.items():
                if row.get(key) != value:
                    raise ValueError(f'{method}, seed {seed}: expected {key}={value}')
            keys = common_keys+(() if smoke else ('steps',))
            for key in keys:
                if key not in row or key not in anchor or row[key] != anchor[key]:
                    raise ValueError(f'Cross-arm final protocol differs: seed {seed}, {key}')
            if smoke:
                skipped.append({'method': method, 'seed': seed, 'key': 'steps'})
            elif row['steps'] != 3000:
                raise ValueError('Production protocol requires 3000 final training steps')
            if ssv:
                gamma = entry['arrays'].get('gamma')
                if gamma is None or shared_gamma is None:
                    raise ValueError(f'{method}, seed {seed}: missing frozen shared gamma')
                np.testing.assert_allclose(gamma, shared_gamma, rtol=1e-6, atol=1e-8,
                                           err_msg=f'{method}: gamma must be shared without reselection')
            if not gf and (row.get('gf_times') != [] or row.get('gf_max_weight') != 0):
                raise ValueError(f'{method}: a no-GF arm has an active GF record')
            checked.append({'method': method, 'seed': seed, 'common_metadata_and_arm_flags': True})
        gf_records = [(method, entry['row']) for method, entry in available if method.endswith('_gf')]
        if gf_records:
            anchor_gf = gf_records[0][1]
            for method, row in gf_records:
                for key in gf_keys+(() if smoke else ('gf_max_weight',)):
                    if key not in row or key not in anchor_gf or row[key] != anchor_gf[key]:
                        raise ValueError(f'GF protocol differs: seed {seed}, {key}')
                if smoke:
                    skipped.append({'method': method, 'seed': seed, 'key': 'gf_max_weight'})
                elif row['gf_max_weight'] != .0001:
                    raise ValueError('Production GF maximum weight differs from 1e-4')
    return {'status': 'checked' if seeds else 'pending_completed_predictions',
            'passed': True if seeds else None, 'checked': checked, 'skipped_smoke_only': skipped,
            'same_seed_initial_weights': True if cross_arm_seeds else None,
            'cross_arm_comparison_seeds': cross_arm_seeds,
            'note': 'Initial backbone hashes are compared across available same-seed arms; energy hashes/grid across GF arms. '
                    'Data-dependent GF residual normalization and selected checkpoint step may differ.'}


def _plot_curves(out, data, records, seeds, outputs, plt):
    if not seeds:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), layout='constrained')
    t = data['t']
    for method in METHODS:
        curves = np.asarray([records[method][seed]['curve'] for seed in seeds])*100
        mean, sd = curves.mean(0), curves.std(0, ddof=0)
        for axis, end in zip(axes, (10., 6., 1.25)):
            keep = t <= end
            axis.plot(t[keep], np.maximum(mean[keep], 1e-8), color=COLORS[method],
                      lw=1.8, label=LABELS[method])
            if len(seeds) > 1:
                axis.fill_between(t[keep], np.maximum(mean[keep]-sd[keep], 1e-8),
                                  np.maximum(mean[keep]+sd[keep], 1e-8), color=COLORS[method], alpha=.12)
    for index, (axis, end) in enumerate(zip(axes, (10., 6., 1.25))):
        axis.set(xlim=(0, end), xlabel='Physical time t', ylabel='Full-domain RelRMSE (%)',
                 title=f't <= {end:g}', yscale='log' if index < 2 else 'linear')
        axis.axvspan(0, 1.25, color='gray', alpha=.06)
        axis.axvline(1.25, color='gray', ls=':', lw=.8)
        axis.grid(alpha=.2)
    for time in OBSERVED:
        axes[-1].axvline(time, color='gray', alpha=.16, lw=.6)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Ordinary two-Gaussian ICs; four-way completed seeds {seeds}\n'
                 'Mean +/- population SD; logarithmic error scale on extrapolation panels', fontsize=11)
    _save(fig, out/'figures'/'four_arm_error_curves.png', outputs, pdf=True)
    plt.close(fig)


def _plot_example(out, data, records, seeds, outputs, plt):
    info = {'sample_id': EXAMPLE_ID, 'seed': EXAMPLE_SEED, 'times': list(EXAMPLE_TIMES), 'status': 'pending',
            'selection': 'test_001 and optimizer seed 0 are fixed in advance; no error-based selection.'}
    if EXAMPLE_SEED not in seeds:
        return info
    test = np.asarray(data['split']).astype(str) == 'test'
    ids = np.asarray(data['sample_id'][test]).astype(str)
    indices = np.flatnonzero(ids == EXAMPLE_ID)
    if len(indices) != 1:
        raise ValueError('The prespecified test_001 must occur exactly once')
    index = int(indices[0])
    truth = data['u'][test][index]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.3), layout='constrained')
    for axis, time in zip(axes.flat, EXAMPLE_TIMES):
        j = _time_index(data['t'], time)
        axis.plot(data['x'], truth[j], 'k-', lw=2.2, label='Heat-equation reference')
        for method in METHODS:
            pred = records[method][EXAMPLE_SEED]['arrays']['pred'][index, j]
            axis.plot(data['x'], pred, color=COLORS[method], ls='--', lw=1.25, label=LABELS[method])
        axis.set(xlim=(-16., 16.), xlabel='x', ylabel='u(x,t)', title=f't = {time:g}')
        axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', fontsize=8, ncol=3)
    fig.suptitle('Prespecified two-peak test_001: four methods, optimizer seed 0', fontsize=12)
    filename = 'profiles_test_001_seed0_evolution.png'
    _save(fig, out/'figures'/filename, outputs, pdf=True)
    plt.close(fig)
    info.update(status='available', figure='figures/'+filename)
    return info


def _frozen_diagnostic(out):
    relative = 'diagnostics/frozen_latent/summary.json'
    saved = _read(out/relative, {})
    pending = {'status': 'pending', 'summary_path': relative,
               'note': 'Optional separate frozen-latent diagnostic; not run by this report.'}
    if not saved or not saved.get('summary'):
        return pending, ['', 'FROZEN-LATENT DIAGNOSTIC: pending', f'Expected summary: {relative}']
    if saved.get('units') != 'percent':
        pending['note'] = 'Diagnostic summary has unrecognized units; main trained-model report remains available.'
        return pending, ['', 'FROZEN-LATENT DIAGNOSTIC: awaiting a summary with explicit percent units.']
    compact = {}
    for method, record in saved['summary'].items():
        compact[method] = {'n': record.get('n'), 'seeds': record.get('seeds', [])}
        for variant in ('normal', 'frozen'):
            compact[method][variant] = {
                metric: value for metric, value in record.get(variant, {}).items()
                if isinstance(value, dict) and 'mean' in value
            }
    curve = 'diagnostics/frozen_latent/error_curves.png'
    curve = curve if (out/curve).exists() else None
    interpretation = saved.get('interpretation',
        'Prediction counterfactual using the same weights; not a separately trained no-latent or no-GF ablation.')
    info = {'status': 'available', 'summary_path': relative, 'curve_figure': curve,
            'units': 'percent', 'frozen_latent_time': saved.get('frozen_latent_time'),
            'interpretation': interpretation, 'summary': compact,
            'training_performed': saved.get('training_performed'),
            'weights_changed': saved.get('weights_changed'), 'gamma_changed': saved.get('gamma_changed')}
    lines = ['', 'FROZEN-LATENT DIAGNOSTIC: saved prediction counterfactual',
             f"Latent state held at t={saved.get('frozen_latent_time')}; gamma and trained weights remain unchanged.",
             'Values are already percentages, mean +/- population SD; normal -> frozen. Diagnostic seeds are listed per method.']

    def direct_percent(value):
        if not isinstance(value, dict) or value.get('mean') is None:
            return 'pending'
        return f"{value['mean']:.3f} +/- {value.get('std_ddof0', 0.):.3f}"

    for method, record in compact.items():
        values = []
        for metric in ('t6', 't10'):
            normal = direct_percent(record['normal'].get(metric))
            frozen = direct_percent(record['frozen'].get(metric))
            values.append(f'{metric}: {normal} -> {frozen}')
        lines.append(f"{LABELS.get(method, method)}; seeds={record['seeds']}: "+' | '.join(values))
    lines.extend([interpretation, f'Diagnostic summary: {relative}'])
    if curve:
        lines.append('Diagnostic curve: '+curve)
    return info, lines


def build_report(out: Path) -> list[Path]:
    out = Path(out).resolve()
    config = _read(out/'config.json', {})
    smoke = config.get('smoke_only') is True
    source_value = config.get('source_run', config.get('source_shared_run', config.get('parent')))
    if not source_value or not Path(source_value).is_absolute():
        raise ValueError('config["source_run"] must be the absolute source ordinary shared-gamma run')
    source = Path(source_value).resolve()
    if source == out:
        raise ValueError('The four-arm run must have its own output directory')
    data = _load_data(out)
    identity = _data_audit(data, _load_data(source))
    source_selection = _read(source/'shared_gamma_selection.json', {})
    gamma = source_selection.get('selected_gamma')
    if gamma is None or not np.isfinite(float(gamma)) or float(gamma) <= 0:
        raise ValueError('The completed source run must provide a positive selected shared gamma')
    gamma = float(gamma)
    selection = _read(out/'shared_gamma_selection.json', {})
    if 'selected_gamma' in selection and float(selection['selected_gamma']) != gamma:
        raise ValueError('Shared gamma must be reused from the source; no no-GF reselection')
    records, omitted = _collect(out, data)
    seeds = sorted(set.intersection(*(set(records[method]) for method in METHODS)))
    training = _training_audit(records, gamma, smoke)
    count = config.get('seeds', config.get('arguments', {}).get('seeds', 4))
    expected = list(range(count)) if isinstance(count, int) else [int(seed) for seed in count]
    completed = {method: sorted(records[method]) for method in METHODS}
    status = ('complete' if set(expected).issubset(seeds) else
              'partial' if seeds else 'pending_four_way_seed_intersection')
    summary = {
        'status': status, 'smoke_only': smoke, 'formal_comparison': not smoke,
        'run': str(out), 'source_run': str(source), 'expected_seeds': expected,
        'paired_seeds': seeds, 'completed_seeds': completed, 'omitted_records': omitted,
        'data_identity': identity, 'final_training_audit': training,
        'gamma': {'shared': gamma, 'source_file': str(source/'shared_gamma_selection.json'),
                  'frozen_during_final_training': True, 'reselected_for_no_gf': False,
                  'ssv_alpha_beta': [.5, .5], 'physical_alpha_beta': [0., 0.]},
        'metric': {
            'formula': 'sqrt(sum_test_IC,x((prediction(t)-truth(t))**2) / sum_test_IC,x(truth(t)**2))',
            'units_in_json': 'Fractions; multiply by 100 for percent or paired percentage points.',
            'time_aggregation': 'Arithmetic mean of per-time pooled error over every saved time in the specified interval.',
            'observed_window': '[0,1.25] includes all saved times, not only six supervised frames.',
            'std_ddof': 0, 'domain': [-16., 16.], 'nx': len(data['x']), 'final_time': 10.,
        },
        'arms': {}, 'paired_effects': {},
        'protocol_notes': [
            'All four arms use exactly the same ordinary two-Gaussian initial conditions, splits, spatial grid and six labels through t=1.25.',
            'Shared SSV uses the source-selected gamma, frozen with alpha=beta=0.5. Removing GF does not trigger another gamma selection.',
            'Physical-coordinate arms have alpha=beta=0. All arms share the backbone scale and matched same-seed initial weights.',
            'The existing source SSV+GF fits are reused; the other three arms are separately fitted from fresh initial weights.',
            'Every main-table statistic, paired contrast and curve uses the four-way completed seed intersection; completion counts are listed separately.',
            'This is a new six-frame experiment with a gamma selector, not a reproduction of the original paper\'s three-frame Gaussian experiment.',
            'Only saved predictions are analyzed. No additional network training or inference is performed by this report.',
        ],
    }
    for method in METHODS:
        summary['arms'][method] = {
            'label': LABELS[method],
            'metrics': {key: _stats([records[method][seed]['metrics'][key] for seed in seeds]) for key in METRICS},
            'per_seed': [{'seed': seed, **records[method][seed]['metrics']} for seed in seeds],
        }
    for name, (baseline, treatment) in CONTRASTS.items():
        effect = {'baseline': baseline, 'treatment': treatment, 'seeds': seeds,
                  'sign': 'Treatment minus baseline; negative is lower error after adding the component.', 'metrics': {}}
        for key in METRICS:
            differences = [records[treatment][seed]['metrics'][key]-records[baseline][seed]['metrics'][key]
                           for seed in seeds]
            effect['metrics'][key] = {**_stats(differences),
                                      'improved_seed_count': sum(value < 0 for value in differences)}
        summary['paired_effects'][name] = effect
    outputs = []
    (out/'figures').mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _plot_curves(out, data, records, seeds, outputs, plt)
    summary['prespecified_example'] = _plot_example(out, data, records, seeds, outputs, plt)
    summary['curves'] = {'t': data['t'].tolist(),
                         **{method: {str(seed): records[method][seed]['curve'].tolist() for seed in seeds}
                            for method in METHODS}}
    previous_summary = _read(out/'report_summary.json', {})
    summary['diagnostics'] = previous_summary.get('diagnostics', {})
    summary['diagnostics']['frozen_latent'], diagnostic_lines = _frozen_diagnostic(out)
    summary['figures'] = [str(path.relative_to(out)).replace('\\', '/') for path in outputs]
    lines = [
        'Ordinary two-Gaussian ICs: four-arm SSV / GF comparison',
        ('SMOKE TEST ONLY: mixed shortened/reused training; no formal accuracy conclusion.' if smoke else
         'Production experiment: common data, six observation frames, domain [-16,16], rollout through t=10.'),
        f'Status: {status}; four-way completed seeds: {seeds}; expected: {expected}',
        'Completed seeds by arm: '+json.dumps(completed),
        f'Shared gamma: {gamma}; source: {source}; fixed alpha=beta=0.5 in both SSV arms.',
        'All source and new data.npz arrays match exactly. SSV without GF reuses gamma without another search.',
        'Original-paper pooled current-time full-domain RelRMSE. Mean +/- population seed SD (ddof=0).',
        'Main table values are percentages. Observed includes all grid times [0,1.25].',
        'One-seed SD=0 is not evidence of reproducibility.', '',
        f"{'Method':<30} "+' | '.join(f'{window:<22}' for window in WINDOWS),
    ]
    for method in METHODS:
        lines.append(f'{LABELS[method]:<30} '+' | '.join(
            f'{_percent(summary["arms"][method]["metrics"][key]):<22}' for key in METRICS))
    lines.extend(['', 'PAIRED COMPONENT EFFECTS: treatment minus baseline, percentage points',
                  'Negative means lower error after adding GF or SSV. SD is calculated across seed-level differences.',
                  'These small-sample descriptive differences are not a significance test.'])
    for name, effect in summary['paired_effects'].items():
        lines.append(f'{name}: {LABELS[effect["treatment"]]} minus {LABELS[effect["baseline"]]}')
        lines.append(' | '.join(f'{window}: {_percent(effect["metrics"][key])}' for window, key in zip(WINDOWS, METRICS)))
    lines.extend(['', 'PROTOCOL']+summary['protocol_notes'])
    lines.extend(['', 'Prespecified example: '+json.dumps(summary['prespecified_example']),
                  'Final training audit: '+training['status'], '', 'Generated figures:']+summary['figures'])
    lines.extend(diagnostic_lines)
    report_path, summary_path = out/'REPORT.txt', out/'report_summary.json'
    report_path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    return [report_path, summary_path, *outputs]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    for generated in build_report(args.out):
        print(generated)
