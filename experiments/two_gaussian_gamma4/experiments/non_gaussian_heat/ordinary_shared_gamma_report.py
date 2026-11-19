"""Reports for ordinary two-Gaussian ICs with shared-dictionary gamma and GF.

Only saved arrays are read. No model is imported, evaluated, or trained.
The controlled-IC shared-gamma experiment is a descriptive reference from a
different IC distribution, never a paired accuracy comparison or a baseline
trained on this dataset. All outputs are written to the new run directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ic_gamma_report import _load_data, _method, _prediction, _read, _rows, _save


METRICS = ('observed_0_1p25', 'long_2_6', 'final_6', 'long_2_10',
           'extrapolation_1p25_10', 'final_10')
LABELS = ('Observed [0,1.25]', 'Long (2,6]', 'Final t=6', 'Long (2,10]',
          'Extrapolation (1.25,10]', 'Final t=10')
OBSERVED = (0., .25, .5, .75, 1., 1.25)
EXAMPLE_TIMES = (0., .5, 1.25, 2., 6., 10.)
COLOR = '#376ab1'


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
    value = f"{100*stats['mean']:.3f} +/- {100*stats['std']:.3f}"
    return value+' (one seed)' if stats['count'] == 1 else value


def _time_index(t, value):
    indices = np.flatnonzero(np.isclose(t, value, rtol=0., atol=1e-9))
    if len(indices) != 1:
        raise ValueError(f'Expected exactly one saved time at {value}')
    return int(indices[0])


def _grid(data):
    if not data:
        raise ValueError('data.npz is required before reporting')
    np.testing.assert_allclose(data['x'], np.linspace(-16., 16., 192, endpoint=False),
                               rtol=0., atol=1e-12)
    np.testing.assert_allclose(data['t'], np.linspace(0., 10., 81), rtol=0., atol=1e-12)


def _curve_metrics(curve, t):
    return dict(zip(METRICS, (
        float(curve[(t >= 0) & (t <= 1.25)].mean()),
        float(curve[(t > 2) & (t <= 6)].mean()),
        float(curve[_time_index(t, 6.)]),
        float(curve[(t > 2) & (t <= 10)].mean()),
        float(curve[(t > 1.25) & (t <= 10)].mean()),
        float(curve[_time_index(t, 10.)]),
    )))


def _collect(out, data):
    """Validate ordering against this run's truth, not the other distribution."""
    test = np.asarray(data['split']).astype(str) == 'test'
    expected_ids = np.asarray(data['sample_id'][test]).astype(str)
    records, omitted = {}, []
    for row in _rows(_read(out/'results.json', [])):
        if _method(row) != 'ssv_gf' or 'seed' not in row:
            continue
        seed = int(row['seed'])
        if seed in records:
            raise ValueError(f'Duplicate ssv_gf seed {seed} in {out}')
        arrays = _prediction(out, row, data)
        if arrays is None:
            omitted.append({'seed': seed, 'reason': 'Prediction missing or invalid'})
            continue
        for key in ('x', 't'):
            np.testing.assert_array_equal(arrays[key], data[key], err_msg=f'{out.name}: {key}')
        np.testing.assert_array_equal(arrays['truth'], data['u'][test], err_msg=f'{out.name}: truth')
        ids = arrays.get('sample_ids', arrays.get('sample_id'))
        if ids is None:
            raise ValueError(f'{out.name}, seed {seed}: missing prediction sample IDs')
        np.testing.assert_array_equal(np.asarray(ids).astype(str), expected_ids,
                                      err_msg=f'{out.name}: sample ordering')
        truth = arrays['truth'].astype(np.float64)
        error = arrays['pred'].astype(np.float64)-truth
        denominator = np.square(truth).sum(axis=(0, 2))
        if np.any(denominator <= 0):
            raise ValueError('A test field has zero pooled norm')
        curve = np.sqrt(np.square(error).sum(axis=(0, 2))/denominator)
        records[seed] = {'row': row, 'arrays': arrays, 'curve': curve,
                         'metrics': _curve_metrics(curve, data['t'])}
    return records, omitted


def _check_shared_gamma(records, selection, run_name):
    gamma = selection.get('selected_gamma')
    if gamma is None:
        if records:
            raise ValueError(f'{run_name}: completed predictions have no shared selected_gamma')
        return None
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('Shared gamma must be finite and positive')
    for seed, record in records.items():
        saved = record['arrays'].get('gamma')
        if saved is None:
            raise ValueError(f'{run_name}, seed {seed}: missing saved gamma')
        np.testing.assert_allclose(saved, gamma, rtol=1e-6, atol=1e-8,
                                   err_msg=f'{run_name}: shared gamma mismatch')
    return gamma


def _protocol_audit(records, reference, smoke):
    """Same-seed initialization is a protocol check, not an error pairing."""
    keys = (
        'steps', 'backbone_parameters', 'energy_parameters', 'physical', 'gf',
        'gamma_frozen', 'scales', 'decoder_extent', 'fourier_frequencies',
        'label_times', 'gf_times', 'gf_collocation', 'gf_spatial_extent',
        'gf_max_weight', 'initial_parameters_sha256', 'initial_energy_sha256',
    )
    skip = ('steps', 'gf_max_weight') if smoke else ()
    checked, skipped = [], []
    for seed, record in sorted(records.items()):
        if seed not in reference:
            raise ValueError(f'No completed reference seed {seed} for final-protocol validation')
        row, other = record['row'], reference[seed]['row']
        for key in keys:
            if key in skip:
                skipped.append({'seed': seed, 'key': key, 'reason': 'Smoke-only shortened run'})
            elif key not in row or key not in other:
                raise ValueError(f'Missing final-protocol field {key} for seed {seed}')
            elif row[key] != other[key]:
                raise ValueError(f'Final protocol differs for seed {seed}: {key}')
            else:
                checked.append({'seed': seed, 'key': key})
        if not row['gamma_frozen'] or not row['gf'] or row['physical']:
            raise ValueError('Expected SSV+GF with gamma frozen before final training')
    return {'status': 'checked' if records else 'pending_completed_predictions',
            'passed': True if records else None, 'checked': checked, 'skipped_smoke_only': skipped,
            'scope': 'Final-training metadata and initialization only; IC values and prediction errors are not paired.',
            'normalization_note': 'The data-dependent GF residual normalization may differ.'}


def _aggregate(records):
    seeds = sorted(records)
    return {'completed_seeds': seeds,
            'metrics': {key: _stats([records[s]['metrics'][key] for s in seeds]) for key in METRICS},
            'per_seed': [{'seed': s, **records[s]['metrics']} for s in seeds]}


def _plot_curves(out, data, records, reference, outputs, plt):
    if not records:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), layout='constrained')
    for source, color, style, label in (
        (records, COLOR, '-', 'Ordinary two-Gaussian ICs'),
        (reference, '#777777', '--', 'Controlled ICs: descriptive reference'),
    ):
        if not source:
            continue
        curves = np.asarray([source[s]['curve'] for s in sorted(source)])*100
        mean, sd = curves.mean(0), curves.std(0, ddof=0)
        for axis, end in zip(axes, (10., 6., 1.25)):
            mask = data['t'] <= end
            axis.plot(data['t'][mask], mean[mask], color=color, ls=style, lw=1.8,
                      label=f'{label} (n={len(source)})')
            if len(source) > 1:
                axis.fill_between(data['t'][mask], np.maximum(0, mean[mask]-sd[mask]),
                                  mean[mask]+sd[mask], color=color, alpha=.12)
    for axis, end in zip(axes, (10., 6., 1.25)):
        axis.set(xlim=(0, end), xlabel='Physical time t', ylabel='Full-domain RelRMSE (%)',
                 title=f't <= {end:g}')
        axis.axvspan(0, 1.25, color='gray', alpha=.06)
        axis.axvline(1.25, color='gray', ls=':', lw=.8)
        axis.grid(alpha=.2)
    for time in OBSERVED:
        axes[-1].axvline(time, color='gray', alpha=.16, lw=.6)
    axes[0].legend(fontsize=7)
    fig.suptitle('Shared gamma + SSV + LDNet + GF; different IC distributions, descriptive reference only\n'
                 'Mean +/- population SD; each distribution uses its own completed seeds', fontsize=11)
    _save(fig, out/'figures'/'error_curves.png', outputs, pdf=True)
    plt.close(fig)


def _test_data(data):
    mask = np.asarray(data['split']).astype(str) == 'test'
    return np.asarray(data['sample_id'][mask]).astype(str), data['u'][mask]


def _plot_example(out, data, records, sample, outputs, plt):
    info = {'sample_id': sample, 'seed': None, 'status': 'pending', 'times': list(EXAMPLE_TIMES),
            'selection': 'Sample fixed by data_audit before training; smallest completed seed, never best error.'}
    if not records:
        return info
    ids, truth = _test_data(data)
    positions = np.flatnonzero(ids == sample)
    if len(positions) != 1:
        raise ValueError(f'Prespecified example {sample} not uniquely present in test data')
    index, seed = int(positions[0]), min(records)
    prediction = records[seed]['arrays']['pred'][index]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.1), layout='constrained')
    for axis, time in zip(axes.flat, EXAMPLE_TIMES):
        j = _time_index(data['t'], time)
        axis.plot(data['x'], truth[index, j], 'k-', lw=2, label='Heat-equation reference')
        axis.plot(data['x'], prediction[j], color=COLOR, ls='--', lw=1.6,
                  label='Shared gamma + SSV + LDNet + GF')
        axis.set(xlim=(-16., 16.), xlabel='x', ylabel='u(x,t)', title=f't = {time:g}')
        axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', fontsize=9, ncol=2)
    fig.suptitle(f'Ordinary two-Gaussian ICs: prespecified {sample}, optimizer seed {seed}', fontsize=12)
    filename = 'profiles_prespecified_evolution.png'
    _save(fig, out/'figures'/filename, outputs, pdf=True)
    plt.close(fig)
    info.update(status='available', seed=seed, figure='figures/'+filename)
    return info


def _plot_initials(out, data, sample, outputs, plt):
    ids, truth = _test_data(data)
    requested = list(dict.fromkeys(['test_000', 'test_001', 'test_002', sample]))
    chosen = [name for name in requested if name in ids]
    if not chosen:
        return {'sample_ids': [], 'status': 'unavailable'}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), layout='constrained')
    for axis, name in zip(axes.flat, chosen):
        index = int(np.flatnonzero(ids == name)[0])
        axis.plot(data['x'], truth[index, 0], color='black', lw=1.8)
        suffix = ' (prespecified evolution)' if name == sample else ''
        axis.set(xlim=(-16., 16.), xlabel='x', ylabel='u(x,0)', title=name+suffix)
        axis.grid(alpha=.2)
    for axis in list(axes.flat)[len(chosen):]:
        axis.set_visible(False)
    fig.suptitle('Ordinary two-Gaussian initial fields; selected by sample ID before prediction', fontsize=12)
    _save(fig, out/'figures'/'initial_conditions.png', outputs, pdf=True)
    plt.close(fig)
    return {'sample_ids': chosen, 'status': 'available', 'figure': 'figures/initial_conditions.png',
            'selection': 'Test IDs 0,1,2 plus the prespecified evolution example; no error-based selection.'}


def build_report(out: Path) -> list[Path]:
    out = Path(out).resolve()
    config = _read(out/'config.json', {})
    smoke = config.get('smoke_only') is True
    reference_value = config.get('reference_run')
    if not reference_value or not Path(reference_value).is_absolute():
        raise ValueError('config["reference_run"] must name the absolute controlled-IC shared-gamma run')
    reference_run = Path(reference_value).resolve()
    if reference_run == out:
        raise ValueError('Reference and new run must be separate')
    data, reference_data = _load_data(out), _load_data(reference_run)
    _grid(data)
    _grid(reference_data)
    # The spatial/time protocol is matched. IC fields and parameter arrays
    # intentionally come from different distributions and are not asserted equal.
    for key in ('x', 't'):
        np.testing.assert_array_equal(data[key], reference_data[key])
    records, omitted = _collect(out, data)
    reference, reference_omitted = _collect(reference_run, reference_data)
    selection = _read(out/'shared_gamma_selection.json', {})
    reference_selection = _read(reference_run/'shared_gamma_selection.json', {})
    gamma = _check_shared_gamma(records, selection, out.name)
    reference_gamma = _check_shared_gamma(reference, reference_selection, reference_run.name)
    audit = _protocol_audit(records, reference, smoke)
    data_audit = _read(out/'data_audit.json', {})
    shape_example = data_audit.get('initial_shape_audit', {}).get('selected_test_example', {})
    sample = (data_audit.get('prespecified_example_sample_id') or
              data_audit.get('representative_sample_id') or
              shape_example.get('sample_id') or 'test_000')
    if not isinstance(sample, str):
        raise ValueError('prespecified_example_sample_id must be a string')
    number = config.get('seeds', config.get('arguments', {}).get('seeds', 4))
    expected = list(range(number)) if isinstance(number, int) else [int(s) for s in number]
    status = ('pending_gamma_selection' if gamma is None else
              'pending_final_predictions' if not records else
              'complete' if set(expected).issubset(records) else 'partial')
    summary = {
        'status': status, 'smoke_only': smoke, 'formal_results': not smoke,
        'run': str(out), 'reference_run': str(reference_run),
        'source_dataset_run': config.get('source_dataset_run'),
        'expected_seeds': expected, 'omitted_records': omitted,
        'metric': {
            'formula': 'sqrt(sum_test_IC,x((prediction(t)-truth(t))**2) / sum_test_IC,x(truth(t)**2))',
            'units_in_json': 'fraction; multiply by 100 for percent',
            'time_aggregation': 'Arithmetic mean of per-time pooled errors over every saved time in each interval.',
            'observed_window': '[0,1.25] includes all saved times, not only the six labeled frames.',
            'std_ddof': 0, 'domain': [-16., 16.], 'nx': len(data['x']), 'final_time': 10.,
        },
        'shared_gamma_selection': {'selected_gamma': gamma, 'file': 'shared_gamma_selection.json',
                                   'candidates': selection.get('candidates'), 'scores': selection.get('scores')},
        'ordinary_two_gaussian': _aggregate(records),
        'controlled_distribution_reference': {
            **_aggregate(reference), 'selected_gamma': reference_gamma,
            'omitted_records': reference_omitted,
            'interpretation': 'Different IC distribution. Descriptive context only; not a matched-data baseline or paired accuracy comparison.',
        },
        'final_protocol_matching': audit,
        'scope': [
            'This run trains shared gamma dictionary + SSV + LDNet + GF on ordinary two-Gaussian initial conditions.',
            'Continuous IC parameters originate from the older ordinary dataset, resampled on [-16,16] through t=10.',
            'One gamma is selected using early validation data and frozen for all ICs during fresh final LDNet+GF training.',
            'The controlled-IC shared-gamma run is a descriptive reference from a different IC distribution.',
            'Equal optimizer seeds verify initial weights and protocol; they do not pair distinct initial conditions for error statistics.',
            'Each distribution is summarized independently over its own completed seeds; no paired differences or significance claims.',
            'No bare-LDNet arm has been added for this dataset. Older t=6 bare-LDNet results are not same-setting baselines.',
        ],
    }
    outputs = []
    (out/'figures').mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _plot_curves(out, data, records, reference, outputs, plt)
    summary['prespecified_example'] = _plot_example(out, data, records, sample, outputs, plt)
    summary['initial_condition_examples'] = _plot_initials(out, data, sample, outputs, plt)
    summary['curves'] = {'t': data['t'].tolist(),
                         'ordinary': {str(s): records[s]['curve'].tolist() for s in sorted(records)},
                         'controlled_reference': {str(s): reference[s]['curve'].tolist() for s in sorted(reference)}}
    summary['figures'] = [str(path.relative_to(out)).replace('\\', '/') for path in outputs]
    lines = [
        'Ordinary two-Gaussian ICs: shared gamma dictionary + SSV + LDNet + GF',
        ('SMOKE TEST ONLY: shortened training; not formal accuracy results.' if smoke else
         'Production experiment: four intended optimizer seeds, six observation frames, domain [-16,16], t=10.'),
        f'Status: {status}; completed new seeds: {sorted(records)}; expected: {expected}',
        f'Shared selected gamma: {gamma}',
        f'Continuous-parameter source: {config.get("source_dataset_run")}',
        'Original-paper full-domain pooled current-time RelRMSE, recomputed from saved predictions.',
        'Percentages below are mean +/- population seed SD (ddof=0). One-seed SD=0 is not evidence of reproducibility.',
        'Observed averages all saved times [0,1.25], not only the six supervised frames.',
        '', 'NEW ORDINARY TWO-GAUSSIAN RESULTS',
    ]
    for key, label in zip(METRICS, LABELS):
        lines.append(f'{label:<30} {_percent(summary["ordinary_two_gaussian"]["metrics"][key])}')
    lines.extend(['', 'DESCRIPTIVE REFERENCE ONLY: DIFFERENT INITIAL-CONDITION DISTRIBUTION',
                  f'Controlled-IC shared-gamma run: {reference_run}',
                  f'Completed reference seeds: {sorted(reference)}; selected gamma: {reference_gamma}',
                  'These figures are not matched-data baselines; no paired error difference or significance is calculated.'])
    for key, label in zip(METRICS, LABELS):
        lines.append(f'{label:<30} {_percent(summary["controlled_distribution_reference"]["metrics"][key])}')
    lines.extend(['', 'INTERPRETATION']+summary['scope'])
    lines.extend(['', 'Prespecified example: '+json.dumps(summary['prespecified_example']),
                  'Final protocol audit: '+audit['status'], '', 'Generated figures:']+summary['figures'])
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
