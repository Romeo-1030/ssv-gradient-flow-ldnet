"""Inference-only latent-freezing counterfactual for ordinary SSV models.

The learned state at the observed endpoint t=1.25 is held constant thereafter.
The learned encoder/decoder, external gamma, and physical SSV pullback remain
unchanged. This is not an independently trained no-dynamics baseline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ic_gamma_domain16_model import Domain16GammaModel, pack_inputs, validate_grid
from retry_json import save_json


METRICS = ('observed', 'long2to6', 't6', 'long2to10', 'extrapolation', 't10')
FREEZE_TIME = 1.25


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metrics(prediction, truth, times):
    error = np.square(prediction.astype(np.float64)-truth.astype(np.float64))
    energy = np.square(truth.astype(np.float64))
    curve = 100*np.sqrt(error.sum((0, 2))/energy.sum((0, 2)))
    per_ic = 100*np.sqrt(error.sum(-1)/energy.sum(-1))
    six = int(np.flatnonzero(times == 6.)[0])
    ten = int(np.flatnonzero(times == 10.)[0])
    values = {'observed': float(curve[times <= FREEZE_TIME].mean()),
        'long2to6': float(curve[(times > 2.) & (times <= 6.)].mean()),
        't6': float(curve[six]), 'long2to10': float(curve[times > 2.].mean()),
        'extrapolation': float(curve[times > FREEZE_TIME].mean()),
        't10': float(curve[ten])}
    return values, curve, per_ic


def _resolve_gamma(row, selection):
    value = row.get('shared_gamma', selection.get('selected_gamma'))
    if value is None or np.asarray(value).ndim != 0:
        raise ValueError('Frozen diagnostic requires a recorded scalar shared gamma')
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Shared gamma must be finite and positive')
    return value


def _plot(summary, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    times = np.asarray(summary['time_grid'])
    colors = {'ssv': '#1769aa', 'ssv_gf': '#cc5b20'}
    names = {'ssv': 'SSV', 'ssv_gf': 'SSV + GF'}
    for method, group in summary['summary'].items():
        for variant, style in [('normal', '-'), ('frozen', '--')]:
            mean = np.asarray(group[variant]['curve_mean_percent'])
            std = np.asarray(group[variant]['curve_std_ddof0_percent'])
            label = names.get(method, method)+(' normal' if variant == 'normal' else ' frozen at 1.25')
            ax.plot(times, mean, style, color=colors.get(method), lw=2., label=f'{label} (n={group["n"]})')
            ax.fill_between(times, np.maximum(mean-std, 0), mean+std, color=colors.get(method), alpha=.1)
    ax.axvspan(0, FREEZE_TIME, color='#dddddd', alpha=.45, label='Observed window')
    ax.axvline(FREEZE_TIME, color='#777777', lw=1.)
    ax.set(xlabel='Physical time t', ylabel='Pooled current-time RelRMSE (%)', xlim=(0, 10),
           title='Freeze latent evolution; retain the same SSV scaling')
    if summary['summary']:
        ax.legend(fontsize=8.5, loc='best')
    else:
        ax.text(.5, .5, 'No completed SSV checkpoints available', transform=ax.transAxes, ha='center')
    ax.grid(alpha=.2)
    fig.text(.5, .015, 'Inference counterfactual with unchanged trained weights; shading is seed SD, not a confidence interval.',
             ha='center', fontsize=8.)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(destination/'error_curves.png', dpi=170)
    fig.savefig(destination/'error_curves.pdf')
    plt.close(fig)


def build_diagnostics(out, methods=('ssv', 'ssv_gf'), device='cpu'):
    """Write partial-safe diagnostics for completed requested SSV treatments.

    Reads data.npz, results.json, shared_gamma_selection.json and checkpoints.
    No optimizer is constructed and no module is put in training mode. Future
    truth is used only by the metric calculation, never to construct forecasts.
    Only CPU is accepted so this report cannot compete with GPU training.
    """
    if torch.device(device).type != 'cpu':
        raise ValueError('This inference-only diagnostic intentionally runs on CPU')
    methods = tuple(methods)
    if len(set(methods)) != len(methods) or set(methods)-{'ssv', 'ssv_gf'}:
        raise ValueError('Requested methods must be distinct entries from ssv and ssv_gf')
    out = Path(out).resolve()
    destination = out/'diagnostics'/'frozen_latent'
    destination.mkdir(parents=True, exist_ok=True)
    data_path, result_path = out/'data.npz', out/'results.json'
    with np.load(data_path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    rows = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else []
    if not isinstance(rows, list):
        raise ValueError('results.json must contain a list of completed fit records')
    selection_path = out/'shared_gamma_selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.exists() else {}
    selected = [row for row in rows if row['method'] in methods]
    keys = [(row['method'], row['seed']) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate method/seed records cannot enter this diagnostic')
    selected.sort(key=lambda row: (methods.index(row['method']), row['seed']))
    x_numpy, times = data['x'], data['t']
    x, t = torch.as_tensor(x_numpy, dtype=torch.float32), torch.as_tensor(times, dtype=torch.float32)
    validate_grid(x, t)
    test_indices = np.flatnonzero(data['split'] == 'test')
    if not len(test_indices):
        raise ValueError('No test trajectories found')
    sample_ids = data['sample_id'][test_indices]
    initial = torch.as_tensor(data['u'][test_indices, 0], dtype=torch.float32)
    truth = data['u'][test_indices]
    reference_index = int(np.flatnonzero(times == FREEZE_TIME)[0])
    summary = {'training_performed': False, 'optimization_performed': False, 'device': 'cpu',
        'frozen_latent_time': FREEZE_TIME, 'gamma_changed': False, 'weights_changed': False,
        'centroid_variant_included': False, 'std_ddof': 0, 'units': 'percent',
        'metric': 'Original-paper full-domain pooled-current RelRMSE; pool squared errors over test ICs and space before square root',
        'window_definitions': {'observed': '[0,1.25]', 'long2to6': '(2,6]', 't6': 't=6',
            'long2to10': '(2,10]', 'extrapolation': '(1.25,10]', 't10': 't=10'},
        'counterfactual_formula': 'For t>1.25: u_frozen(x,t)=D(z(1.25), x/(1+gamma*t)^alpha, p)/(1+gamma*t)^beta. For t<=1.25, use normal prediction.',
        'interpretation': 'Prediction counterfactual with the same trained weights, not an independently trained no-latent or no-GF ablation. It measures the effect of continuing learned latent evolution beyond the observed endpoint in these checkpoints.',
        'prediction_inputs': 'initial field, fixed selected gamma, learned parameters and x/t coordinates only',
        'future_truth_used_for_predictions': False, 'future_truth_used_for_evaluation': True,
        'time_grid': times.tolist(), 'sample_ids': sample_ids.tolist(), 'requested_methods': list(methods),
        'source_data_sha256': _sha(data_path), 'source_results_sha256': _sha(result_path) if result_path.exists() else None,
        'runs': {}, 'summary': {}, 'skipped': [], 'checkpoint_hashes_unchanged': {}}
    arrays = {'x': x_numpy, 't': times, 'truth': truth, 'sample_ids': sample_ids}
    hashes = {}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    start = time.perf_counter()
    try:
        for row in selected:
            method, seed = row['method'], int(row['seed'])
            checkpoint_path = out/row.get('checkpoint', f'{method}_{seed}.pt')
            if not checkpoint_path.is_file():
                summary['skipped'].append({'method': method, 'seed': seed, 'reason': 'completed row has no checkpoint file'})
                continue
            if row.get('physical', False):
                raise ValueError('A requested SSV row describes a physical-coordinate model')
            gamma = _resolve_gamma(row, selection)
            hashes[checkpoint_path] = _sha(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            # Construction must not perturb a caller's training/random state.
            with torch.random.fork_rng(devices=[]):
                model = Domain16GammaModel(physical=False)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            inputs = pack_inputs(initial, np.full(len(initial), gamma))
            with torch.inference_mode():
                normal = model(inputs, t, x)
                conditions = model.encode(inputs)
                expert = model.experts[0]
                observed_states, conditions = expert.latent_rollout(conditions, t[:reference_index+1])
                zref = observed_states[:, -1]
                recovered = expert.decode_physical(zref, conditions, t[reference_index], x[None].expand(len(initial), -1))
                torch.testing.assert_close(recovered, normal[:, reference_index], rtol=1e-4, atol=2e-6)
                frozen = normal.clone()
                for j in range(reference_index+1, len(t)):
                    frozen[:, j] = expert.decode_physical(zref, conditions, t[j], x[None].expand(len(initial), -1))
                torch.testing.assert_close(frozen[:, :reference_index+1], normal[:, :reference_index+1], rtol=0, atol=0)
                if not torch.isfinite(normal).all() or not torch.isfinite(frozen).all():
                    raise ValueError('Nonfinite inference in frozen-latent diagnostic')
            normal_numpy, frozen_numpy = normal.numpy(), frozen.numpy()
            record = {'seed': seed, 'shared_gamma': gamma, 'checkpoint': str(checkpoint_path),
                      'observed_predictions_identical': True, 'freeze_anchor_recovery_passed': True, 'metrics': {}}
            saved_path = out/row.get('prediction', f'{method}_{seed}_test.npz')
            if saved_path.is_file():
                with np.load(saved_path, allow_pickle=False) as saved:
                    saved_prediction = saved['pred'] if 'pred' in saved else saved['prediction']
                    if 'sample_ids' in saved:
                        np.testing.assert_array_equal(saved['sample_ids'], sample_ids)
                    np.testing.assert_allclose(normal_numpy, saved_prediction, rtol=2e-4, atol=5e-6)
                    record['maximum_absolute_replay_difference'] = float(np.max(abs(normal_numpy-saved_prediction)))
                    record['saved_prediction_replay_passed'] = True
            for variant, prediction in [('normal', normal_numpy), ('frozen', frozen_numpy)]:
                values, curve, per_ic = _metrics(prediction, truth, times)
                record['metrics'][variant] = values | {'curve_percent': curve.tolist()}
                arrays[f'{method}_seed{seed}_{variant}'] = prediction
                arrays[f'{method}_seed{seed}_{variant}_per_ic_error_percent'] = per_ic
            arrays[f'{method}_seed{seed}_gamma'] = np.full(len(initial), gamma)
            arrays[f'{method}_seed{seed}_frozen_state'] = zref.numpy()
            summary['runs'].setdefault(method, []).append(record)
            del model, checkpoint, normal, frozen, observed_states, recovered
    finally:
        torch.set_num_threads(previous_threads)
    for method, records in summary['runs'].items():
        group = {'n': len(records), 'seeds': [r['seed'] for r in records], 'normal': {}, 'frozen': {}}
        for variant in ('normal', 'frozen'):
            for metric in METRICS:
                values = np.array([r['metrics'][variant][metric] for r in records])
                group[variant][metric] = {'mean': float(values.mean()), 'std_ddof0': float(values.std()), 'values': values.tolist()}
            curves = np.array([r['metrics'][variant]['curve_percent'] for r in records])
            group[variant]['curve_mean_percent'] = curves.mean(0).tolist()
            group[variant]['curve_std_ddof0_percent'] = curves.std(0).tolist()
        summary['summary'][method] = group
    for path, digest in hashes.items():
        if _sha(path) != digest:
            raise RuntimeError('A checkpoint changed while diagnostics ran')
        summary['checkpoint_hashes_unchanged'][str(path)] = digest
    summary['all_model_checkpoints_unchanged'] = True
    summary['unavailable_methods'] = [method for method in methods if method not in summary['runs']]
    summary['completed_fits'] = sum(len(v) for v in summary['runs'].values())
    summary['seconds'] = time.perf_counter()-start
    summary['source_data_unchanged'] = _sha(data_path) == summary['source_data_sha256']
    summary['source_results_unchanged_during_diagnostic'] = (_sha(result_path) if result_path.exists() else None) == summary['source_results_sha256']
    np.savez_compressed(destination/'predictions.npz', **arrays)
    _plot(summary, destination)
    save_json(destination/'summary.json', summary)
    return summary
