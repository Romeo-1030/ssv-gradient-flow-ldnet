"""Unattended pure latent versus pretrained mass-variance SSV plus original GF.

Same two-Gaussian data, original-size backbone and optimizer as global_baselines.
The global IC-to-gamma predictor is calibrated once, then completely frozen.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import platform
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

from experiment import ROOT, metrics
from future_consistency import evaluate, observed_validation
from gamma_gf import make_energy
from gamma_gf_model import raw_gf_terms
from global_baselines import GlobalBaseline, supervised_loss, train_one as train_latent
from moment_gamma import MomentGammaGFModel, MomentGammaPredictor, backbone_hash, calibrate, variance_targets
from moving_charts import state_hash
from original_scale import OBS, cpu_state, save_json, sync, make_gf_indices
from original_scale_model import ORIGINAL_SOURCE
from two_gaussian_data import make_data, fingerprint, boundary_report, PROTOCOL

METHODS = ('latent', 'moment_ssv_gf')
NAMES = {'latent': 'Latent (physical time)', 'moment_ssv_gf': 'Latent + variance-pretrained global SSV + GF'}


def gf_weight(step):
    return 1e-4*min((step-100)/100, 1.) if step > 100 else 0.


def train_ssv(args, seed, gamma_state, ic, observed, valid_ic, valid_observed, x, t, out):
    device = x.device
    torch.manual_seed(seed)
    model = MomentGammaGFModel(gamma_state)
    initial_hash = backbone_hash(model)
    energy = make_energy(seed)
    energy_hash = state_hash(energy.state_dict())
    gamma_hash = state_hash(model.ssv.state_dict())
    model, energy = model.to(device), energy.to(device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    backbone = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in backbone) == 91073
    assert sum(p.numel() for p in model.ssv.parameters()) == 3105
    parameters = backbone + list(energy.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=1e-5)
    scales = model.get_scales().clone()
    with torch.no_grad():
        gamma_before = model.gamma_values(ic).clone()
    gf_indices = make_gf_indices(t, 32)
    history = []
    best, selected, best_state, best_energy, gf_scale = float('inf'), None, None, None, None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()
    print('START moment_ssv_gf', seed, 'trainable_parameters', sum(p.numel() for p in parameters), flush=True)
    for step in range(1, args.steps+1):
        model.train()
        data_value = supervised_loss(model, ic, observed, x, t)
        raw = raw_gf_terms(model, energy, ic, t, gf_indices, 24).mean()
        if step == 101:
            gf_scale = max(float(raw.detach()), 1e-12)
        weight = gf_weight(step)
        normalized = raw/gf_scale if weight else raw.new_zeros(())
        objective = data_value + weight*normalized
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(backbone, 5.)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                val = observed_validation(model, valid_ic, valid_observed, x, t)
            row = {'step': step, 'train': float(objective.detach()), 'data': float(data_value.detach()),
                   'raw_gf': float(raw.detach()), 'normalized_gf': float(normalized.detach()),
                   'gf_weight': weight, 'valid': float(val)}
            if not all(np.isfinite(v) for v in row.values()):
                raise RuntimeError(f'Nonfinite training: seed {seed}, step {step}')
            history.append(row)
            if row['valid'] < best:
                best, selected, best_state, best_energy = row['valid'], step, cpu_state(model), cpu_state(energy)
        if step == 1 or step % 250 == 0 or step == args.steps:
            sync(device)
            progress = {'status': 'training', 'pid': os.getpid(), 'method': 'moment_ssv_gf', 'seed': seed,
                        'step': step, 'steps_per_run': args.steps, 'seconds': time.perf_counter()-start,
                        'best_early_validation': best}
            save_json(out/'progress.json', progress)
            print('TRAIN moment_ssv_gf', seed, step, 'early_valid', best, 'seconds', round(progress['seconds'], 1), flush=True)
    sync(device)
    seconds = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    # Check the terminal state as well as the selected checkpoint for freezing.
    assert state_hash(model.ssv.state_dict()) == gamma_hash
    model.load_state_dict(best_state)
    model.eval()
    assert state_hash(model.ssv.state_dict()) == gamma_hash
    torch.testing.assert_close(model.get_scales(), scales, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(model.gamma_values(ic), gamma_before, rtol=0, atol=0)
    record = {'method': 'moment_ssv_gf', 'seed': seed, 'steps': args.steps, 'selected_step': selected, 'history': history,
              'initial_parameters_sha256': initial_hash, 'initial_energy_sha256': energy_hash,
              'parameters': sum(p.numel() for p in parameters), 'backbone_parameters': 91073,
              'energy_parameters': 4417, 'frozen_ssv_parameters': 3105,
              'gamma_state_sha256': gamma_hash, 'gamma_frozen': True, 'gf_scale': gf_scale,
              'gf_evaluations': args.steps, 'scales': scales.cpu().tolist(), 'scales_unchanged': True,
              'seconds': seconds, 'peak_gpu_memory_bytes': peak}
    checkpoint = {'model': best_state, 'energy': best_energy, 'method': 'moment_ssv_gf', 'seed': seed}
    return model, checkpoint, record


def finish(args, out, records, data, x, t, gamma_checkpoint):
    expected = {(m, s) for m in METHODS for s in range(args.seeds)}
    assert {(r['method'], r['seed']) for r in records} == expected and len(records) == len(expected)
    ids = np.flatnonzero(data['split'] == 'test')
    valid_ids = np.flatnonzero(data['split'] == 'valid')
    ic = torch.tensor(data['u'][:, 0], device=x.device)
    for seed in range(args.seeds):
        assert len({r['initial_parameters_sha256'] for r in records if r['seed'] == seed}) == 1
    for record in records:
        method, seed = record['method'], record['seed']
        assert record['steps'] == args.steps and record['scales_unchanged']
        selected = min(record['history'], key=lambda h: h['valid'])
        assert selected['step'] == record['selected_step']
        for h in record['history']:
            weight = gf_weight(h['step']) if method == 'moment_ssv_gf' else 0.
            assert h['gf_weight'] == weight
            np.testing.assert_allclose(h['train'], h['data']+weight*h['normalized_gf'], rtol=2e-6, atol=1e-8)
        checkpoint = torch.load(out/f'{method}_{seed}.pt', map_location='cpu', weights_only=True)
        model = GlobalBaseline('latent') if method == 'latent' else MomentGammaGFModel(gamma_checkpoint['state'])
        model = model.to(x.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        assert (checkpoint['energy'] is not None) == (method == 'moment_ssv_gf')
        if method == 'moment_ssv_gf':
            assert record['gamma_frozen'] and state_hash(model.ssv.state_dict()) == gamma_checkpoint['state_sha256']
        with torch.no_grad():
            pred = model(ic[ids], t, x)
            val = observed_validation(model, ic[valid_ids], torch.tensor(data['u'][valid_ids][:, list(OBS)], device=x.device), x, t)
        np.testing.assert_allclose(float(val), selected['valid'], rtol=0, atol=1e-8)
        with np.load(out/f'{method}_{seed}_test.npz') as saved:
            np.testing.assert_array_equal(ids, saved['ids'])
            np.testing.assert_array_equal(pred.cpu().numpy(), saved['prediction'])
            recalculated = metrics(torch.tensor(saved['prediction']), torch.tensor(data['u'][ids]), x.cpu(), t.cpu())
        for key in ('curve', 'observed', 'short', 'long', 'final'):
            np.testing.assert_allclose(recalculated[key], record['test'][key], rtol=0, atol=1e-12)
    summary = {}
    for method in METHODS:
        rows = sorted([r for r in records if r['method'] == method], key=lambda r: r['seed'])
        item = {}
        for key in ('initial', 'label_times_error', 'observed', 'short', 'long', 'final'):
            values = np.asarray([r['test'][key] for r in rows])
            item[key] = {'mean': float(values.mean()), 'sample_sd': float(values.std(ddof=1)) if len(values)>1 else 0., 'values': values.tolist()}
        item['training_seconds'] = [r['seconds'] for r in rows]
        summary[method] = item
    delta = np.asarray(summary['moment_ssv_gf']['long']['values'])-summary['latent']['long']['values']
    summary['paired_comparison'] = {'long_error_differences': delta.tolist(), 'improved_seeds': int((delta<0).sum())}
    # Post-hoc gamma diagnostics; held-out frames were unavailable to calibration.
    predictor = MomentGammaPredictor().to(x.device)
    predictor.load_state_dict(gamma_checkpoint['state'])
    gamma_report = {'posthoc_only': True, 'test_uses_only_u0_for_gamma_prediction': True}
    with torch.no_grad():
        for split in ('train', 'valid', 'test'):
            selected_ids = np.flatnonzero(data['split'] == split)
            gamma = predictor(ic[selected_ids])
            targets = variance_targets(torch.tensor(data['u'][selected_ids][:, list(OBS)], device=x.device), x, t[list(OBS)])
            residual = torch.log1p(gamma*t[list(OBS)][None])-targets
            gamma_report[split] = {'ids': selected_ids.tolist(), 'gamma': gamma[:, 0].cpu().tolist(),
                'mean': float(gamma.mean()), 'min': float(gamma.min()), 'max': float(gamma.max()),
                'early_log_variance_mse': float(residual.square().mean())}
    save_json(out/'gamma_diagnostics.json', gamma_report)
    save_json(out/'summary.json', summary)
    save_json(out/'audit.json', {'passed': True, 'runs': len(records), 'updates': args.steps*len(records),
        'paired_initial_parameters': True, 'early_only_checkpoint_selection': True,
        'pretrained_gamma_frozen': True, 'fixed_exponents': True,
        'same_device_checkpoint_predictions_bitwise_equal': True, 'metrics_recomputed': True})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for method, color in zip(METHODS, ('#777777', '#D65A31')):
        curves = 100*np.asarray([r['test']['curve'] for r in records if r['method'] == method])
        for curve in curves:
            ax.plot(data['t'], curve, color=color, alpha=.2, lw=.6)
        ax.plot(data['t'], curves.mean(0), color=color, label=NAMES[method], lw=1.8)
    ax.axvline(.5, color='gray', ls='--', lw=1)
    ax.axhline(100, color='gray', ls=':', lw=1)
    ax.set(xlabel='Physical time t', ylabel='Test relative L2 error (%)', yscale='log', title='Two-Gaussian ICs: individual seeds and mean')
    ax.legend(fontsize=8)
    fig.savefig(out/'errors.png', dpi=170)
    fig.savefig(out/'errors.pdf')
    plt.close(fig)
    lines = ['# Mass-variance SSV + GF versus pure latent', '',
        f'Completed {len(records)} runs, {args.steps} updates each, {args.seeds} paired backbone seeds.', '',
        'Identical two-Gaussian data and 36/12/24 split as the preceding experiments. Labels only at t=0,.25,.5; evaluation through t=6. Inputs are 192 samples of u0, without component parameters.', '',
        'SSV pretraining uses v(t)=integral((x-centroid)^2*u)/integral(u), with mass weights u (not u squared). Minimize mean[(log(v(t)/v(0))-log(1+gamma(u0)*t))^2]. No amplitude matching, oracle gamma, diffusivity or future labels enter this loss. The 192-16-Tanh-1-softplus predictor starts at gamma=1 and is selected using early validation across three restarts. One calibration is shared across all backbone seeds.', '',
        'Alpha=beta=1/2. The pretrained gamma network is independent of the LDNet encoder and frozen during backbone training. The global chart remains centered at x=0; variance uses a measured centroid but no new spatial centering is introduced. Test gamma is predicted from u0 alone.', '',
        'Both arms have the same original-size 91,073-parameter backbone and matching initial backbone weights. Pure latent uses physical time and unscaled x/u. SSV uses log(1+gamma*t), x/sqrt(1+gamma*t), and amplitude sqrt(1+gamma*t). GF adds 4,417 trainable energy parameters; SSV adds 3,105 frozen parameters.', '',
        'Full-batch AdamW: lr=.001, weight decay=1e-6, backbone clip=5, cosine decay to1e-5. Select by three-label validation at step1/every10/final. Original GF: 32 rollout times, 24 xi points, zero weight through100, detached normalization at101, ramp to1e-4 at200. No chart consistency. GF uses future time coordinates, never future labels.', '',
        'Errors are mean +/- sample SD across backbone seeds on 24 held-out ICs. The observed window includes five times, only three of which are labelled.', '',
        '| Method | Three labels (%) | Observed window (%) | (.5,2] (%) | (2,6] (%) | t=6 (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        cells = [f"{100*summary[method][k]['mean']:.2f} +/- {100*summary[method][k]['sample_sd']:.2f}" for k in ('label_times_error', 'observed', 'short', 'long', 'final')]
        lines.append('| '+NAMES[method]+' | '+' | '.join(cells)+' |')
    lines += ['', '![Error curves](errors.png)', '',
        f"SSV + GF improves long-time error in {int((delta<0).sum())}/{args.seeds} paired seeds. Total synchronized backbone training time: {sum(r['seconds'] for r in records)/60:.2f} minutes.", '',
        'This two-arm comparison measures the combined SSV+GF treatment; it does not isolate the incremental GF effect. Calibration sensitivity is not included in backbone seed SD. Automatic checkpoint replay, frozen-gamma checks and independent metric checks passed.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use .venv-cuda')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = ROOT/args.out
    if (out/'config.json').exists() and not args.resume:
        raise FileExistsError('Preserving existing experiment; choose another --out or --resume')
    out.mkdir(parents=True, exist_ok=True)
    data = make_data()
    with np.load(ROOT/'runs/two_gaussian_main_cuda/data.npz') as previous:
        for key in data:
            np.testing.assert_array_equal(data[key], previous[key])
    names = ('moment_baselines.py', 'moment_gamma.py', 'test_moment_gamma.py', 'start_moment_baselines.ps1',
             'global_baselines.py', 'original_scale_model.py', 'original_scale.py', 'gamma_gf.py', 'gamma_gf_model.py',
             'future_consistency.py', 'experiment.py', 'moving_charts.py', 'two_stage.py', 'two_gaussian_data.py')
    config = vars(args) | {'methods': list(METHODS), 'data_protocol': PROTOCOL, 'data_fingerprint': fingerprint(data),
        'boundary_check': boundary_report(data), 'alpha_beta_ssv': [.5, .5],
        'gamma_pretraining': {'predictor': '192-16-Tanh-1-softplus', 'initial': 1., 'optimizer': 'Adam',
            'lr': .003, 'clip': 5., 'seed_base': 9100, 'device': 'CPU', 'shared_across_backbone_seeds': True,
            'loss': 'mean[(log(v(t)/v(0))-log(1+gamma(u0)*t))^2], mass weights u',
            'selection': 'early variance validation, step0/1/every25/final, across restarts',
            'backbone_training': 'entire independent SSV predictor frozen'},
        'gf_weight': 1e-4, 'gf_times': 32, 'gf_collocation': 24, 'consistency_weight': 0.,
        'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': 1e-6, 'eta_min': 1e-5, 'clip_backbone': 5.},
        'labels': [0., .25, .5], 'selection': 'three-label validation, step1/every10/final; no future labels',
        'torch': str(torch.__version__), 'numpy': str(np.__version__), 'python': platform.python_version(),
        'cuda': torch.version.cuda, 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'deterministic': True, 'tf32': False,
        'source_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        'upstream_train_sha256': hashlib.sha256(ORIGINAL_SOURCE.read_bytes()).hexdigest()}
    if (out/'config.json').exists():
        old = json.loads((out/'config.json').read_text())
        for key in ('steps', 'seeds', 'gamma_steps', 'gamma_restarts', 'methods', 'data_fingerprint', 'source_sha256', 'upstream_train_sha256', 'torch', 'device_name'):
            if config[key] != old[key]:
                raise ValueError(f'Resume mismatch: {key}')
    else:
        save_json(out/'config.json', config)
        np.savez_compressed(out/'data.npz', **data)
        source = out/'source_snapshot'
        source.mkdir(exist_ok=True)
        for name in names:
            (source/name).write_bytes((ROOT/name).read_bytes())
        (source/'upstream_train.py').write_bytes(ORIGINAL_SOURCE.read_bytes())
    save_json(out/'progress.json', {'status': 'starting', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()})
    ix = {s: np.flatnonzero(data['split'] == s) for s in ('train', 'valid', 'test')}
    x, t = [torch.tensor(data[k], dtype=torch.float32) for k in ('x', 't')]
    ic = torch.tensor(data['u'][:, 0])
    observed = torch.tensor(data['u'][:, list(OBS)])
    if (out/'gamma_predictor.pt').exists():
        gamma_checkpoint = torch.load(out/'gamma_predictor.pt', map_location='cpu', weights_only=True)
    else:
        gamma_checkpoint = calibrate(args, ic[ix['train']], observed[ix['train']], ic[ix['valid']], observed[ix['valid']], x, t[list(OBS)], out)
    assert state_hash(gamma_checkpoint['state']) == gamma_checkpoint['state_sha256']
    x, t, ic, observed = [v.to(device) for v in (x, t, ic, observed)]
    records = json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
    completed = {(r['method'], r['seed']) for r in records}
    paired = {r['seed']: r['initial_parameters_sha256'] for r in records}
    for seed in range(args.seeds):
        for method in METHODS:
            if (method, seed) in completed:
                continue
            inputs = (ic[ix['train']], observed[ix['train']], ic[ix['valid']], observed[ix['valid']], x, t, out)
            if method == 'latent':
                model, checkpoint, record = train_latent(args, method, seed, *inputs)
            else:
                model, checkpoint, record = train_ssv(args, seed, gamma_checkpoint['state'], *inputs)
            if seed in paired:
                assert paired[seed] == record['initial_parameters_sha256']
            paired[seed] = record['initial_parameters_sha256']
            torch.save(checkpoint, out/f'{method}_{seed}.pt')
            # No held-out future fields are passed to fitting or selection.
            truth = torch.tensor(data['u'][ix['test']], device=device)
            record['test'] = evaluate(model, ic[ix['test']], truth, x, t, ix['test'], out, method, seed)
            del truth, model
            records.append(record)
            save_json(out/'results.json', records)
            print('DONE', method, seed, 'long', record['test']['long'], 'final', record['test']['final'], flush=True)
    save_json(out/'progress.json', {'status': 'verifying', 'pid': os.getpid(), 'runs': len(records)})
    finish(args, out, records, data, x, t, gamma_checkpoint)
    save_json(out/'progress.json', {'status': 'complete', 'runs': len(records),
        'optimizer_updates': args.steps*len(records), 'gamma_updates': args.gamma_steps*args.gamma_restarts,
        'completed_utc': datetime.now(timezone.utc).isoformat()})
    print('COMPLETE', len(records), 'runs', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--gamma-steps', type=int, default=1500)
    parser.add_argument('--gamma-restarts', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', default='runs/moment_baselines_cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.seeds, args.gamma_steps, args.gamma_restarts) < 1:
        parser.error('Positive steps, seeds and restarts required')
    try:
        run(args)
    except Exception as error:
        if not isinstance(error, FileExistsError):
            destination = ROOT/args.out
            destination.mkdir(parents=True, exist_ok=True)
            save_json(destination/'progress.json', {'status': 'failed', 'pid': os.getpid(), 'error': str(error)})
            (destination/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
