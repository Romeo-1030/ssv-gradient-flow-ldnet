"""CUDA comparison using the original-size conditioned SSV-LDNet backbone.

Non-Gaussian adaptation: the full IC is encoded into a shared condition vector.
Three treatments: global SSV, global SSV with the original learned GF penalty,
and three full-size fixed charts with early-frame physical overlap consistency.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiment import ROOT, dataset, loss, metrics
from moving_charts import overlap_loss, state_hash
from original_scale_model import OriginalScaleModel, EnergyDensity, gradient_rhs, ORIGINAL_SOURCE

METHODS = ('global_ssv', 'charts_overlap', 'global_gf')
OBS = (0, 2, 4)


def make_gf_indices(t, time_points=32):
    future = torch.linspace(.5, float(t[-1]), time_points - 2, device=t.device)[1:]
    target = torch.cat((t[list(OBS)], future))
    pick = torch.searchsorted(t, target).clamp_max(len(t) - 1).unique()
    if len(pick) != time_points:
        raise ValueError('GF time grid must contain the requested number of distinct states')
    return pick


def raw_gf_loss(model, energy, ic, t, gf_indices, collocation=24):
    """Reuse the released gradient_rhs; no future reference field is an input."""
    if model.charts != 1:
        raise ValueError('This comparison applies original GF only to the global baseline')
    expert = model.experts[0]
    p = model.encode(ic)
    z, p = expert.latent_rollout(p, t)
    selected = z[:, gf_indices]
    zz = selected.reshape(-1, model.latent)
    pp = p[:, None].expand(-1, len(gf_indices), -1).reshape(-1, model.condition_dim)
    xi = (2 * torch.rand(len(zz), collocation, device=ic.device) - 1) * 24.
    intrinsic, rhs = gradient_rhs(expert, energy, zz, pp, xi)
    return (intrinsic - rhs).square().mean()


def validation_loss(pred, truth, indices=OBS):
    """Original release: mean per-IC relative initial-norm error, not MSE."""
    error = (pred[:, list(indices)] - truth[:, list(indices)]).square().sum(-1)
    energy = truth[:, 0].square().sum(-1).clamp_min(1e-6)
    return (error / energy[:, None]).sqrt().mean()


def save_json(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def diagnostics(model, pred, parts, truth, x, t):
    w = model.weights(x).double()[:, None].expand(-1, len(t), -1)
    p, u, y = pred.double(), parts.double(), truth.double()
    initial_energy = y[:, 0].square().sum(-1).clamp_min(1e-12)
    disagreement = ((u - p[:, None]).square() * w[None]).sum((1, 3))
    mass0 = y[:, 0].sum(-1)
    mass = p.sum(-1)
    signal = y.square().sum(0)
    return {
        'overlap_rms_curve': (disagreement.div(initial_energy[:, None]).mean(0).sqrt()).tolist(),
        'mass_error_curve': ((mass - mass0[:, None]).abs() / mass0.abs().clamp_min(1e-12)[:, None]).mean(0).tolist(),
        'mass_drift_curve': ((mass - mass[:, :1]).abs() / mass0.abs().clamp_min(1e-12)[:, None]).mean(0).tolist(),
        'chart_energy_shares': ((w * signal[None]).sum(-1) / signal.sum(-1).clamp_min(1e-12)[None]).tolist(),
        'gate_motion_rms_curve': [0.] * len(t),
    }


def evaluate(model, y, ix, labels, x, t, out, method, seed, device):
    report = {}
    model.eval()
    with torch.no_grad():
        for split in ('test', 'ood'):
            ids = ix[split]
            sync(device)
            start = time.perf_counter()
            pred, parts = model(y[ids, 0], t, x, components=True)
            sync(device)
            prediction_seconds = time.perf_counter() - start
            if not torch.isfinite(pred).all():
                raise RuntimeError(f'Nonfinite prediction: {method}, {seed}, {split}')
            # Match prior reporting arithmetic on CPU float64.
            pc, uc, yc = pred.cpu(), parts.cpu(), y[ids].cpu()
            xc, tc = x.cpu(), t.cpu()
            record = metrics(pc, yc, xc, tc)
            record['initial'] = record['curve'][0]
            record['prediction_seconds'] = prediction_seconds
            record['diagnostics'] = diagnostics(model, pred, parts, y[ids], x, t)
            record['family_metrics'] = {}
            for family in np.unique(labels[ids]):
                keep = np.flatnonzero(labels[ids] == family)
                record['family_metrics'][str(family)] = metrics(pc[keep], yc[keep], xc, tc)
            report[split] = record
            np.savez_compressed(out / f'{method}_{seed}_{split}.npz', prediction=pc.numpy(), ids=ids)
    return report


def run(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this run; use the configured .venv-cuda Python')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, out = ROOT / args.scale_source, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists() and not args.resume:
        raise FileExistsError('Existing results are preserved; use --resume or a new --out')
    xx, tt, yy, labels, splits = dataset()
    with np.load(source / 'data.npz') as saved:
        for key, value in zip(('x', 't', 'u', 'family', 'split'), (xx, tt, yy, labels, splits)):
            np.testing.assert_array_equal(saved[key], value)
    if tt[0] != 0 or not np.all(np.diff(tt) > 0):
        raise ValueError('The shared time grid must start at zero and be increasing')
    np.savez_compressed(out / 'data.npz', x=xx, t=tt, u=yy, family=labels, split=splits)
    x, t, y = [torch.tensor(value, dtype=torch.float32, device=device) for value in (xx, tt, yy)]
    ix = {key: np.flatnonzero(splits == key) for key in ('train', 'valid', 'test', 'ood')}
    global_scales = torch.load(source / 'global_scales.pt', map_location='cpu', weights_only=True)
    chart_scales = torch.load(source / 'charts3_scales.pt', map_location='cpu', weights_only=True)
    gf_indices = make_gf_indices(t)
    config = vars(args) | {
        'model_description': 'Original-size conditional shifted Euler SSV-LDNet with full-IC encoder adaptation',
        'latent': 16, 'width': 128, 'depth': 3, 'condition_dim': 16, 'ic_samples': 192,
        'fourier_frequencies': 16, 'fourier_coefficients': 33, 'gamma': 1,
        'global_scales': global_scales.tolist(), 'chart_scales': chart_scales.tolist(),
        'scale_protocol': 'Reuse frozen three-frame non-Gaussian profile alignment; Gaussian gamma network is not used',
        'supervised_times': [0., .25, .5], 'overlap_times': [0., .25, .5],
        'gf_times': t[gf_indices].cpu().tolist(), 'gf_time_points': len(gf_indices), 'gf_collocation': 24,
        'gf_sampling_coordinates': 'fixed transformed xi window [-24,24]',
        'gf_schedule': 'zero through step100; step101 detached raw normalization; linear ramp to lambda_gf at step200',
        'checkpoint_selection': 'minimum mean per-trajectory initial-L2-normalized error at t=0,.25,.5; every 10 steps',
        'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': 1e-6, 'clip_norm': 5,
                      'clip_scope': 'backbone only, matching original GF training',
                      'scheduler': 'CosineAnnealingLR', 'eta_min': 1e-5},
        'parameter_matching': False,
        'split_counts': {key: len(value) for key, value in ix.items()},
        'domain': [-24, 24], 'boundary': 'periodic reference; learned Fourier/POU predictions not constrained to exact periodicity',
        'torch': torch.__version__, 'python': platform.python_version(), 'cuda_runtime': torch.version.cuda,
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'device_memory_bytes': torch.cuda.get_device_properties(device).total_memory if device.type == 'cuda' else None,
        'threads': 1, 'tf32': False, 'deterministic_algorithms': True, 'synchronized_gpu_timing': True,
        'scale_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                         for name in ('global_scales.pt', 'charts3_scales.pt')},
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ('original_scale.py', 'original_scale_model.py', 'experiment.py', 'moving_charts.py')},
        'original_train_sha256': hashlib.sha256(ORIGINAL_SOURCE.read_bytes()).hexdigest(),
    }
    if args.resume and (out / 'config.json').exists():
        old = json.loads((out / 'config.json').read_text())
        for key in ('steps', 'seeds', 'methods', 'overlap_lambda', 'lambda_gf', 'scale_sha256', 'source_sha256', 'torch', 'device_name'):
            if old[key] != config[key]:
                raise ValueError(f'Resume protocol mismatch: {key}')
    else:
        save_json(out / 'config.json', config)
        snapshot = out / 'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for name in config['source_sha256']:
            (snapshot / name).write_bytes((ROOT / name).read_bytes())
        (snapshot / 'upstream_train.py').write_bytes(ORIGINAL_SOURCE.read_bytes())
    reports = json.loads((out / 'results.json').read_text()) if args.resume and (out / 'results.json').exists() else []
    completed = {(r['method'], r['seed']) for r in reports}
    for seed in range(args.seeds):
        for method in args.methods:
            if (method, seed) in completed:
                continue
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
            charts = 3 if method == 'charts_overlap' else 1
            model = OriginalScaleModel(charts=charts)
            model.set_scales(chart_scales if charts == 3 else global_scales)
            init_hash = state_hash(model.state_dict())
            encoder_hash = state_hash(model.encoder.state_dict())
            model = model.to(device)
            energy = EnergyDensity().to(device)
            use_gf = method == 'global_gf'
            variables = list(model.parameters()) + (list(energy.parameters()) if use_gf else [])
            opt = torch.optim.AdamW(variables, lr=.001, weight_decay=1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
            scales_before = model.get_scales().clone()
            ic = y[ix['train'], 0]
            weights = model.weights(x)[:, None].expand(-1, 5, -1).detach()
            best, best_state, best_energy, selected_step = float('inf'), None, None, None
            history, gf_scale = [], None
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            sync(device)
            start = time.perf_counter()
            print('START', method, seed, 'parameters', sum(p.numel() for p in variables), 'device', device, flush=True)
            for step in range(1, args.steps + 1):
                model.train()
                pred, parts = model(ic, t[:5], x, components=True)
                data_value = loss(pred, y[ix['train'], :5], list(OBS))
                ov = overlap_loss(parts, weights, ic) if charts == 3 else data_value.new_zeros(())
                raw = raw_gf_loss(model, energy, ic, t, gf_indices) if use_gf else data_value.new_zeros(())
                if use_gf and step == 101:
                    gf_scale = max(float(raw.detach()), 1e-12)
                gf_weight = 0. if not use_gf or step <= 100 else args.lambda_gf * min((step - 100) / 100, 1.)
                normalized = raw / gf_scale if gf_weight else data_value.new_zeros(())
                objective = data_value + (args.overlap_lambda * ov if charts == 3 else 0.) + gf_weight * normalized
                opt.zero_grad()
                objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                opt.step()
                scheduler.step()
                if step == 1 or step % 10 == 0 or step == args.steps:
                    model.eval()
                    with torch.no_grad():
                        val = validation_loss(model(y[ix['valid'], 0], t[:5], x), y[ix['valid'], :5])
                    values = torch.stack((objective.detach(), data_value.detach(), ov.detach(), raw.detach(), normalized.detach(), val)).cpu().tolist()
                    if not all(np.isfinite(values)):
                        raise RuntimeError(f'Nonfinite optimization: {method}, seed{seed}, step{step}')
                    valid = values[-1]
                    history.append(dict(zip(('train', 'data', 'overlap', 'raw_gf', 'normalized_gf', 'valid'), values)) | {'step': step, 'gf_weight': gf_weight})
                    if valid < best:
                        best, selected_step = valid, step
                        best_state = cpu_state(model)
                        best_energy = cpu_state(energy) if use_gf else None
                if step % 500 == 0:
                    sync(device)
                    elapsed = time.perf_counter() - start
                    progress = {'method': method, 'seed': seed, 'step': step, 'elapsed_seconds': elapsed, 'best_early_valid': best}
                    save_json(out / 'progress.json', progress)
                    print('TRAIN', method, seed, step, 'early_valid', valid, 'seconds', round(elapsed, 1), flush=True)
            sync(device)
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
            model.load_state_dict(best_state)
            torch.testing.assert_close(model.get_scales(), scales_before, rtol=0, atol=0)
            record = {
                'method': method, 'seed': seed, 'charts': charts, 'width': 128,
                'parameters': sum(p.numel() for p in variables), 'backbone_parameters': sum(p.numel() for p in model.parameters()),
                'energy_parameters': sum(p.numel() for p in energy.parameters()) if use_gf else 0,
                'steps': args.steps, 'selected_step': selected_step, 'seconds': seconds, 'peak_gpu_memory_bytes': peak,
                'initial_state_sha256': init_hash, 'initial_encoder_sha256': encoder_hash,
                'scales': scales_before.cpu().tolist(), 'scales_unchanged': True,
                'overlap_lambda': args.overlap_lambda if charts == 3 else 0.,
                'lambda_gf': args.lambda_gf if use_gf else 0., 'gf_scale': gf_scale,
                'history': history,
            }
            record.update(evaluate(model, y, ix, labels, x, t, out, method, seed, device))
            torch.save({'model': best_state, 'energy': best_energy, 'method': method, 'seed': seed}, out / f'{method}_{seed}.pt')
            reports.append(record)
            save_json(out / 'results.json', reports)
            print('DONE', method, seed, 'test_long', record['test']['long'], 'ood_long', record['ood']['long'],
                  'seconds', round(seconds, 1), 'peak_MiB', round(peak / 2**20, 1), flush=True)
    save_json(out / 'progress.json', {'status': 'complete', 'runs': len(reports)})
    print('COMPLETE', len(reports), 'runs', out, flush=True)
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--overlap-lambda', type=float, default=.1)
    parser.add_argument('--lambda-gf', type=float, default=1e-4)
    parser.add_argument('--scale-source', default='runs/two_stage')
    parser.add_argument('--out', default='runs/original_scale_cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.seeds) < 1 or min(args.overlap_lambda, args.lambda_gf) < 0 or len(set(args.methods)) != len(args.methods):
        parser.error('Positive steps/seeds, nonnegative penalties, unique methods required')
    run(args)
