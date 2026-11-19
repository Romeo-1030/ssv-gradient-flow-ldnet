"""Independent ordinary-network gamma calibration, then fresh shared LDNet.

Each IC supplies six early frames to its own non-latent calibration. Final
shared LDNet weights use train ICs only; validation ICs select checkpoints.
The two physical controls keep the original u0-only conditioning interface.
Consequently comparisons are full-pipeline comparisons, not a claim that
test-time information supplied to all three pipelines is identical.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import hashlib
import json
import platform
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiment import loss
from forecast_gamma_model import checkpoint_score, six_frame_metrics
from ic_gamma_data import make_dataset, audit_dataset, PROTOCOL, SIX
from ic_gamma_model import ExternalGammaModel, pack_inputs, per_ic_metrics
from moment_baselines import gf_weight
from moving_charts import state_hash
from original_scale import cpu_state, sync, make_gf_indices, raw_gf_loss
from original_scale_model import EnergyDensity, ORIGINAL_SOURCE
from retry_json import save_json
from two_gaussian_data import fingerprint


ROOT = Path(__file__).resolve().parent
METHODS = ('latent', 'latent_gf', 'ssv_gf')


def utc():
    return datetime.now(timezone.utc).isoformat()


def progress(out, phase, **extra):
    save_json(out/'progress.json', {'status': 'running', 'pid': os.getpid(),
        'phase': phase, 'updated_utc': utc(), **extra})


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def atomic_torch_save(value, path):
    temp = path.with_name(path.name+'.tmp')
    torch.save(value, temp)
    temp.replace(path)


def fit_final(args, method, seed, inputs, labels, x, t, out):
    """Accept six labeled frames only; no test data or future truth inputs."""
    physical, use_gf = method != 'ssv_gf', method.endswith('_gf')
    torch.manual_seed(seed)
    if x.device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = ExternalGammaModel(physical=physical)
    initial_hash = state_hash(dict(model.named_parameters()))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10000+seed)
        energy = EnergyDensity() if use_gf else None
    energy_initial_hash = state_hash(energy.state_dict()) if energy is not None else None
    model.to(x.device)
    if energy is not None:
        energy.to(x.device)
    backbone = list(model.parameters())
    assert sum(p.numel() for p in backbone) == 91329
    optimizer = torch.optim.AdamW(backbone+(list(energy.parameters()) if use_gf else []),
                                  lr=.001, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=1e-5)
    indices = make_gf_indices(t, 32) if use_gf else None
    original_inputs = {k: v.clone() for k, v in inputs.items()}
    scales = model.get_scales().clone()
    best, selected, normalization = float('inf'), None, None
    best_model, best_energy, history = None, None, []
    if x.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(x.device)
    sync(x.device)
    start = time.perf_counter()
    print('FINAL START', method, seed, flush=True)
    for step in range(1, args.steps+1):
        model.train()
        predictions = model(inputs['train'], t[:11], x)[:, list(SIX)]
        supervised = loss(predictions, labels['train'])
        raw = raw_gf_loss(model, energy, inputs['train'], t, indices, 24) if use_gf else supervised.new_zeros(())
        if use_gf and step == 101:
            normalization = max(float(raw.detach()), 1e-12)
        weight = gf_weight(step) if use_gf else 0.
        normalized = raw/normalization if weight else raw.new_zeros(())
        objective = supervised+weight*normalized
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(backbone, 5.)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                valid = checkpoint_score(model, inputs['valid'], labels['valid'], x, t, SIX)
            row = {'step': step, 'train': float(objective.detach()), 'data': float(supervised.detach()),
                   'raw_gf': float(raw.detach()), 'normalized_gf': float(normalized.detach()),
                   'gf_weight': weight, 'valid': float(valid)}
            if not all(np.isfinite(v) for v in row.values()):
                raise RuntimeError(f'Nonfinite final training: {method}, seed={seed}, step={step}')
            history.append(row)
            if row['valid'] < best:
                best, selected = row['valid'], step
                best_model = cpu_state(model)
                best_energy = cpu_state(energy) if use_gf else None
        if step == 1 or step % 250 == 0 or step == args.steps:
            sync(x.device)
            elapsed = time.perf_counter()-start
            progress(out, 'final_training', method=method, seed=seed, step=step,
                     steps=args.steps, seconds=elapsed, best_validation=best)
            save_json(out/f'{method}_{seed}_history.json', history)
            print('FINAL', method, seed, step, 'validation', best, 'seconds', round(elapsed, 1), flush=True)
    sync(x.device)
    seconds = time.perf_counter()-start
    model.load_state_dict(best_model)
    if use_gf:
        energy.load_state_dict(best_energy)
    model.eval()
    torch.testing.assert_close(model.get_scales(), scales, rtol=0, atol=0)
    for split in inputs:
        torch.testing.assert_close(inputs[split], original_inputs[split], rtol=0, atol=0)
        assert not inputs[split].requires_grad
    assert not any('gamma' in key for key, _ in model.named_parameters())
    record = {'method': method, 'seed': seed, 'physical': physical, 'gf': use_gf,
        'steps': args.steps, 'selected_step': selected, 'best_validation': best, 'seconds': seconds,
        'backbone_parameters': 91329, 'energy_parameters': 4417 if use_gf else 0,
        'initial_parameters_sha256': initial_hash, 'initial_energy_sha256': energy_initial_hash,
        'gamma_frozen': True, 'scales': scales.cpu().tolist(), 'gf_scale': normalization,
        'gf_max_weight': gf_weight(args.steps) if use_gf else 0., 'gf_selected_weight': gf_weight(selected) if use_gf else 0.,
        'gf_times': t[indices].cpu().tolist() if use_gf else [], 'gf_collocation': 24 if use_gf else 0,
        'label_times': t[list(SIX)].cpu().tolist(), 'history': history,
        'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(x.device) if x.device.type == 'cuda' else 0}
    checkpoint = {'model': best_model, 'energy': best_energy, 'method': method, 'seed': seed,
                  'scales': record['scales'], 'gamma_source': 'gamma_selection.json; external fixed per-IC input'}
    return model, energy, record, checkpoint


def energy_surface(energy, pred, gamma, x, t, path, physical):
    values = pred.detach().cpu().numpy().astype(np.float64)
    scale = np.ones((len(gamma), len(t))) if physical else np.sqrt(1+gamma[:, None]*t[None])
    transformed = values*scale[:, :, None]
    derivative = np.gradient(transformed, x, axis=-1)*scale[:, :, None]
    low, high = np.quantile(transformed, [.005, .995])
    extent = max(float(np.quantile(np.abs(derivative), .995)), .05)
    wgrid = np.linspace(min(low, -.05), max(high, .1), 65)
    wxgrid = np.linspace(-extent, extent, 65)
    device = next(energy.parameters()).device
    w, wx = np.meshgrid(wgrid, wxgrid, indexing='ij')
    with torch.no_grad():
        density = energy(torch.tensor(w, device=device, dtype=torch.float32),
                         torch.tensor(wx, device=device, dtype=torch.float32)).cpu().numpy()
        offset = float(energy(torch.zeros((), device=device), torch.zeros((), device=device)))
    np.savez_compressed(path, wgrid=wgrid, wxgrid=wxgrid, e=density, origin_energy=offset)


def evaluate_and_save(model, energy, row, checkpoint, data, ids, inputs, x, t, out):
    method, seed = row['method'], row['seed']
    with torch.no_grad():
        pred = model(inputs, t, x)
        truth = torch.tensor(data['u'][ids], device=x.device)
        row['metrics'] = six_frame_metrics(pred, truth, t) | per_ic_metrics(pred, truth, t)
    row['prediction'] = f'{method}_{seed}_test.npz'
    row['checkpoint'] = f'{method}_{seed}.pt'
    gamma = inputs[:, -1].cpu().numpy()
    np.savez_compressed(out/row['prediction'], pred=pred.cpu().numpy(), truth=truth.cpu().numpy(),
        x=data['x'], t=data['t'], gamma=gamma, sample_ids=data['sample_id'][ids])
    atomic_torch_save(checkpoint, out/row['checkpoint'])
    # Independent checkpoint replay verifies the saved model really produces the reported field.
    replay = ExternalGammaModel(physical=row['physical']).to(x.device)
    restored = torch.load(out/row['checkpoint'], map_location=x.device, weights_only=True)
    replay.load_state_dict(restored['model'])
    replay.eval()
    with torch.no_grad():
        torch.testing.assert_close(replay(inputs, t, x), pred, rtol=0, atol=0)
    row['checkpoint_replay_passed'] = True
    if energy is not None:
        row['energy_surface'] = f'energy_{method}_{seed}.npz'
        energy_surface(energy, pred, gamma, data['x'], data['t'], out/row['energy_surface'], row['physical'])
    return row


def source_manifest():
    files = sorted(ROOT.glob('*.py')) + [ROOT/'start_ic_gamma.ps1']
    result = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}
    result['upstream_train.py'] = hashlib.sha256(ORIGINAL_SOURCE.read_bytes()).hexdigest()
    return result


def write_report(out):
    from ic_gamma_report import build_report
    return build_report(out)


def run(args):
    from ic_gamma_selector import calibrate_all, evaluate_selected
    out = (ROOT/args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the requested full experiment')
    config_path = out/'config.json'
    manifest = source_manifest()
    args_config = vars(args).copy()
    args_config.pop('resume')
    if config_path.exists():
        if not args.resume:
            raise FileExistsError('Run exists; resume explicitly instead of overwriting')
        previous = read(config_path)
        if previous['arguments'] != args_config or previous['source_sha256'] != manifest:
            raise RuntimeError('Resume configuration or source differs from frozen run')
    else:
        snapshot = out/'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for name in manifest:
            shutil.copy2(ORIGINAL_SOURCE if name == 'upstream_train.py' else ROOT/name, snapshot/name)
        save_json(config_path, {'arguments': args_config, 'source_sha256': manifest,
            'created_utc': utc(), 'data_protocol': PROTOCOL, 'seeds': args.seeds, 'steps': args.steps,
            'methods': list(METHODS), 'selector_steps': args.selector_steps,
            'selector_seeds': list(range(200, 200+args.selector_seeds)),
            'backbone': 'original latent16,width128,depth3,33 Fourier coefficients; IC encoder16 plus supplied log(gamma)',
            'training': 'One shared model per arm/seed, freshly initialized after independent ordinary per-IC gamma calibration',
            'selection': 'Each IC, including test ICs, uses its own six early observations. No fields after1.25 enter selection.',
            'test_time_information': {'ssv_gf': 'u0 plus own six-frame gamma calibration',
                'latent': 'u0-only conditioning, original interface', 'latent_gf': 'u0-only conditioning, original interface'},
            'comparison_scope': 'Full-pipeline baseline comparison; SSV gains also include extra per-test-IC early-observation adaptation.',
            'gamma_reference_usage': 'Generator and diagnostics only, never provided to ordinary selector or final network.',
            'gf': 'Original residual; same learned density; perIC gamma fixed; physical residual uses gamma1 and alpha=beta0.',
            'target_metric': 'test mean relative L2 over 2<t<=6; 10 percent is a target, not a stopping or tuning criterion',
            'torch': torch.__version__, 'python': platform.python_version(),
            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'})
    progress(out, 'data')
    counts = {'train': 3, 'valid': 2, 'test': 2} if args.smoke else None
    if (out/'data.npz').exists():
        with np.load(out/'data.npz') as loaded:
            data = {key: loaded[key] for key in loaded.files}
    else:
        data = make_dataset(args.data_seed, counts)
        np.savez_compressed(out/'data.npz', **data)
    audit = audit_dataset(data)
    audit['fingerprint'] = fingerprint(data)
    save_json(out/'data_audit.json', audit)
    write_report(out)
    # Only this small observed tensor crosses the selector API boundary.
    observed = np.ascontiguousarray(data['u'][:, list(SIX)])
    selected_path = out/'gamma_selection.json'
    if selected_path.exists():
        selection = read(selected_path)
    else:
        progress(out, 'gamma_calibration', trajectories=len(observed))
        def selector_progress(value):
            progress(out, 'gamma_calibration', selector=value)
        selection = calibrate_all(observed, data['x'], data['t'][list(SIX)], out/'calibration',
            sample_ids=data['sample_id'], device=str(device), steps=args.selector_steps,
            seeds=tuple(range(200, 200+args.selector_seeds)), batch_ics=args.selector_batch,
            width=64, resume=True, progress_callback=selector_progress)
        for i, row in enumerate(selection['records']):
            assert str(row['sample_id']) == str(data['sample_id'][i])
            row['split'] = str(data['split'][i])
            row['gamma_reference'] = float(data['gamma_reference'][i])
        selection['audit_reference_added_after_selection'] = True
        selection['data_fingerprint'] = audit['fingerprint']
        save_json(selected_path, selection)
    if selection['data_fingerprint'] != audit['fingerprint']:
        raise RuntimeError('Gamma assignments do not match this dataset')
    gammas = np.asarray(selection['selected_gamma'], dtype=np.float32)
    assert gammas.shape == (len(data['u']),) and np.isfinite(gammas).all() and (gammas > 0).all()
    if not (out/'ordinary_selected.npz').exists():
        progress(out, 'ordinary_function_export')
        ordinary = evaluate_selected(out/'calibration', data['x'], data['t'][list(SIX)], device=str(device))
        np.savez_compressed(out/'ordinary_selected.npz', **ordinary, truth=observed)
    x = torch.tensor(data['x'], dtype=torch.float32, device=device)
    t = torch.tensor(data['t'], dtype=torch.float32, device=device)
    ix = {s: np.flatnonzero(data['split'] == s) for s in ('train', 'valid', 'test')}
    fields = {s: torch.tensor(data['u'][ids, 0], device=device) for s, ids in ix.items()}
    labels = {s: torch.tensor(observed[ix[s]], device=device) for s in ('train', 'valid')}
    # Test labels are absent from the final fit interface.
    rows = read(out/'results.json', [])
    write_report(out)
    for seed in range(args.seeds):
        for method in METHODS:
            if any(r['method'] == method and r['seed'] == seed for r in rows):
                continue
            inputs = {s: pack_inputs(fields[s], gammas[ids] if method == 'ssv_gf' else np.ones(len(ids)))
                      for s, ids in ix.items()}
            model, energy, row, checkpoint = fit_final(args, method, seed,
                {s: inputs[s] for s in ('train', 'valid')}, labels, x, t, out)
            row = evaluate_and_save(model, energy, row, checkpoint, data, ix['test'], inputs['test'], x, t, out)
            row['gamma_selection_file'] = 'gamma_selection.json' if method == 'ssv_gf' else None
            rows.append(row)
            save_json(out/'results.json', rows)
            write_report(out)
            print('FINAL DONE', method, seed, 'long', row['metrics']['long'], 'seconds', row['seconds'], flush=True)
            del model, energy, checkpoint
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    for seed in range(args.seeds):
        paired = [r for r in rows if r['seed'] == seed]
        assert len(paired) == 3 and len({r['initial_parameters_sha256'] for r in paired}) == 1
        gf_rows = [r for r in paired if r['gf']]
        assert len({r['initial_energy_sha256'] for r in gf_rows}) == 1
        plain, physical_gf = [next(r for r in paired if r['method'] == m) for m in ('latent', 'latent_gf')]
        a = [(r['step'], r['data'], r['valid']) for r in plain['history'] if r['step'] <= 100]
        b = [(r['step'], r['data'], r['valid']) for r in physical_gf['history'] if r['step'] <= 100]
        assert a == b, 'Physical paired arms differ before GF activates'
    save_json(out/'audit.json', {'completed_utc': utc(), 'complete_runs': len(rows),
        'expected_runs': 3*args.seeds, 'paired_initialization_passed': True,
        'physical_GF_warmup_equivalence_passed': True, 'checkpoint_replay_passed': True,
        'gamma_frozen_during_final_training': True, 'final_weights_fresh_after_calibration': True,
        'gamma_selector_contains_no_latent_dynamics': True,
        'per_test_IC_six_observations_used_for_gamma': True,
        'test_future_after_1_25_used_only_for_evaluation': True})
    write_report(out)
    save_json(out/'progress.json', {'status': 'complete', 'pid': os.getpid(), 'phase': 'complete',
                                  'completed_utc': utc(), 'completed_runs': len(rows)})
    (out/'stop_awake').touch()
    print('COMPLETE', out, flush=True)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='runs/ic_gamma_controlled_cuda')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--data-seed', type=int, default=91)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--seeds', type=int, default=4)
    p.add_argument('--selector-steps', type=int, default=1500)
    p.add_argument('--selector-seeds', type=int, default=2)
    p.add_argument('--selector-batch', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if min(a.steps, a.seeds, a.selector_steps, a.selector_seeds, a.selector_batch) < 1:
        p.error('Counts and update budgets must be positive')
    return a


if __name__ == '__main__':
    args = arguments()
    try:
        run(args)
    except BaseException as error:
        output = (ROOT/args.out).resolve()
        output.mkdir(parents=True, exist_ok=True)
        save_json(output/'progress.json', {'status': 'failed', 'pid': os.getpid(),
            'failed_utc': utc(), 'error': repr(error), 'traceback': traceback.format_exc()})
        (output/'stop_awake').touch()
        raise
