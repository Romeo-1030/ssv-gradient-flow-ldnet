"""Four-arm ordinary-data comparison with the unchanged domain16 fitting protocol.

The only fit changes are accepting SSV without GF and assigning physical clocks
explicitly. Evaluation is imported unchanged from the existing implementation.
"""
from __future__ import annotations

from ic_gamma_domain16_fit import evaluate_and_save

import time
import numpy as np
import torch
from torch import nn

from experiment import loss
from forecast_gamma_model import checkpoint_score, six_frame_metrics
from ic_gamma_data import SIX
from ic_gamma_domain16_model import Domain16GammaModel, raw_gf_loss, validate_grid, per_ic_metrics
from ic_gamma_experiment import progress, atomic_torch_save, energy_surface
from moment_baselines import gf_weight
from moving_charts import state_hash
from original_scale import cpu_state, sync, make_gf_indices
from original_scale_model import EnergyDensity
from retry_json import save_json


def fit_final(args, method, seed, inputs, labels, x, t, out):
    """Accept only six labels from train/validation ICs, never future truth."""
    if method not in ('latent', 'latent_gf', 'ssv', 'ssv_gf'):
        raise ValueError('Unknown domain16 treatment')
    validate_grid(x, t)
    if set(inputs) != {'train', 'valid'} or set(labels) != {'train', 'valid'}:
        raise ValueError('Final fitting accepts only training and validation splits')
    for split in inputs:
        if labels[split].shape != (len(inputs[split]), 6, 192):
            raise ValueError('Only the six observed fields can enter fitting')
    physical, use_gf = method in ('latent', 'latent_gf'), method.endswith('_gf')
    torch.manual_seed(seed)
    if x.device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = Domain16GammaModel(physical=physical)
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
    if use_gf:
        assert len(indices) == 32 and float(t[indices[-1]]) == 10.
    original_inputs = {k: v.clone() for k, v in inputs.items()}
    scales = model.get_scales().clone()
    best, selected, normalization = float('inf'), None, None
    best_model, best_energy, history = None, None, []
    if x.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(x.device)
    sync(x.device)
    start = time.perf_counter()
    print('FINAL START', method, seed, 'extent16', flush=True)
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
                     steps=args.steps, seconds=elapsed, best_validation=best, spatial_extent=16.)
            save_json(out/f'{method}_{seed}_history.json', history)
            print('FINAL', method, seed, step, 'validation', best, 'seconds', round(elapsed, 1), flush=True)
    sync(x.device)
    seconds = time.perf_counter()-start
    model.load_state_dict(best_model)
    if use_gf:
        energy.load_state_dict(best_energy)
    model.eval()
    model.assert_geometry()
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
        'gf_spatial_extent': 16. if use_gf else None, 'decoder_extent': 16., 'fourier_frequencies': 16,
        'label_times': t[list(SIX)].cpu().tolist(), 'history': history,
        'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(x.device) if x.device.type == 'cuda' else 0}
    checkpoint = {'model': best_model, 'energy': best_energy, 'method': method, 'seed': seed,
        'model_class': 'ic_gamma_domain16_model.Domain16GammaModel',
        'spatial_domain': [-16.,16.], 'endpoint': False, 'spatial_points': 192,
        'fourier_frequencies': 16, 'decoder_coefficients': 33,
        'scales': record['scales'], 'gamma_source': 'gamma_selection.json; external fixed per-IC input'}
    return model, energy, record, checkpoint

