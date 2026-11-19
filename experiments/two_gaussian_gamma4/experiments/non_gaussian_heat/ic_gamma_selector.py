"""Per-trajectory SSV calibration with ordinary coordinate MLPs, never LDNet.

Each IC/candidate/restart owns independent MLP weights.  Only the first three
fields fit those weights; the last three observed fields select gamma.  A
shape-dependent search interval is a heuristic around an observed moment
clock, not a Bayesian posterior or a supplied generative/oracle gamma.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import time

import numpy as np
import torch
from torch import nn

from retry_json import save_json


OBSERVED_TIMES = (0., .25, .5, .75, 1., 1.25)
WEIGHTS = (1., 2., 3.)
SEARCH_SUPPORT = (.125, 8.)
PRIOR_LOG_MEAN = 0.
PRIOR_LOG_SD = math.log(2.)
MODEL_VERSION = 1


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _digest_arrays(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(_numpy(value))
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def _state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(_numpy(value)).tobytes())
    return digest.hexdigest()


def _validate_observations(observed, x, observed_times):
    observed, x, observed_times = map(_numpy, (observed, x, observed_times))
    if observed.ndim != 3 or len(observed) == 0 or observed.shape[1:] != (6, len(x)):
        raise ValueError('Only N-by-6-by-X observed fields are accepted')
    if x.ndim != 1 or len(x) < 3 or not np.all(np.diff(x) > 0):
        raise ValueError('An increasing spatial grid is required')
    if not np.allclose(np.diff(x), x[1]-x[0], rtol=1e-8, atol=1e-10):
        raise ValueError('The observed spatial grid must be uniform')
    if observed_times.shape != (6,) or not np.array_equal(observed_times, OBSERVED_TIMES):
        raise ValueError('Only observed times 0,.25,.5,.75,1,1.25 are accepted')
    if not np.isfinite(observed).all() or not np.isfinite(x).all() or np.any(observed < 0):
        raise ValueError('Positive-mixture observations must be finite and nonnegative')
    if np.any(observed.sum(-1) <= 0):
        raise ValueError('Observed fields must have positive mass')
    return observed.astype(np.float32), x.astype(np.float32), observed_times.astype(np.float32)


def initial_dictionary(early, x):
    """Nine candidates from the first three fields only, with two anchors.

The moment clock is measured slope(V)/V(0), not 2*known_diffusivity/V(0).
Nine lognormal-prior quantiles cover [g_m/4,4*g_m] intersect [.125,8].
Interior points nearest g_m and 1 are replaced by those anchors when inside.
The endpoints are preserved.  Shape conditioning here is a search heuristic.
"""
    early = _numpy(early).astype(np.float64)
    x = _numpy(x).astype(np.float64)
    if early.ndim != 2 or early.shape != (3, len(x)):
        raise ValueError('Dictionary construction accepts exactly three early fields')
    if not np.isfinite(early).all() or np.any(early < 0):
        raise ValueError('Finite nonnegative early observations required')
    mass = early.sum(-1)
    if np.any(mass <= 0):
        raise ValueError('Positive observed mass required')
    centers = (early*x[None]).sum(-1)/mass
    variance = (early*(x[None]-centers[:, None])**2).sum(-1)/mass
    if np.any(variance <= 0):
        raise ValueError('Positive observed variance required')
    early_t = np.asarray(OBSERVED_TIMES[:3])
    slope = float(np.dot(early_t, variance-variance[0])/np.dot(early_t, early_t))
    estimated = slope/float(variance[0])
    fallback = not np.isfinite(estimated) or estimated <= 0
    moment = 1. if fallback else estimated
    support_lo, support_hi = SEARCH_SUPPORT
    # This clipping only defines the search center, not the data distribution.
    center = float(np.clip(moment, support_lo, support_hi))
    low = max(support_lo, center/4.)
    high = min(support_hi, center*4.)
    normal = NormalDist(PRIOR_LOG_MEAN, PRIOR_LOG_SD)
    left, right = normal.cdf(math.log(low)), normal.cdf(math.log(high))
    candidates = np.array([math.exp(normal.inv_cdf(left+(right-left)*q))
                           for q in np.linspace(0., 1., 9)])
    candidates[0], candidates[-1] = low, high
    reserved = {0, 8}
    for anchor in (center, 1.):
        if not low <= anchor <= high:
            continue
        exact = np.flatnonzero(np.isclose(candidates, anchor, rtol=1e-10, atol=1e-12))
        if len(exact):
            reserved.add(int(exact[0]))
            continue
        available = [i for i in range(1, 8) if i not in reserved]
        nearest = min(available, key=lambda i: abs(math.log(candidates[i]/anchor)))
        candidates[nearest] = anchor
        reserved.add(nearest)
    candidates.sort()
    if len(np.unique(candidates)) != 9 or not np.all(np.diff(candidates) > 0):
        raise AssertionError('Dictionary must have exactly nine distinct candidates')
    return {'candidates': candidates.tolist(), 'gamma_moment': float(moment),
            'moment_fallback': fallback, 'moment_slope': slope,
            'early_variance': variance.tolist(), 'early_centroid': centers.tolist(),
            'search_bounds': [low, high], 'prior_log_mean': PRIOR_LOG_MEAN,
            'prior_log_sd': PRIOR_LOG_SD,
            'dictionary_rule': 'nine conditional lognormal-prior quantiles, preserving endpoints and moment/one anchors'}


class IndependentCoordinateMLPs(nn.Module):
    """Batched independent 2 -> width -> width -> 1 tanh coordinate nets.

The leading parameter dimension indexes independent functions.  Candidate
networks have exactly matching initialization within a restart.  There is no
latent state, learned time integrator, trajectory encoder, or PDE residual.
"""
    def __init__(self, count, width=64, seed=200, dtype=torch.float32):
        super().__init__()
        if count < 1 or width < 2:
            raise ValueError('Positive network count and width >= 2 required')
        self.count, self.width = int(count), int(width)
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for fan_in, fan_out in ((2, width), (width, width), (width, 1)):
            bound = 1/math.sqrt(fan_in)
            weight = torch.empty(fan_in, fan_out, dtype=dtype).uniform_(-bound, bound, generator=generator)
            bias = torch.empty(1, fan_out, dtype=dtype).uniform_(-bound, bound, generator=generator)
            self.weights.append(nn.Parameter(weight[None].repeat(count, 1, 1)))
            self.biases.append(nn.Parameter(bias[None].repeat(count, 1, 1)))

    def forward(self, coordinates):
        if coordinates.ndim != 3 or coordinates.shape[0] != self.count or coordinates.shape[-1] != 2:
            raise ValueError('Expected network-by-point-by-2 coordinates')
        value = coordinates
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = torch.bmm(value, weight)+bias
            if i < 2:
                value = torch.tanh(value)
        return value[..., 0]


def physical_prediction(model, gamma, amplitude, x, times):
    """Evaluate ordinary coordinate networks after an explicit SSV pullback."""
    scale = 1+gamma[:, None]*times[None]
    xi = x[None, None]/scale[:, :, None].sqrt()
    tau = scale.log()[:, :, None].expand_as(xi)
    coordinates = torch.stack((xi/4., tau), -1).reshape(len(gamma), -1, 2)
    output = model(coordinates).reshape(len(gamma), len(times), len(x))
    return amplitude[:, None, None]*output/scale[:, :, None].sqrt()


def per_model_data_loss(prediction, targets):
    if prediction.shape != targets.shape or prediction.ndim != 3:
        raise ValueError('Matching network-by-time-by-space arrays required')
    return ((prediction-targets).square().sum(-1)
            / targets[:, :1].square().sum(-1).clamp_min(1e-12)).mean(-1)


def per_model_holdout_curve(prediction, targets):
    if prediction.shape != targets.shape or prediction.ndim != 3 or prediction.shape[1] != 3:
        raise ValueError('Matching network-by-three-heldout-times-by-space arrays required')
    return ((prediction.double()-targets.double()).square().sum(-1)
            / targets.double().square().sum(-1).clamp_min(1e-12)).sqrt()


def select_per_ic(candidates, curves, weights=WEIGHTS):
    """curves shape: restart-by-IC-by-candidate-by-three-times."""
    candidates, curves = np.asarray(candidates), np.asarray(curves)
    if candidates.ndim != 2 or curves.ndim != 4 or curves.shape[1:3] != candidates.shape or curves.shape[-1] != 3:
        raise ValueError('Candidate/curve shapes do not match')
    if not np.isfinite(curves).all() or np.any(curves < 0):
        raise ValueError('Finite nonnegative calibration errors required')
    weights = np.asarray(weights, dtype=float)
    weights = weights/weights.sum()
    seed_scores = curves@weights
    scores = seed_scores.mean(0)
    indices = np.array([min(range(len(row)), key=lambda j: (scores[i, j], abs(math.log(row[j])), row[j]))
                        for i, row in enumerate(candidates)], dtype=np.int64)
    return indices, scores, seed_scores


def _clip_independent_gradients(model, maximum=5.):
    square = None
    for parameter in model.parameters():
        if parameter.grad is not None:
            term = parameter.grad.square().flatten(1).sum(1)
            square = term if square is None else square+term
    factor = (maximum/square.sqrt().clamp_min(1e-12)).clamp_max(1.)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(factor.reshape((-1,)+(1,)*(parameter.ndim-1)))


def _fit_chunk(early, x, gamma, seed, steps, width, device, progress=None):
    """This fitting function receives no held-out frames at all."""
    device = torch.device(device)
    targets = torch.as_tensor(early, device=device, dtype=torch.float32)
    spatial = torch.as_tensor(x, device=device, dtype=torch.float32)
    rates = torch.as_tensor(gamma, device=device, dtype=torch.float32)
    times = spatial.new_tensor(OBSERVED_TIMES[:3])
    amplitude = targets[:, 0].amax(-1).clamp_min(1e-12)
    model = IndependentCoordinateMLPs(len(rates), width, seed).to(device)
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=2e-5)
    best = torch.full((len(rates),), float('inf'), device=device)
    best_step = torch.zeros(len(rates), dtype=torch.int64, device=device)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for step in range(1, steps+1):
        model.train()
        prediction = physical_prediction(model, rates, amplitude, spatial, times)
        losses = per_model_data_loss(prediction, targets)
        if not bool(torch.isfinite(losses).all()):
            raise RuntimeError('Nonfinite ordinary-network calibration loss')
        # Sum, not mean, gives each independent fit its standalone gradient.
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        _clip_independent_gradients(model)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 25 == 0 or step == steps:
            with torch.no_grad():
                evaluated = per_model_data_loss(physical_prediction(model, rates, amplitude, spatial, times), targets)
                improved = evaluated < best
                best[improved] = evaluated[improved]
                best_step[improved] = step
                for key, value in model.state_dict().items():
                    best_state[key][improved] = value[improved]
            if progress:
                progress(step, float(best.mean()), time.perf_counter()-start)
    model.load_state_dict(best_state)
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return {'model': model, 'state': {k: v.detach().cpu() for k, v in best_state.items()},
            'amplitude': amplitude.detach().cpu(), 'best_training_loss': best.cpu().tolist(),
            'best_step': best_step.cpu().tolist(), 'initial_state_sha256': _state_hash(initial_state),
            'seconds': time.perf_counter()-start}


def _save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def calibrate_all(observed, x, observed_times, out, *, sample_ids=None,
                  device='cuda:0', steps=1500, seeds=(200, 201), batch_ics=4,
                  width=64, resume=True, stop_file=None, progress_callback=None):
    """Calibrate all trajectories independently using six available frames.

The caller may pass train/validation/test observations together because no
weights, losses, moments, or gamma decisions mix ICs. Test-trajectory first
three fields FIT its ordinary network and last three SELECT its gamma. This
is explicit six-frame test-time adaptation, not u0-only inference.
"""
    observed, x, times = _validate_observations(observed, x, observed_times)
    if steps < 1 or batch_ics < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Positive fit sizes and distinct nonempty restart seeds required')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ids = np.arange(len(observed), dtype=np.int64) if sample_ids is None else np.asarray(sample_ids)
    if ids.shape != (len(observed),) or len(set(ids.tolist())) != len(ids):
        raise ValueError('Unique sample IDs are required')
    dictionaries = [initial_dictionary(row[:3], x) for row in observed]
    candidates = np.asarray([r['candidates'] for r in dictionaries], dtype=np.float64)
    config = {'version': MODEL_VERSION, 'steps': int(steps), 'seeds': list(map(int, seeds)),
              'batch_ics': int(batch_ics), 'width': int(width), 'observed_times': list(OBSERVED_TIMES),
              'holdout_weights': list(WEIGHTS), 'sample_ids': ids.tolist(),
              'input_sha256': _digest_arrays(observed, x, times, ids),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'candidate_sha256': _digest_arrays(candidates), 'torch': torch.__version__,
              'architecture': 'independent ordinary coordinate MLP 2-width-width-1 tanh',
              'spatial_input': 'xi/4', 'time_input': 'log(1+gamma*t)',
              'amplitude_normalizer': 'max(initial field)',
              'fit_labels': list(OBSERVED_TIMES[:3]), 'selection_labels': list(OBSERVED_TIMES[3:]),
              'checkpoint_selection': 'per-model first-three-frame data loss, evaluated every25steps',
              'information': 'Each IC uses its own six observations; all ordinary fits independent; no latent network, PDE coefficient or generative parameters supplied',
              'dictionary_prior': {'log_mean': PRIOR_LOG_MEAN, 'log_sd': PRIOR_LOG_SD,
                  'padded_search_support': list(SEARCH_SUPPORT), 'interval': 'intersection of padded support and [observed_moment_gamma/4, 4*observed_moment_gamma]',
                  'candidate_rule': '9 prior quantiles including endpoints; replace nearest unreserved interior nodes with moment gamma and1 if in interval',
                  'interpretation': 'initial-dependent heuristic search interval, not posterior'}}
    config_path = out/'config.json'
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding='utf-8'))
        if not resume or saved != config:
            raise ValueError('Existing selector output does not match immutable input/config/source')
    else:
        save_json(config_path, config)
    curves = np.empty((len(seeds), len(observed), 9, 3), dtype=np.float64)
    fit_losses = np.empty((len(seeds), len(observed), 9), dtype=np.float64)
    chunk_records = []
    full_start = time.perf_counter()
    for seed_index, seed in enumerate(seeds):
        for first in range(0, len(observed), batch_ics):
            last = min(len(observed), first+batch_ics)
            key = f'ordinary_seed{seed}_ics{first:03d}-{last:03d}'
            record_path, checkpoint_path = out/f'{key}.json', out/f'{key}.pt'
            identity = {'seed': int(seed), 'first': first, 'last': last,
                        'input_sha256': _digest_arrays(observed[first:last], candidates[first:last], ids[first:last]),
                        'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
            if record_path.exists() and checkpoint_path.exists():
                record = json.loads(record_path.read_text(encoding='utf-8'))
                if record['identity'] != identity:
                    raise ValueError('Selector shard identity mismatch')
                cp = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                if record['state_sha256'] != _state_hash(cp['state']):
                    raise ValueError('Selector shard checkpoint checksum mismatch')
            else:
                def progress(step, loss, seconds):
                    if stop_file and Path(stop_file).exists():
                        raise RuntimeError('Selector stopped by explicit stop marker')
                    update = {'status': 'calibrating', 'seed': int(seed),
                        'first': first, 'last': last, 'step': step, 'steps': steps,
                        'fit_loss_mean': loss, 'chunk_seconds': seconds,
                        'completed_chunks': len(chunk_records),
                        'total_chunks': len(seeds)*math.ceil(len(observed)/batch_ics), 'pid': os.getpid()}
                    save_json(out/'progress.json', update)
                    if progress_callback:
                        progress_callback(update)
                    if step == 1 or step % 250 == 0 or step == steps:
                        print('ORDINARY SELECTOR', seed, first, last, step, 'fit', loss, 'seconds', round(seconds, 1), flush=True)
                early = np.repeat(observed[first:last, :3], 9, axis=0)
                fitted = _fit_chunk(early, x, candidates[first:last].reshape(-1), int(seed), steps, width, device, progress)
                model = fitted.pop('model')
                with torch.no_grad():
                    model_device = next(model.parameters()).device
                    pred = physical_prediction(model,
                        torch.as_tensor(candidates[first:last].reshape(-1), dtype=torch.float32, device=model_device),
                        fitted['amplitude'].to(model_device),
                        torch.as_tensor(x, device=model_device),
                        torch.tensor(OBSERVED_TIMES[3:], dtype=torch.float32, device=model_device))
                    targets = torch.as_tensor(np.repeat(observed[first:last, 3:], 9, axis=0), device=model_device)
                    holdout = per_model_holdout_curve(pred, targets).cpu().numpy().reshape(last-first, 9, 3)
                checkpoint = {'state': fitted['state'], 'amplitude': fitted['amplitude'],
                              'gamma': torch.tensor(candidates[first:last].reshape(-1), dtype=torch.float32),
                              'sample_ids': ids[first:last].tolist(), 'width': int(width), 'seed': int(seed)}
                record = {'identity': identity, 'checkpoint': checkpoint_path.name,
                          'state_sha256': _state_hash(fitted['state']),
                          'initial_state_sha256': fitted['initial_state_sha256'],
                          'best_training_loss': np.asarray(fitted['best_training_loss']).reshape(last-first, 9).tolist(),
                          'best_step': np.asarray(fitted['best_step']).reshape(last-first, 9).tolist(),
                          'holdout_curve': holdout.tolist(), 'seconds': fitted['seconds'],
                          'parameters_per_ordinary_network': sum(p.numel() for p in model.parameters())//model.count,
                          'optimizer_updates': int(steps), 'independent_networks': model.count}
                _save_checkpoint(checkpoint_path, checkpoint)
                save_json(record_path, record)
                del model, fitted, pred, targets, checkpoint
            curves[seed_index, first:last] = record['holdout_curve']
            fit_losses[seed_index, first:last] = record['best_training_loss']
            chunk_records.append(record)
            if progress_callback:
                progress_callback({'status': 'chunk_complete', 'completed_chunks': len(chunk_records),
                    'total_chunks': len(seeds)*math.ceil(len(observed)/batch_ics), 'seed': int(seed),
                    'first': first, 'last': last, 'chunk_seconds': record['seconds']})
    chosen, scores, seed_scores = select_per_ic(candidates, curves)
    rows = []
    for i, dictionary in enumerate(dictionaries):
        j = int(chosen[i])
        rows.append(dict(dictionary, sample_id=ids[i].item(), gamma_selected=float(candidates[i, j]),
            selected_index=j, scores=scores[i].tolist(), seed_scores=seed_scores[:, i].tolist(),
            holdout_curves=curves[:, i].tolist(), fit_losses=fit_losses[:, i].tolist(),
            selected_score=float(scores[i, j]),
            selected_fit_loss=float(fit_losses[:, i, j].mean()),
            selected_at_boundary=j in (0, 8)))
    report = {'schema_version': MODEL_VERSION, 'selected_gamma': [r['gamma_selected'] for r in rows],
              'records': rows, 'config': config, 'chunks': chunk_records,
              'seconds_training_total': sum(r['seconds'] for r in chunk_records),
              'seconds_this_invocation': time.perf_counter()-full_start,
              'all_trajectories_independent': True, 'selection_uses_latent': False,
              'post_1_25_labels_used': False, 'ordinary_weights_passed_to_ldnet': False,
              'selected_at_boundary_count': sum(r['selected_at_boundary'] for r in rows)}
    save_json(out/'calibration.json', report)
    save_json(out/'progress.json', {'status': 'complete', 'records': len(rows),
                                  'seconds_training_total': report['seconds_training_total']})
    return report


@torch.no_grad()
def evaluate_selected(out, x, times, *, device='cpu'):
    """Evaluate retained selected ordinary functions; receives no true fields.

Returns every restart's prediction and the mean. Gamma selection is by mean
individual restart error, not by the error of this plotted ensemble mean.
"""
    out = Path(out)
    report = json.loads((out/'calibration.json').read_text(encoding='utf-8'))
    config = report['config']
    x, times = np.asarray(_numpy(x), dtype=np.float32), np.asarray(_numpy(times), dtype=np.float32)
    if times.ndim != 1 or np.any(times < 0) or not np.isfinite(times).all():
        raise ValueError('Finite nonnegative evaluation times required')
    predictions = np.empty((len(config['seeds']), len(report['records']), len(times), len(x)), dtype=np.float32)
    for chunk in report['chunks']:
        cp = torch.load(out/chunk['checkpoint'], map_location='cpu', weights_only=True)
        if _state_hash(cp['state']) != chunk['state_sha256']:
            raise ValueError('Retained ordinary-function checkpoint checksum mismatch')
        identity = chunk['identity']
        first, last = identity['first'], identity['last']
        indices = torch.tensor([9*(i-first)+report['records'][i]['selected_index'] for i in range(first, last)])
        state = {k: v[indices] for k, v in cp['state'].items()}
        model = IndependentCoordinateMLPs(last-first, config['width'], 0).to(device)
        model.load_state_dict(state)
        pred = physical_prediction(model, cp['gamma'][indices].to(device), cp['amplitude'][indices].to(device),
                                   torch.as_tensor(x, device=device), torch.as_tensor(times, device=device))
        predictions[config['seeds'].index(identity['seed']), first:last] = pred.cpu().numpy()
    return {'by_seed': predictions, 'mean': predictions.mean(0),
            'gamma': np.asarray(report['selected_gamma']), 'x': x, 't': times,
            'sample_ids': np.asarray(config['sample_ids'])}
