"""Observed mass-variance calibration of a frozen, global IC-to-gamma map.

For positive whole-line heat solutions, variance grows linearly even when the
total field is not Gaussian. Only observed fields enter this calibration;
diffusivity, mixture parameters and future fields are not inputs or targets.
"""
import math
import os
import time

import torch
from torch import nn
from torch.nn import functional as F

from gamma_gf_model import GammaGFModel
from original_scale import cpu_state, save_json
from moving_charts import state_hash


class MomentGammaPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(192, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(math.expm1(1.)))

    def forward(self, ic):
        return F.softplus(self.net(ic))


class MomentGammaGFModel(GammaGFModel):
    def __init__(self, gamma_state):
        # Construct the backbone before SSV to match the pure latent RNG path.
        super().__init__('global_fixed_gf')
        self.method = 'moment_ssv_gf'
        self.ssv = MomentGammaPredictor()
        self.ssv.load_state_dict(gamma_state)
        self.ssv.requires_grad_(False)
        self.ssv.eval()


def mass_moments(fields, x):
    """Rectangle quadrature on the existing equally spaced, half-open grid.

Weights are u, NOT u**2. The centroid is computed from each observed field;
it is used to measure variance, not to translate the network's coordinates.
"""
    if x.ndim != 1 or len(x) < 2 or fields.shape[-1] != len(x):
        raise ValueError('Expected fields on a one-dimensional spatial grid')
    if not torch.isfinite(fields).all() or (fields < 0).any():
        raise ValueError('Mass variance requires finite nonnegative fields')
    dx = x[1] - x[0]
    if dx <= 0 or not torch.allclose(x[1:] - x[:-1], dx.expand(len(x)-1)):
        raise ValueError('Expected a uniform increasing grid')
    mass = fields.sum(-1) * dx
    if (mass <= 0).any():
        raise ValueError('Positive mass is required')
    center = (fields * x).sum(-1) * dx / mass
    variance = (fields * (x - center[..., None]).square()).sum(-1) * dx / mass
    if not torch.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError('Positive finite variance is required')
    return mass, center, variance


def variance_targets(observed, x, times):
    if observed.ndim != 3 or observed.shape[1:] != (3, len(x)):
        raise ValueError('Calibration accepts only three early observed frames')
    expected = times.new_tensor([0., .25, .5])
    if times.shape != (3,) or not torch.equal(times, expected):
        raise ValueError('Calibration times must be 0, .25, .5')
    # Compute moments in float64 before casting the small growth differences.
    _, _, variance = mass_moments(observed.double(), x.double())
    return (variance / variance[:, :1]).log().to(observed.dtype)


def variance_loss(predictor, ic, targets, times):
    if targets.shape != (len(ic), 3) or times.shape != (3,):
        raise ValueError('Expected three early log-variance targets per IC')
    return (torch.log1p(predictor(ic) * times[None]) - targets).square().mean()


def calibrate(args, train_ic, train_observed, valid_ic, valid_observed, x, times, out):
    """Fit once on CPU, select by early validation, and share across LDNet seeds."""
    values = (train_ic, train_observed, valid_ic, valid_observed, x, times)
    train_ic, train_observed, valid_ic, valid_observed, x, times = [v.cpu() for v in values]
    target = variance_targets(train_observed, x, times)
    valid_target = variance_targets(valid_observed, x, times)
    history = []
    best, best_state, selected = float('inf'), None, None
    start = time.perf_counter()
    for restart in range(args.gamma_restarts):
        torch.manual_seed(9100 + restart)
        predictor = MomentGammaPredictor()
        with torch.no_grad():
            torch.testing.assert_close(predictor(train_ic), torch.ones(len(train_ic), 1), rtol=0, atol=1e-7)
            initial_valid = float(variance_loss(predictor, valid_ic, valid_target, times))
        if initial_valid < best:
            best, best_state, selected = initial_valid, cpu_state(predictor), {'restart': restart, 'step': 0, 'valid': initial_valid}
        optimizer = torch.optim.Adam(predictor.parameters(), lr=.003)
        for step in range(1, args.gamma_steps + 1):
            objective = variance_loss(predictor, train_ic, target, times)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), 5.)
            optimizer.step()
            if step == 1 or step % 25 == 0 or step == args.gamma_steps:
                with torch.no_grad():
                    valid = float(variance_loss(predictor, valid_ic, valid_target, times))
                    gamma = predictor(train_ic)
                if not math.isfinite(valid) or not torch.isfinite(objective) or not torch.isfinite(gamma).all() or not (gamma > 0).all():
                    raise RuntimeError('Nonfinite calibration or nonpositive gamma')
                row = {'restart': restart, 'step': step, 'train': float(objective.detach()), 'valid': valid,
                       'gamma_min': float(gamma.min()), 'gamma_mean': float(gamma.mean()), 'gamma_max': float(gamma.max())}
                history.append(row)
                if valid < best:
                    best, best_state, selected = valid, cpu_state(predictor), row.copy()
            if step == 1 or step % 500 == 0 or step == args.gamma_steps:
                save_json(out/'progress.json', {'status': 'pretraining_gamma', 'pid': os.getpid(),
                    'restart': restart, 'step': step, 'steps_per_restart': args.gamma_steps,
                    'best_early_variance_validation': best})
                print('GAMMA', restart, step, 'early_variance_valid', best, flush=True)
    predictor.load_state_dict(best_state)
    predictor.requires_grad_(False)
    report = {'selection': selected, 'history': history, 'seconds': time.perf_counter()-start,
              'gamma_initial': 1., 'initial_valid': initial_valid, 'parameters': 3105,
              'state_sha256': state_hash(best_state), 'shared_across_ldnet_seeds': True,
              'objective': 'mean[(log(v(t)/v(0)) - log(1+gamma(u0)*t))^2], mass weights u',
              'label_times': times.tolist(), 'selection_uses_test': False}
    for split, ic, targets in (('train', train_ic, target), ('valid', valid_ic, valid_target)):
        with torch.no_grad():
            gamma = predictor(ic)
            residual = torch.log1p(gamma*times[None])-targets
        report[split] = {'gamma': gamma[:, 0].tolist(), 'loss': float(residual.square().mean()),
                         'max_abs_log_variance_residual': float(residual.abs().max())}
    checkpoint = {'state': best_state, 'selection': selected, 'state_sha256': report['state_sha256']}
    torch.save(checkpoint, out/'gamma_predictor.pt')
    save_json(out/'gamma_results.json', report)
    print('GAMMA COMPLETE', selected, 'seconds', round(report['seconds'], 2), flush=True)
    return checkpoint


def backbone_hash(model):
    return state_hash({name.replace('.decoder.base.', '.decoder.'): value
                       for name, value in model.named_parameters() if not name.startswith('ssv.')})
