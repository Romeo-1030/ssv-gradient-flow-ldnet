"""Matched-seed ablation of fixed versus SSV-dilating chart coverage.

Reuse the original three-frame scales and data without fitting them again.
Only physical-space gate geometry and early-frame overlap regularization vary.
No future labels, PDE residual, learned gate, or moving center enters training.
"""
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
from two_stage import StagedModel

METHODS = ('fixed', 'moving', 'moving_overlap')
ALL_METHODS = (*METHODS, 'fixed_overlap')
OBS = (0, 2, 4)


class MovingChartModel(StagedModel):
    def __init__(self, width=34, gate='fixed'):
        if gate not in ('fixed', 'moving'):
            raise ValueError(gate)
        super().__init__(3, width)
        self.gate = gate

    def gate_widths(self, t):
        alpha = self.dictionary[self.assignment, 0]
        if self.gate == 'fixed':
            return torch.full((self.charts, len(t)), 4., device=t.device, dtype=t.dtype)
        return 4. * (1. + t[None]) ** alpha[:, None]

    def gate_weights(self, x, t):
        if self.gate == 'fixed':
            return self.weights(x)[:, None, :].expand(-1, len(t), -1)
        widths = self.gate_widths(t)
        logits = -.5 * ((x[None, None] - self.centers[:, None, None]) / widths[:, :, None]) ** 2
        logits = logits.masked_fill(self.enabled[:, None, None] == 0, -torch.inf)
        return logits.softmax(dim=0)

    def forward(self, ic, t, x, components=False):
        # Reuse the original experts verbatim; only replace their blend.
        fixed, parts = super().forward(ic, t, x, components=True)
        pred = fixed if self.gate == 'fixed' else (parts * self.gate_weights(x, t)[None]).sum(1)
        return (pred, parts) if components else pred


def overlap_loss(parts, weights, ic, indices=OBS):
    """Pairwise physical disagreement, via the weighted-variance identity.

    parts: B,J,T,X physical fields, weights: J,T,X. Normalize by known IC
    energy, exactly as the data objective; no division by tiny overlap mass.
    """
    if indices is not None:
        parts = parts[:, :, list(indices)]
        weights = weights[:, list(indices)]
    mix = (parts * weights[None]).sum(1, keepdim=True)
    disagreement = ((parts - mix).square() * weights[None]).sum((1, 3))
    return (disagreement / ic.square().sum(-1).clamp_min(1e-12)[:, None]).mean()


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def diagnostic_metrics(model, pred, parts, truth, x, t):
    # Evaluation-only reference energy measures coverage where the signal is.
    w = model.gate_weights(x, t).double()
    p, u, v = pred.double(), parts.double(), truth.double()
    initial_energy = v[:, 0].square().sum(-1).clamp_min(1e-12)
    disagreement = ((u - p[:, None]).square() * w[None]).sum((1, 3))
    overlap = (disagreement / initial_energy[:, None]).mean(0).sqrt()
    true_mass = v[:, 0].sum(-1).abs().clamp_min(1e-12)
    mass = p.sum(-1)
    mass_error = ((mass - v[:, 0].sum(-1)[:, None]).abs() / true_mass[:, None]).mean(0)
    mass_drift = ((mass - mass[:, :1]).abs() / true_mass[:, None]).mean(0)
    energy = v.square().sum(0)
    shares = (w * energy[None]).sum(-1) / energy.sum(-1).clamp_min(1e-12)[None]
    gate_motion = torch.zeros_like(p)
    if model.gate == 'moving':
        alpha = model.dictionary[model.assignment, 0].double()
        widths = model.gate_widths(t).double()
        logit_t = alpha[:, None, None] * (x.double()[None, None] - model.centers.double()[:, None, None]).square()
        logit_t = logit_t / widths[:, :, None].square() / (1. + t.double()[None, :, None])
        wt = w * (logit_t - (w * logit_t).sum(0, keepdim=True))
        gate_motion = (u * wt[None]).sum(1)
    gate_motion_rms = (gate_motion.square().sum(-1) / initial_energy[:, None]).mean(0).sqrt()
    return {
        'overlap_rms_curve': overlap.tolist(),
        'mass_error_curve': mass_error.tolist(),
        'mass_drift_curve': mass_drift.tolist(),
        'chart_energy_shares': shares.tolist(),
        'gate_motion_rms_curve': gate_motion_rms.tolist(),
    }


def evaluate(model, y, ix, labels, x, t, out, method, seed):
    result = {}
    model.eval()
    with torch.no_grad():
        for split in ('test', 'ood'):
            ids = ix[split]
            pred, parts = model(y[ids, 0], t, x, components=True)
            if not torch.isfinite(pred).all():
                raise RuntimeError(f'Nonfinite predictions: {method}, seed {seed}')
            report = metrics(pred, y[ids], x, t)
            report['initial'] = report['curve'][0]
            report['diagnostics'] = diagnostic_metrics(model, pred, parts, y[ids], x, t)
            report['family_metrics'] = {}
            for family in np.unique(labels[ids]):
                keep = np.flatnonzero(labels[ids] == family)
                report['family_metrics'][str(family)] = metrics(pred[keep], y[ids][keep], x, t)
            result[split] = report
            np.savez_compressed(out / f'{method}_{seed}_{split}.npz', prediction=pred.numpy(), ids=ids)
    return result


def representation_diagnostic(x, y, splits, scales):
    """Unrestricted least-squares IC fit; diagnostic, never used for training."""
    model = MovingChartModel().double()
    model.dictionary.copy_(scales.double())
    xx = torch.tensor(x, dtype=torch.float64)
    w = model.weights(xx).detach().numpy()
    basis = []
    for j, c in enumerate(model.centers.tolist()):
        phase = np.pi * (x[:, None] - c) * np.arange(1, 9)[None] / 24
        features = np.concatenate((np.ones((len(x), 1)), np.cos(phase), np.sin(phase)), axis=1)
        basis.append(w[j, :, None] * features)
    design = np.concatenate(basis, axis=1)
    coef, _, rank, singular = np.linalg.lstsq(design, y[:, 0].astype(np.float64).T, rcond=None)
    reconstructed = (design @ coef).T
    report = {'columns': design.shape[1], 'rank': int(rank), 'condition_number': float(singular[0] / singular[-1]),
              'definition': 'Per-IC unrestricted linear fit in the exact t=0 blended Fourier basis; no encoder constraint. Evaluation diagnostic only.'}
    for split in ('train', 'valid', 'test', 'ood'):
        keep = splits == split
        report[split] = float(np.linalg.norm(reconstructed[keep] - y[keep, 0]) / np.linalg.norm(y[keep, 0]))
    return report


def run(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = ROOT / args.scale_source
    scale_config = json.loads((source / 'config.json').read_text())
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        raise FileExistsError(f'Preserving existing results: choose a new --out directory: {out}')
    scales = torch.load(source / 'charts3_scales.pt', map_location='cpu', weights_only=True)
    xx, tt, yy, labels, splits = dataset()
    with np.load(source / 'data.npz') as previous:
        for key, arr in zip(('x', 't', 'u', 'family', 'split'), (xx, tt, yy, labels, splits)):
            np.testing.assert_array_equal(arr, previous[key])
    np.savez_compressed(out / 'data.npz', x=xx, t=tt, u=yy, family=labels, split=splits)
    x, t, y = [torch.tensor(a, dtype=torch.float32) for a in (xx, tt, yy)]
    ix = {split: np.flatnonzero(splits == split) for split in ('train', 'valid', 'test', 'ood')}
    config = vars(args) | {
        'methods': list(args.methods), 'width': 34, 'parameters': 6407, 'gamma': 1, 'centers': [-4, 0, 4],
        'initial_gate_width': 4, 'scales': scales.tolist(), 'scale_fitting': 'reuse frozen source fit',
        'scale_observation_times': scale_config.get('scale_observation_times', np.linspace(0, .5, scale_config.get('scale_frames', 3)).tolist()),
        'scale_sha256': hashlib.sha256((source / 'charts3_scales.pt').read_bytes()).hexdigest(),
        'supervised_times': [0, .25, .5], 'overlap_times': [0, .25, .5],
        'overlap_definition': 'sum_j w_j*(u_j-u_hat)^2 / sum_x u0^2, averaged over ICs and observed times',
        'overlap_lambda_selection': 'fixed in advance; no sweep or future/test selection',
        'checkpoint_selection': 'minimum validation data loss at 0,.25,.5 only; every 10 steps',
        'optimizer': {'name': 'AdamW', 'lr': .003, 'weight_decay': 1e-6, 'clip_norm': 5,
                      'scheduler': 'CosineAnnealingLR', 'eta_min': 1e-5},
        'split_counts': {key: len(value) for key, value in ix.items()},
        'torch': torch.__version__, 'python': platform.python_version(), 'device': 'cpu', 'threads': 1,
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ('moving_charts.py', 'experiment.py', 'two_stage.py')},
    }
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in config['source_sha256']:
        (snapshot / name).write_bytes((ROOT / name).read_bytes())
    diagnostics = {'representation_lower_bound': representation_diagnostic(xx, yy, splits, scales),
                   'posthoc_gate_switch': []}
    (out / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    reports = []
    for seed in range(args.seeds):
        # Construct once and load identical state in all three treatments.
        torch.manual_seed(seed)
        initial = MovingChartModel(width=34, gate='fixed')
        initial.dictionary.copy_(scales)
        initial_state = copy.deepcopy(initial.state_dict())
        initial_digest = state_hash(initial_state)
        for method in args.methods:
            model = MovingChartModel(width=34, gate='fixed' if method.startswith('fixed') else 'moving')
            model.load_state_dict(initial_state)
            assert state_hash(model.state_dict()) == initial_digest
            frozen = model.dictionary.clone()
            coefficient = args.overlap_lambda if method.endswith('_overlap') else 0.
            opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
            best, selected, selected_step = float('inf'), None, None
            history = []
            start = time.perf_counter()
            ic = y[ix['train'], 0]
            early_weights = model.gate_weights(x, t[:5]).detach()
            for step in range(1, args.steps + 1):
                model.train()
                pred, parts = model(ic, t[:5], x, components=True)
                data_value = loss(pred, y[ix['train'], :5], list(OBS))
                ov_value = overlap_loss(parts, early_weights, ic) if coefficient else data_value.new_zeros(())
                objective = data_value + coefficient * ov_value
                if not torch.isfinite(objective):
                    raise RuntimeError(f'Nonfinite objective: {method}, {seed}, step {step}')
                opt.zero_grad()
                objective.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
                scheduler.step()
                if step == 1 or step % 10 == 0 or step == args.steps:
                    with torch.no_grad():
                        valid = float(loss(model(y[ix['valid'], 0], t[:5], x), y[ix['valid'], :5], list(OBS)))
                    history.append({'step': step, 'train': float(objective.detach()), 'data': float(data_value.detach()),
                                    'overlap': float(ov_value.detach()), 'valid': valid})
                    if valid < best:
                        best, selected, selected_step = valid, copy.deepcopy(model.state_dict()), step
                if step % 500 == 0:
                    print('TRAIN', method, seed, step, 'early_valid', valid, flush=True)
            seconds = time.perf_counter() - start
            model.load_state_dict(selected)
            torch.testing.assert_close(model.dictionary, frozen, rtol=0, atol=0)
            assert not model.dictionary.requires_grad
            record = {'method': method, 'seed': seed, 'width': 34,
                      'parameters': sum(p.numel() for p in model.parameters()), 'steps': args.steps,
                      'selected_step': selected_step, 'seconds': seconds, 'initial_state_sha256': initial_digest,
                      'scales': scales.tolist(), 'scales_unchanged': True, 'overlap_lambda': coefficient,
                      'history': history}
            record.update(evaluate(model, y, ix, labels, x, t, out, method, seed))
            torch.save(selected, out / f'{method}_{seed}.pt')
            reports.append(record)
            (out / 'results.json').write_text(json.dumps(reports, indent=2))
            print('DONE', method, seed, 'test_long', record['test']['long'], 'ood_long', record['ood']['long'],
                  'seconds', round(seconds, 1), flush=True)
            if method == 'fixed':
                # Frozen-weights intervention isolates geometry, not a retrained competitor.
                model.gate = 'moving'
                intervention = {'seed': seed, 'description': 'Change gates of the trained fixed model; no optimization'}
                with torch.no_grad():
                    for split in ('test', 'ood'):
                        pred = model(y[ix[split], 0], t, x)
                        intervention[split] = metrics(pred, y[ix[split]], x, t)
                diagnostics['posthoc_gate_switch'].append(intervention)
                (out / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    print('COMPLETE', len(reports), 'runs in', out, flush=True)
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--overlap-lambda', type=float, default=.1)
    parser.add_argument('--scale-source', default='runs/two_stage')
    parser.add_argument('--out', default='runs/moving_charts')
    parser.add_argument('--methods', nargs='+', choices=ALL_METHODS, default=list(METHODS))
    args = parser.parse_args()
    if args.steps < 1 or args.seeds < 1 or args.overlap_lambda < 0:
        parser.error('steps and seeds must be positive; overlap-lambda must be nonnegative')
    if len(set(args.methods)) != len(args.methods):
        parser.error('methods must be unique')
    run(args)
