"""Select one frozen gamma with temporary domain16 LDNet forecasts.

The public entry point accepts exactly six observed train/validation fields.
Temporary LDNet fits use the first three; their checkpoints also use only the
first three validation fields. The later three validation fields rank the
candidate clocks with the historical 1:2:3 weighted, pooled-current error.
There is no GF, continuous gamma refinement, or test/future-field input here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from forecast_gamma_model import EARLY, HOLDOUT, checkpoint_score, data_loss, holdout_score
from gamma_selection import choose_gamma, weighted_score
from ic_gamma_domain16_model import Domain16GammaModel, pack_inputs
from ic_gamma_experiment import atomic_torch_save
from moving_charts import state_hash
from original_scale import cpu_state, sync
from original_scale_model import ORIGINAL_SOURCE
from retry_json import save_json


DEFAULT_CANDIDATES = (.125, .25, .5, 1., 2., 4., 8.)
DEFAULT_SEEDS = (200, 201, 202)
TIME_WEIGHTS = (1., 2., 3.)


def _digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _observations(observed_train, observed_valid, x, t_observed, device):
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=device).detach().clone()

    train, valid, x, times = map(tensor, (observed_train, observed_valid, x, t_observed))
    for name, value in [('train', train), ('valid', valid)]:
        if value.ndim != 3 or value.shape[1:] != (6, 192) or not len(value):
            raise ValueError(f'{name}: expected nonempty B-by-6-by-192 observed fields only')
        if not torch.isfinite(value).all():
            raise ValueError(f'{name}: observed fields must be finite')
        if (value[:, 0].square().sum(-1) <= 0).any():
            raise ValueError(f'{name}: initial fields must have nonzero L2 energy')
    expected_x = torch.arange(192, device=device, dtype=torch.float32)*(32/192)-16
    expected_t = torch.arange(6, device=device, dtype=torch.float32)/4
    if x.shape != (192,) or not torch.allclose(x, expected_x, rtol=0, atol=2e-6):
        raise ValueError('Expected 192 samples on [-16,16), without endpoint')
    if times.shape != (6,) or not torch.equal(times, expected_t):
        raise ValueError('Only the six observed times 0,.25,.5,.75,1,1.25 are accepted')
    return train, valid, x, times


def calibrate_shared(observed_train, observed_valid, x, t_observed, out,
                     device='cuda:0', steps=3000, seeds=DEFAULT_SEEDS,
                     candidates=DEFAULT_CANDIDATES, progress_callback=None,
                     resume=True):
    """Return a shared clock selected without any test or post-1.25 truth.

    Inputs can be NumPy arrays or tensors, and are copied/detached. Completed
    candidate/seed fits resume after input/config/checkpoint integrity checks;
    an interrupted, incomplete fit restarts from its original seed. Progress
    callbacks receive one JSON-serializable dictionary. Scores are fractions,
    not percentages, matching gamma_selection.choose_gamma.
    """
    if isinstance(steps, bool) or int(steps) != steps or steps < 1:
        raise ValueError('steps must be a positive integer')
    steps = int(steps)
    raw_seeds = tuple(seeds)
    if not raw_seeds or any(isinstance(s, bool) or int(s) != s or s < 0 for s in raw_seeds):
        raise ValueError('seeds must be nonnegative integers')
    seeds = tuple(map(int, raw_seeds))
    candidates = tuple(map(float, candidates))
    if len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct')
    if not candidates or len(set(candidates)) != len(candidates) or any(
            not np.isfinite(g) or g <= 0 for g in candidates):
        raise ValueError('candidates must be distinct, positive and finite')
    if len({float(np.float32(g)) for g in candidates}) != len(candidates):
        raise ValueError('candidates must remain distinct in model float32 arithmetic')
    device = torch.device(device)
    train, valid, x, observed_times = _observations(observed_train, observed_valid, x, t_observed, device)
    dense_t = torch.arange(11, device=device, dtype=torch.float32)/8
    early_train, early_valid, later_valid = train[:, :3], valid[:, :3], valid[:, 3:]
    data_hash = state_hash({'train_six': train, 'valid_six': valid, 'x': x, 'observed_t': observed_times})
    source_dir = Path(__file__).resolve().parent
    sources = {name: _digest_file(source_dir/name) for name in (
        Path(__file__).name, 'forecast_gamma_model.py', 'gamma_selection.py',
        'ic_gamma_domain16_model.py', 'ic_gamma_model.py', 'original_scale_model.py',
        'experiment.py')}
    sources['preserved_train.py'] = _digest_file(ORIGINAL_SOURCE)
    identity = {'schema_version': 1, 'steps': steps, 'seeds': list(seeds),
        'candidates': list(candidates), 'time_weights': list(TIME_WEIGHTS),
        'observations_sha256': data_hash, 'source_sha256': sources,
        'train_count': len(train), 'valid_count': len(valid),
        'model': 'ic_gamma_domain16_model.Domain16GammaModel', 'parameters': 91329,
        'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': 1e-6,
                      'scheduler': 'cosine', 'eta_min': 1e-5, 'clip': 5.},
        'training_times': [0., .25, .5], 'checkpoint_times': [0., .25, .5],
        'holdout_times': [.75, 1., 1.25], 'GF': False, 'continuous_refinement': False,
        'data_loss': 'historical per-IC initial-energy-normalized squared error',
        'checkpoint_criterion': 'historical per-IC initial-L2-normalized mean error',
        'candidate_score': '1:2:3 weighted pooled-current relative L2; seed mean',
        'torch_version': str(torch.__version__)}
    out = Path(out)
    calibration = out/'shared_gamma_calibration'
    calibration.mkdir(parents=True, exist_ok=True)
    config_path, records_path = calibration/'config.json', calibration/'records.json'
    if config_path.exists():
        if not resume:
            raise FileExistsError('Calibration output already exists; use a new output or resume=True')
        if json.loads(config_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('Resume configuration, source, or observed data changed')
    else:
        if records_path.exists():
            raise ValueError('Existing calibration records have no configuration')
        save_json(config_path, identity)
    records = json.loads(records_path.read_text(encoding='utf-8')) if records_path.exists() else []
    expected = {(g, s) for g in candidates for s in seeds}
    keys = [(r['gamma'], r['seed']) for r in records]
    if len(set(keys)) != len(keys) or not set(keys) <= expected:
        raise ValueError('Unexpected or duplicate resumed calibration records')
    for row in records:
        checkpoint_path = calibration/row['checkpoint']
        if not checkpoint_path.is_file() or _digest_file(checkpoint_path) != row['checkpoint_sha256']:
            raise ValueError('A completed calibration checkpoint is missing or changed')
        if not np.isclose(weighted_score(row['holdout_curve'], TIME_WEIGHTS), row['holdout_score'], rtol=0, atol=1e-12):
            raise ValueError('A resumed calibration score does not match its holdout curve')
    expected_initial = {}
    for row in records:
        known = expected_initial.setdefault(row['seed'], row['initial_parameters_sha256'])
        if known != row['initial_parameters_sha256']:
            raise ValueError('Candidate models did not share the same seeded initialization')

    def notify(status, **fields):
        update = {'phase': 'shared_gamma_calibration', 'status': status,
                  'completed_fits': len(records), 'total_fits': len(expected), **fields}
        save_json(calibration/'progress.json', update)
        if progress_callback is not None:
            progress_callback(update)

    invocation_start = time.perf_counter()
    resumed_count = len(records)
    for candidate_index, gamma in enumerate(candidates):
        inputs = {name: pack_inputs(fields[:, 0], torch.full((len(fields),), gamma,
                    device=device, dtype=torch.float32)) for name, fields in [('train', train), ('valid', valid)]}
        for seed in seeds:
            if (gamma, seed) in keys:
                continue
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
            model = Domain16GammaModel(physical=False).to(device)
            initial_hash = state_hash(dict(model.named_parameters()))
            known = expected_initial.setdefault(seed, initial_hash)
            if known != initial_hash:
                raise RuntimeError('Candidate models have different initial parameters at the same seed')
            scales = model.get_scales().clone()
            backbone = list(model.parameters())
            optimizer = torch.optim.AdamW(backbone, lr=.001, weight_decay=1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=1e-5)
            best, selected_step, best_state, history = float('inf'), None, None, []
            sync(device)
            fit_start = time.perf_counter()
            notify('training', gamma=gamma, seed=seed, step=0, steps=steps)
            print('SHARED SEARCH START', gamma, seed, flush=True)
            for step in range(1, steps+1):
                model.train()
                objective = data_loss(model, inputs['train'], early_train, x, dense_t, EARLY)
                if not torch.isfinite(objective):
                    raise RuntimeError(f'Nonfinite shared calibration loss: gamma={gamma}, seed={seed}, step={step}')
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                nn.utils.clip_grad_norm_(backbone, 5.)
                optimizer.step()
                scheduler.step()
                if step == 1 or step % 10 == 0 or step == steps:
                    model.eval()
                    with torch.no_grad():
                        score = float(checkpoint_score(model, inputs['valid'], early_valid, x, dense_t, EARLY))
                    if not np.isfinite(score):
                        raise RuntimeError('Nonfinite early validation score')
                    history.append({'step': step, 'train': float(objective.detach()), 'valid': score})
                    if score < best:
                        best, selected_step, best_state = score, step, cpu_state(model)
                if step == 1 or step % 250 == 0 or step == steps:
                    sync(device)
                    notify('training', gamma=gamma, seed=seed, step=step, steps=steps,
                           seconds=time.perf_counter()-fit_start, best_validation=best)
                    save_json(calibration/f'candidate{candidate_index}_{seed}_history.json', history)
                    print('SHARED SEARCH', gamma, seed, step, 'best_valid', best, flush=True)
            model.load_state_dict(best_state)
            model.eval()
            model.assert_geometry()
            torch.testing.assert_close(model.get_scales(), scales, rtol=0, atol=0)
            assert not any('gamma' in name for name, _ in model.named_parameters())
            with torch.no_grad():
                replay_valid = float(checkpoint_score(model, inputs['valid'], early_valid, x, dense_t, EARLY))
                uniform, curve = holdout_score(model, inputs['valid'], later_valid, x, dense_t)
            if not np.isclose(replay_valid, best, rtol=0, atol=1e-7):
                raise RuntimeError('Selected checkpoint did not replay its validation score')
            score = weighted_score(curve, TIME_WEIGHTS)
            sync(device)
            seconds = time.perf_counter()-fit_start
            filename = f'candidate{candidate_index}_{seed}.pt'
            checkpoint = {'model': best_state, 'gamma': gamma, 'seed': seed,
                'selected_step': selected_step, 'model_class': identity['model'],
                'spatial_domain': [-16., 16.], 'observations_sha256': data_hash,
                'gamma_frozen': True, 'GF': False}
            atomic_torch_save(checkpoint, calibration/filename)
            row = {'gamma': gamma, 'seed': seed, 'candidate_index': candidate_index,
                'steps': steps, 'selected_step': selected_step, 'best_validation': best,
                'holdout_curve': curve, 'holdout_score': score, 'unweighted_holdout_score': uniform,
                'seconds': seconds, 'initial_parameters_sha256': initial_hash,
                'checkpoint': filename, 'checkpoint_sha256': _digest_file(calibration/filename),
                'backbone_parameters': 91329, 'GF': False, 'gamma_frozen': True,
                'label_indices': list(EARLY), 'holdout_indices': list(HOLDOUT),
                'history': history}
            records.append(row)
            keys.append((gamma, seed))
            save_json(records_path, records)
            notify('fit_complete', gamma=gamma, seed=seed, score=score, seconds=seconds)
            print('SHARED SEARCH DONE', gamma, seed, 'weighted_valid', score, flush=True)
            del model, optimizer, scheduler, best_state, checkpoint, backbone
    selection = choose_gamma(records, candidates, seeds, TIME_WEIGHTS)
    selection.update(candidates=list(candidates), seconds=float(sum(r['seconds'] for r in records)),
        invocation_seconds=time.perf_counter()-invocation_start, resumed_fits=resumed_count,
        steps_per_fit=steps, completed_fits=len(records), calibration_records=records,
        calibration_directory=str(calibration.resolve()), observations_sha256=data_hash,
        selector_model=identity['model'], training_GF=False, continuous_refinement=False,
        final_gamma_frozen=True, test_time_information='initial field only; shared gamma comes from train/validation',
        comparison_scope='complete selection pipelines; selector architecture and calibration unit differ from per-IC ordinary MLP')
    save_json(out/'shared_gamma_selection.json', selection)
    notify('complete', selected_gamma=selection['selected_gamma'], seconds=selection['seconds'])
    return selection
