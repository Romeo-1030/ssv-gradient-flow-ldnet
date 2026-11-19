"""Controlled two-Gaussian shapes with a prescribed reference-clock prior.

The reference clock is an auditing quantity, not a selector input or target.
It is sampled first and retained while geometric candidates are rejected.
Consequently geometric acceptance does not reweight the clock distribution,
although the accepted shape and centroid become conditional on that clock.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import numpy as np

from two_gaussian_data import PARAMETER_NAMES, PROTOCOL as BASE_PROTOCOL, solution


SIX = (0, 2, 4, 6, 8, 10)
SPLITS = ('train', 'valid', 'test')
RAW_SHAPE_NAMES = ('p', 'sigma1_raw', 'sigma2_raw', 'distance_raw')
PROTOCOL = {
    'name': 'ic_gamma_controlled_v1',
    'seed': 91,
    'counts': {'train': 36, 'valid': 12, 'test': 24},
    'diffusivity': 1.,
    'mass': 1.,
    'gamma_reference_prior': {
        'distribution': 'truncated_lognormal',
        'median_before_truncation': 1.,
        'log_mean': 0.,
        'log_standard_deviation': math.log(2.),
        'support': [.25, 4.],
        'sampling': 'reject out-of-support normal log draws; never clip',
    },
    'raw_shape': {
        'p': [.25, .75],
        'sigma1': [.7, 1.3],
        'sigma2': [.7, 1.3],
        'distance': [0., 4.],
        'centroid': [-1., 1.],
        'draws': 'independent uniform candidate draws before acceptance',
    },
    'normalization': 'divide both raw widths and distance by sqrt(raw central variance), then multiply by sqrt(2*diffusivity/gamma_reference)',
    'centers': 'c1=C-(1-p)*d; c2=C+p*d',
    'minimum_component_sigma': .5,
    'maximum_exact_tail_mass_at_t6': 1e-7,
    'acceptance': 'retain gamma_reference; resample only shape and centroid until width and exact whole-space tail checks pass',
    'accepted_shape_distribution': 'conditional on gamma_reference; accepted widths and distances are not asserted lognormal',
    'random_streams': 'SeedSequence(seed).spawn(2): independent gamma and geometric streams',
    'maximum_shape_attempts_per_ic': 100000,
    'x': '192 samples on [-24,24), dx=0.25',
    't': '49 samples on [0,6], dt=0.125',
    'labels': [0., .25, .5, .75, 1., 1.25],
    'reference': 'analytic whole-space heat solution; no boundary condition imposed',
    'gamma_reference_meaning': '2*diffusivity/whole-space initial central variance, not a guaranteed loss-optimal clock',
    'hidden_audit_fields': ['parameters', 'gamma_reference', 'raw_shape', 'shape_attempts', 'gamma_draw_attempts'],
    'selector_inputs': 'observed sampled fields and their x/t coordinates only; hidden auditing fields are excluded',
}


def _counts(counts):
    if counts is None:
        return dict(PROTOCOL['counts'])
    if not isinstance(counts, Mapping) or set(counts) - set(SPLITS):
        raise ValueError('counts must map train/valid/test to nonnegative integers')
    result = {}
    for split in SPLITS:
        value = counts.get(split, 0)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError('counts must map train/valid/test to nonnegative integers')
        result[split] = int(value)
    if not sum(result.values()):
        raise ValueError('At least one trajectory is required')
    return result


def _draw_gamma(rng):
    prior = PROTOCOL['gamma_reference_prior']
    low, high = prior['support']
    attempts = 0
    while True:
        attempts += 1
        log_gamma = rng.normal(prior['log_mean'], prior['log_standard_deviation'])
        if math.log(low) <= log_gamma <= math.log(high):
            return math.exp(log_gamma), attempts


def _draw_shape(rng, gamma):
    ranges = PROTOCOL['raw_shape']
    p = rng.uniform(*ranges['p'])
    raw1 = rng.uniform(*ranges['sigma1'])
    raw2 = rng.uniform(*ranges['sigma2'])
    raw_distance = rng.uniform(*ranges['distance'])
    center = rng.uniform(*ranges['centroid'])
    raw_variance = p*raw1**2 + (1-p)*raw2**2 + p*(1-p)*raw_distance**2
    variance = 2*PROTOCOL['diffusivity']/gamma
    scale = math.sqrt(variance/raw_variance)
    sigma1, sigma2, distance = scale*raw1, scale*raw2, scale*raw_distance
    c1, c2 = center-(1-p)*distance, center+p*distance
    row = (PROTOCOL['mass'], p, sigma1, sigma2, center, distance,
           c1, c2, variance, distance/math.sqrt(sigma1**2+sigma2**2))
    return row, (p, raw1, raw2, raw_distance)


def exact_tail_mass(parameters, time=6., extent=24.):
    """Whole-space mass outside [-extent, extent] for one parameter row."""
    row = np.asarray(parameters, dtype=np.float64)
    tail = 0.
    for fraction, sigma, center in ((row[1], row[2], row[6]), (1-row[1], row[3], row[7])):
        scale = math.sqrt(2*(sigma*sigma + 2*PROTOCOL['diffusivity']*time))
        tail += .5*row[0]*fraction*(math.erfc((extent-center)/scale) + math.erfc((extent+center)/scale))
    return tail


def _admissible_shape(row):
    return (min(row[2:4]) >= PROTOCOL['minimum_component_sigma']
            and exact_tail_mass(row) < PROTOCOL['maximum_exact_tail_mass_at_t6'])


def make_dataset(seed=91, counts=None):
    """Return NPZ-compatible arrays, including explicitly hidden provenance.

    ``counts`` can specify any nonempty subset of train/valid/test. Gamma and
    geometric streams are separate so retrying one shape cannot change any
    sampled gamma, including clocks for later trajectories.
    """
    counts = _counts(counts)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if BASE_PROTOCOL['diffusivity'] != PROTOCOL['diffusivity']:
        raise ValueError('Analytic reference diffusivity does not match this protocol')
    gamma_stream, shape_stream = np.random.SeedSequence(int(seed)).spawn(2)
    gamma_rng, shape_rng = np.random.default_rng(gamma_stream), np.random.default_rng(shape_stream)
    parameters, raw_shapes, gammas, splits, identifiers, attempts, gamma_attempts = [], [], [], [], [], [], []
    for split, count in counts.items():
        for index in range(count):
            gamma, gamma_tries = _draw_gamma(gamma_rng)
            for tries in range(1, PROTOCOL['maximum_shape_attempts_per_ic']+1):
                row, raw_shape = _draw_shape(shape_rng, gamma)
                if _admissible_shape(row):
                    break
            else:
                raise RuntimeError(f'No admissible shape for retained gamma {gamma}; seed {seed}, {split} {index}')
            parameters.append(row)
            raw_shapes.append(raw_shape)
            gammas.append(gamma)
            splits.append(split)
            identifiers.append(f'{split}_{index:03d}')
            attempts.append(tries)
            gamma_attempts.append(gamma_tries)
    x, t = np.linspace(-24., 24., 192, endpoint=False), np.linspace(0., 6., 49)
    parameters = np.asarray(parameters, dtype=np.float64)
    return {
        'x': x, 't': t, 'u': solution(x, t, parameters).astype(np.float32),
        'family': np.full(len(parameters), 'two_gaussian'), 'split': np.asarray(splits),
        'parameters': parameters, 'parameter_names': np.asarray(PARAMETER_NAMES),
        'gamma_reference': np.asarray(gammas, dtype=np.float64),
        'sample_id': np.asarray(identifiers),
        'raw_shape': np.asarray(raw_shapes, dtype=np.float64),
        'raw_shape_names': np.asarray(RAW_SHAPE_NAMES),
        'shape_attempts': np.asarray(attempts, dtype=np.int64),
        'gamma_draw_attempts': np.asarray(gamma_attempts, dtype=np.int64),
        'generation_seed': np.asarray(seed, dtype=np.int64),
        'protocol_name': np.asarray(PROTOCOL['name']),
    }


def audit_dataset(data):
    """JSON-compatible geometric, sampled-moment, and per-IC provenance audit."""
    rows = np.asarray(data['parameters'], dtype=np.float64)
    fields = np.asarray(data['u'], dtype=np.float64)
    x, t = np.asarray(data['x']), np.asarray(data['t'])
    dx = float(x[1]-x[0])
    mass = fields.sum(-1)*dx
    if np.any(mass <= 0) or not np.isfinite(fields).all():
        raise ValueError('Reference fields must be finite and have positive sampled mass')
    center = (fields*x).sum(-1)*dx/mass
    variance = (fields*(x[None, None]-center[..., None])**2).sum(-1)*dx/mass
    p = rows[:, 1]
    constructed_variance = p*rows[:, 2]**2 + (1-p)*rows[:, 3]**2 + p*(1-p)*rows[:, 5]**2
    expected_variance = constructed_variance[:, None] + 2*PROTOCOL['diffusivity']*t
    gamma_error = np.abs(2*PROTOCOL['diffusivity']/constructed_variance-data['gamma_reference'])
    six_time = t[list(SIX)]
    centered_time = six_time-six_time.mean()
    slope = (variance[:, list(SIX)]*centered_time).sum(-1)/np.square(centered_time).sum()
    tails = np.asarray([exact_tail_mass(row, float(t[-1])) for row in rows])
    per_ic = []
    for i, row in enumerate(rows):
        per_ic.append({
            'sample_id': str(data['sample_id'][i]), 'split': str(data['split'][i]),
            'gamma_reference': float(data['gamma_reference'][i]),
            'gamma_draw_attempts': int(data['gamma_draw_attempts'][i]),
            'shape_attempts': int(data['shape_attempts'][i]),
            'raw_shape': dict(zip(RAW_SHAPE_NAMES, map(float, data['raw_shape'][i]))),
            'parameters': dict(zip(PARAMETER_NAMES, map(float, row))),
            'minimum_component_sigma': float(min(row[2:4])),
            'exact_tail_mass_at_t6': float(tails[i]),
            'max_sampled_mass_error': float(np.abs(mass[i]-row[0]).max()),
            'sampled_initial_central_variance': float(variance[i, 0]),
            'sampled_six_frame_variance_slope': float(slope[i]),
            'max_sampled_variance_error': float(np.abs(variance[i]-expected_variance[i]).max()),
        })
    return {
        'protocol': copy.deepcopy(PROTOCOL), 'seed': int(data['generation_seed']),
        'counts': {split: int(np.sum(data['split'] == split)) for split in SPLITS},
        'reference_gamma_is_audit_only': True,
        'all_fields_nonnegative_and_finite': bool(np.all(fields >= 0) and np.isfinite(fields).all()),
        'minimum_component_sigma': float(rows[:, 2:4].min()),
        'maximum_exact_tail_mass_at_t6': float(tails.max()),
        'max_sampled_mass_error': float(np.abs(mass-rows[:, :1]).max()),
        'max_initial_variance_construction_error': float(np.abs(constructed_variance-rows[:, 8]).max()),
        'max_gamma_reference_construction_error': float(gamma_error.max()),
        'max_sampled_variance_error': float(np.abs(variance-expected_variance).max()),
        'max_six_frame_variance_slope_error': float(np.abs(slope-2*PROTOCOL['diffusivity']).max()),
        'gamma_reference_range': [float(data['gamma_reference'].min()), float(data['gamma_reference'].max())],
        'shape_attempts_total': int(data['shape_attempts'].sum()),
        'shape_rejections_total': int((data['shape_attempts']-1).sum()),
        'per_ic': per_ic,
    }
