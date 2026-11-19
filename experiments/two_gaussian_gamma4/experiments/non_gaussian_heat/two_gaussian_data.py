"""Fixed, reproducible two-Gaussian main experiment (whole-space heat).

Mixture parameters are stored for auditing, never supplied to either network.
The only network input is the complete sampled initial field.
"""
import hashlib
import math

import numpy as np


PROTOCOL = {
    'seed': 71, 'diffusivity': 1., 'mass': 1.,
    'mass_fraction': [.25, .75], 'sigma1': [.5, 1.5], 'sigma2': [.5, 1.5],
    'centroid': [-1., 1.], 'center_distance': [0., 6.],
    'draws': 'independent uniform draws; sigma1 and sigma2 are independent',
    'centers': 'c1=C-(1-p)*d; c2=C+p*d (mass centroid C)',
    'counts': {'train': 36, 'valid': 12, 'test': 24},
    'x': '192 equally spaced samples on [-24,24), dx=0.25',
    't': '49 equally spaced samples on [0,6], dt=0.125',
    'labels': [0., .25, .5],
    'reference': 'analytic whole-space heat solution; sampled observation window, no boundary condition imposed',
    'input': '192 samples of the total mixture at t=0; no mixture parameters or component labels',
    'future_scan': 'no controlled separation sweep or OOD family in this main experiment',
}
PARAMETER_NAMES = ('mass', 'p', 'sigma1', 'sigma2', 'centroid', 'distance',
                   'c1', 'c2', 'variance0', 'dimensionless_distance')


def solution(x, t, parameters):
    """Normalized Gaussian mixtures, each variance increasing by 2*kappa*t."""
    p = np.asarray(parameters, dtype=np.float64)
    variance1 = p[:, 2, None] ** 2 + 2 * PROTOCOL['diffusivity'] * t[None]
    variance2 = p[:, 3, None] ** 2 + 2 * PROTOCOL['diffusivity'] * t[None]
    first = np.exp(-.5 * (x[None, None] - p[:, 6, None, None]) ** 2 / variance1[:, :, None])
    second = np.exp(-.5 * (x[None, None] - p[:, 7, None, None]) ** 2 / variance2[:, :, None])
    first /= np.sqrt(2 * np.pi * variance1[:, :, None])
    second /= np.sqrt(2 * np.pi * variance2[:, :, None])
    return p[:, 0, None, None] * (p[:, 1, None, None] * first + (1-p[:, 1, None, None]) * second)


def make_data():
    rng = np.random.default_rng(PROTOCOL['seed'])
    parameters, splits = [], []
    for split, count in PROTOCOL['counts'].items():
        for _ in range(count):
            p = rng.uniform(*PROTOCOL['mass_fraction'])
            sigma1 = rng.uniform(*PROTOCOL['sigma1'])
            sigma2 = rng.uniform(*PROTOCOL['sigma2'])
            centroid = rng.uniform(*PROTOCOL['centroid'])
            distance = rng.uniform(*PROTOCOL['center_distance'])
            c1, c2 = centroid-(1-p)*distance, centroid+p*distance
            variance = p*sigma1**2 + (1-p)*sigma2**2 + p*(1-p)*distance**2
            delta = distance / math.sqrt(sigma1**2+sigma2**2)
            parameters.append((1., p, sigma1, sigma2, centroid, distance, c1, c2, variance, delta))
            splits.append(split)
    x, t = np.linspace(-24, 24, 192, endpoint=False), np.linspace(0, 6, 49)
    parameters = np.asarray(parameters)
    return {'x': x, 't': t, 'u': solution(x, t, parameters).astype(np.float32),
            'family': np.full(len(parameters), 'two_gaussian'), 'split': np.asarray(splits),
            'parameters': parameters, 'parameter_names': np.asarray(PARAMETER_NAMES)}


def fingerprint(data):
    digest = hashlib.sha256()
    for key, value in sorted(data.items()):
        array = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def boundary_report(data):
    """Exact mass outside [-24,24] at t=6; all centers lie inside the window."""
    tails = []
    for row in data['parameters']:
        tail = 0.
        for mass, sigma, center in ((row[1], row[2], row[6]), (1-row[1], row[3], row[7])):
            scale = math.sqrt(2*(sigma*sigma+2*PROTOCOL['diffusivity']*data['t'][-1]))
            tail += .5*mass*(math.erfc((24-center)/scale)+math.erfc((24+center)/scale))
        tails.append(tail)
    return {'maximum_exact_mass_outside_window_at_t6': float(max(tails)),
            'mean_exact_mass_outside_window_at_t6': float(np.mean(tails)),
            'all_reference_fields_positive': bool(np.all(data['u'] >= 0)),
            'max_sampled_mass_error': float(np.max(np.abs(data['u'].astype(np.float64).sum(-1)*.25-1)))}
