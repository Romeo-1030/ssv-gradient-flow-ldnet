"""Shared-gamma selection from held-out forecasts in physical time.

The search domain is a user-approved fixed grid motivated by the IC family.
It is not an estimate of a trajectory's true similarity parameter. Selection
uses validation forecasts only; each GF treatment gets its own calibration.
"""
import numpy as np

DEFAULT_CANDIDATES = (.125, .25, .5, 1., 2., 4., 8.)
DEFAULT_WEIGHTS = (1., 2., 3.)
HOLDOUT_TIMES = (.75, 1., 1.25)
FINAL_METHODS = ('latent6', 'fixed1_gf6', 'selected6', 'selected_gf6')
SEARCH_FOR = {'selected6': 'search_ssv', 'selected_gf6': 'search_gf'}
GF_METHODS = ('fixed1_gf6', 'selected_gf6', 'search_gf')


def normalized_weights(weights):
    value = np.asarray(weights, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all() or (value <= 0).any():
        raise ValueError('Supply three positive finite temporal weights')
    value = value / value.max()
    return value / value.sum()


def weighted_score(curve, weights=DEFAULT_WEIGHTS):
    value = np.asarray(curve, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError('Expected three finite nonnegative holdout errors')
    return float(value @ normalized_weights(weights))


def choose_gamma(records, candidates, seeds, weights=DEFAULT_WEIGHTS):
    candidates, seeds = list(candidates), list(seeds)
    if (not candidates or len(set(candidates)) != len(candidates)
            or any(not np.isfinite(g) or g <= 0 for g in candidates)):
        raise ValueError('Candidates must be distinct positive finite numbers')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Calibration seeds must be nonempty and distinct')
    expected = {(float(g), int(s)) for g in candidates for s in seeds}
    if len(records) != len(expected) or {(r['gamma'], r['seed']) for r in records} != expected:
        raise ValueError('Incomplete or duplicate candidate/seed calibration records')
    lookup = {(r['gamma'], r['seed']): r for r in records}
    scores = []
    for gamma in candidates:
        curves = np.array([lookup[gamma, seed]['holdout_curve'] for seed in seeds])
        values = [weighted_score(curve, weights) for curve in curves]
        scores.append({'gamma': float(gamma), 'mean': float(np.mean(values)),
            'sample_sd': float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
            'values': values, 'mean_holdout_curve': curves.mean(0).tolist()})
    chosen = min(scores, key=lambda row: (row['mean'], abs(np.log(row['gamma'])), row['gamma']))
    return {'selected_gamma': chosen['gamma'], 'scores': scores, 'search_seeds': seeds,
        'time_weights': list(weights), 'normalized_time_weights': normalized_weights(weights).tolist(),
        'criterion': 'seed mean of weighted physical-space population relative L2 at held-out times',
        'inner_labels': [0., .25, .5], 'holdout_labels': list(HOLDOUT_TIMES),
        'test_used': False, 'future_labels_after_1_25_used': False}


def select_variants(records, candidates, seeds, weights=DEFAULT_WEIGHTS):
    expected_methods = set(SEARCH_FOR.values())
    if {r['method'] for r in records} != expected_methods:
        raise ValueError('Both GF and no-GF calibration arms are required')
    return {'schema_version': 2, 'independent_selection_per_gf_treatment': True,
        'by_method': {final: choose_gamma([r for r in records if r['method'] == search],
            candidates, seeds, weights) for final, search in SEARCH_FOR.items()}}
