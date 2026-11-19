"""Reuse the original independently sampled two-Gaussian ICs on domain16.

No widths, separation, variance, split, or mixture weights are changed. Only
the observation grid and evaluation horizon change. All shape-example choices
are computed from initial fields before any model predictions are available.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np

import two_gaussian_data as original


SIX = (0, 2, 4, 6, 8, 10)
SPLITS = ('train', 'valid', 'test')
PROTOCOL_NAME = 'ordinary_independent_two_gaussian_domain16_t10'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def _tail_mass(row, time=10., extent=16.):
    total = 0.
    for fraction, sigma, center in ((row[1], row[2], row[6]), (1-row[1], row[3], row[7])):
        scale = math.sqrt(2*(sigma*sigma+2*time))
        total += row[0]*fraction*.5*(math.erfc((extent-center)/scale)+math.erfc((extent+center)/scale))
    return total


def _initial_shape_audit(data):
    # This uses initial fields only. The fine grid prevents a missed shallow
    # peak merely because the 192-point training grid is coarser.
    x = np.linspace(-16., 16., 8193)
    initial = original.solution(x, np.array([0.]), data['parameters'])[:, 0]
    records = []
    split_indices = {s: 0 for s in SPLITS}
    for index, (values, split, sample_id) in enumerate(zip(initial, data['split'], data['sample_id'])):
        peaks = np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))+1
        split_index = split_indices[str(split)]
        split_indices[str(split)] += 1
        row = {'sample_id': str(sample_id), 'split': str(split), 'global_index': index,
               'index_in_split': split_index, 'local_maxima_count': int(len(peaks)),
               'peak_positions': x[peaks].tolist(), 'peak_heights': values[peaks].tolist(),
               'bimodal': bool(len(peaks) == 2), 'clearly_bimodal': False}
        if len(peaks) == 2:
            left, right = peaks
            valley = left+int(np.argmin(values[left:right+1]))
            valley_ratio = float(values[valley]/min(values[left], values[right]))
            peak_ratio = float(min(values[left], values[right])/max(values[left], values[right]))
            row.update(valley_position=float(x[valley]), valley_height=float(values[valley]),
                       valley_to_smaller_peak_ratio=valley_ratio,
                       smaller_to_larger_peak_ratio=peak_ratio,
                       peak_separation=float(x[right]-x[left]),
                       clearly_bimodal=bool(valley_ratio <= .8 and peak_ratio >= .1))
        records.append(row)
    test = [r for r in records if r['split'] == 'test']
    bimodal = [r for r in test if r['bimodal']]
    if not bimodal:
        raise ValueError('The ordinary test split unexpectedly has no bimodal initial field')
    # A predetermined morphology rule, independent of any future error.
    eligible = [r for r in bimodal if r['clearly_bimodal']] or bimodal
    selected = min(eligible, key=lambda r: (r['valley_to_smaller_peak_ratio'],
                   -r['peak_separation'], r['sample_id']))
    return {'diagnostic_time': 0., 'diagnostic_grid': '8193 points on [-16,16], endpoints included',
        'peak_definition': 'strict left and non-strict right local maximum on fine sampled initial field',
        'clear_bimodality_definition': 'two maxima, valley/smaller peak <=0.8, smaller/larger peak >=0.1',
        'example_selection_rule': 'Among clearly bimodal test initial fields, minimize valley/smaller peak; tie-break by greater peak separation then sample_id. If none are clear, use all bimodal initial fields.',
        'example_selection_uses_predictions_or_future_fields': False,
        'counts_by_split': {s: {'total': sum(r['split'] == s for r in records),
            'bimodal': sum(r['split'] == s and r['bimodal'] for r in records),
            'clearly_bimodal': sum(r['split'] == s and r['clearly_bimodal'] for r in records)} for s in SPLITS},
        'test_bimodal_count': len(bimodal),
        'test_clear_bimodal_count': sum(r['clearly_bimodal'] for r in test),
        'selected_test_example': selected,
        'test_014_shape': next(r for r in test if r['sample_id'] == 'test_014'),
        'records': records}


def prepare_data(source_run):
    """Return (previous, data, audit), without writing or changing source data.

    ``source_run`` must contain the exact original seed71, 36/12/24 dataset.
    All original arrays are checked against two_gaussian_data.make_data().
    Mixture parameters remain audit-only; shared-gamma selection receives only
    the six observed train/validation fields through its separate interface.
    """
    source_run = Path(source_run).resolve()
    source_file = source_run/'data.npz'
    previous = _load(source_file)
    regenerated = original.make_data()
    if set(previous) != set(regenerated):
        raise ValueError('Expected the original ordinary two-Gaussian dataset schema')
    for key in regenerated:
        np.testing.assert_array_equal(previous[key], regenerated[key], err_msg=f'Original seed71 mismatch: {key}')
    parameters = previous['parameters'].copy()
    splits = previous['split'].copy()
    assert parameters.shape == (72, 10)
    assert original.PROTOCOL['diffusivity'] == 1.
    counts = {s: int(np.sum(splits == s)) for s in SPLITS}
    assert counts == {'train': 36, 'valid': 12, 'test': 24}
    sample_ids, counters = [], {s: 0 for s in SPLITS}
    for split in splits:
        sample_ids.append(f'{split}_{counters[str(split)]:03d}')
        counters[str(split)] += 1
    x = np.linspace(-16., 16., 192, endpoint=False, dtype=np.float64)
    t = np.arange(81, dtype=np.float64)/8
    data = {'x': x, 't': t, 'u': original.solution(x, t, parameters).astype(np.float32),
        'family': previous['family'].copy(), 'split': splits, 'parameters': parameters,
        'parameter_names': previous['parameter_names'].copy(), 'sample_id': np.asarray(sample_ids),
        'protocol_name': np.asarray(PROTOCOL_NAME)}
    assert data['u'].shape == (72, 81, 192)
    assert np.all(np.isfinite(data['u'])) and np.all(data['u'] >= 0)
    assert not {'gamma_reference', 'raw_shape', 'shape_attempts', 'gamma_draw_attempts'} & set(data)
    for key in ('parameters', 'split', 'family', 'parameter_names'):
        np.testing.assert_array_equal(data[key], previous[key])

    fields = data['u'].astype(np.float64)
    dx = float(x[1]-x[0])
    sampled_mass = fields.sum(-1)*dx
    center = (fields*x).sum(-1)*dx/sampled_mass
    variance = (fields*(x[None, None]-center[:, :, None])**2).sum(-1)*dx/sampled_mass
    tails = np.array([_tail_mass(row) for row in parameters])
    edge = original.solution(np.array([-16., 16.]), np.array([10.]), parameters)[:, 0].max(-1)
    peak = original.solution(np.linspace(-16., 16., 8193), np.array([10.]), parameters)[:, 0].max(-1)
    phase = np.pi*x[:, None]*np.arange(1, 17)[None, :]/16
    basis = np.column_stack([np.ones(len(x)), np.cos(phase), np.sin(phase)])
    initial = fields[:, 0]
    projected = (basis@np.linalg.lstsq(basis, initial.T, rcond=None)[0]).T
    floors, per_ic_floor = {}, 100*np.sqrt(np.square(projected-initial).sum(-1)/np.square(initial).sum(-1))
    for split in SPLITS:
        mask = splits == split
        floors[split] = float(100*np.sqrt(np.square(projected[mask]-initial[mask]).sum()/np.square(initial[mask]).sum()))
    shape_audit = _initial_shape_audit(data)
    protocol = copy.deepcopy(original.PROTOCOL)
    protocol.update(name=PROTOCOL_NAME, x='192 equally spaced samples on [-16,16), dx=1/6',
        t='81 samples on [0,10], dt=0.125', labels=t[list(SIX)].tolist(),
        reference='Whole-space analytic heat solution restricted to observation window; no zero or periodic PDE boundary imposed',
        data_reuse='Exact original seed71 parameters and 36/12/24 split, verified against independent-uniform generator',
        normalization='None: no total-variance rescaling or reference-clock prior',
        acceptance='All original ICs retained; no rejection, resampling, or shape filtering',
        future_scan='No future or prediction error used to choose displayed examples',
        audit_fields=['parameters', 'parameter_names'],
        selector_inputs='Only six observed train/validation fields and their coordinates; no component parameters',
        shared_gamma='A single dictionary-selected clock; not an IC-specific reference clock')
    audit = {'protocol': protocol, 'fingerprint': original.fingerprint(data),
        'source_data_path': str(source_file), 'source_data_sha256': _sha(source_file),
        'source_fingerprint': original.fingerprint(previous),
        'source_generator_sha256': _sha(Path(original.__file__)),
        'preparation_source_sha256': _sha(Path(__file__)),
        'source_matches_original_make_data_exactly': True,
        'IC_parameters_and_splits_unchanged': True, 'spatial_fields_resampled': True,
        'parameter_shaping_or_rejection_applied': False, 'counts': counts,
        'all_fields_nonnegative_and_finite': True,
        'maximum_exact_mass_outside_domain_at_t10': float(tails.max()),
        'median_exact_mass_outside_domain_at_t10': float(np.median(tails)),
        'mean_exact_mass_outside_domain_at_t10': float(tails.mean()),
        'maximum_boundary_to_peak_ratio_at_t10': float(np.max(edge/peak)),
        'maximum_sampled_mass_error': float(np.max(abs(sampled_mass-parameters[:, :1]))),
        'maximum_initial_variance_error': float(np.max(abs(variance[:, 0]-parameters[:, 8]))),
        'initial_K16_projection_floor_percent': floors,
        'per_ic_initial_K16_projection_floor_percent': dict(zip(sample_ids, map(float, per_ic_floor))),
        'per_ic_t10_tail_mass': [{'sample_id': str(i), 'mass': float(v)} for i, v in zip(sample_ids, tails)],
        'initial_shape_audit': shape_audit,
        'representative_sample_id': shape_audit['selected_test_example']['sample_id'],
        'parameter_realized_ranges': {str(name): [float(parameters[:, i].min()), float(parameters[:, i].max())]
            for i, name in enumerate(data['parameter_names'])}}
    return previous, data, audit
