"""Check the archived paper metrics with NumPy and no training."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'experiments/non_gaussian_heat/runs/ordinary_full_comparison_domain16_t10_cuda'
SEEDS = ('0', '1', '2', '3')
TABLE = {
    'latent': ((5.918, .197), (244.727, 30.566), (508.612, 83.792)),
    'latent_gf': ((5.898, .233), (183.365, 43.104), (362.336, 89.888)),
    'ssv': ((4.814, .202), (6.161, .237), (7.623, .406)),
    'ssv_gf': ((4.691, .193), (5.824, .138), (7.209, .261)),
}
FROZEN = {'ssv': (33.705, .188), 'ssv_gf': (33.736, .180)}


def moments(values):
    return float(np.mean(values)), float(np.std(values, ddof=0))


def pooled_percent(prediction, truth):
    return 100 * np.sqrt(np.sum((prediction - truth) ** 2) / np.sum(truth ** 2))


def verify():
    report = json.loads((RUN / 'report_summary.json').read_text(encoding='utf-8'))
    selected = json.loads((RUN.parent / 'ordinary_shared_gamma_domain16_t10_cuda/shared_gamma_selection.json').read_text(encoding='utf-8'))
    freeze = json.loads((RUN / 'diagnostics/frozen_latent/summary.json').read_text(encoding='utf-8'))
    assert report['status'] == 'complete' and report['paired_seeds'] == [0, 1, 2, 3]
    assert selected['selected_gamma'] == 4.0 and selected['completed_fits'] == 21
    assert not selected['test_used'] and not selected['future_labels_after_1_25_used']
    assert freeze['frozen_latent_time'] == 1.25
    times = np.asarray(report['curves']['t'])
    assert times.shape == (81,) and times[-1] == 10
    with np.load(RUN / 'terminal_predictions.npz', allow_pickle=False) as archive:
        truth = archive['truth'].astype(np.float64)
        assert truth.shape == (24, 192)
        for method, expected in TABLE.items():
            curves = 100 * np.asarray([report['curves'][method][seed] for seed in SEEDS])
            final = [pooled_percent(archive[f'{method}_{seed}'], truth) for seed in SEEDS]
            np.testing.assert_allclose(final, curves[:, -1], rtol=3e-7, atol=4e-6)
            actual = [moments(curves[:, times <= 1.25].mean(axis=1)),
                      moments(curves[:, times > 1.25].mean(axis=1)), moments(final)]
            np.testing.assert_allclose(actual, expected, atol=.00051, rtol=0)
            print(f'{method:10s} ' + ' | '.join(f'{mean:.3f} +/- {sd:.3f}%' for mean, sd in actual))
        for method, expected in FROZEN.items():
            values = [pooled_percent(archive[f'{method}_{seed}_frozen'], truth) for seed in SEEDS]
            np.testing.assert_allclose(values, freeze['summary'][method]['frozen']['t10']['values'], rtol=3e-7, atol=4e-6)
            mean, sd = actual = moments(values)
            np.testing.assert_allclose(actual, expected, atol=.00051, rtol=0)
            print(f'{method:10s} frozen t=10: {mean:.3f} +/- {sd:.3f}%')


if __name__ == '__main__':
    verify()
