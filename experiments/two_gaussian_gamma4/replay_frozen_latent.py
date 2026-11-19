"""Run the original frozen-latent diagnostic against the archived checkpoints."""

import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / 'experiments/non_gaussian_heat'
FULL = EXPERIMENT / 'runs/ordinary_full_comparison_domain16_t10_cuda'
GF = EXPERIMENT / 'runs/ordinary_shared_gamma_domain16_t10_cuda'
METHODS = ('ssv', 'ssv_gf')


def replay():
    sys.path.insert(0, str(EXPERIMENT))
    from ordinary_full_comparison_freeze import build_diagnostics

    expected = json.loads((FULL / 'diagnostics/frozen_latent/summary.json').read_text(encoding='utf-8'))
    with TemporaryDirectory(prefix='ssv-frozen-') as folder:
        destination = Path(folder)
        shutil.copyfile(FULL / 'results.json', destination / 'results.json')
        for name in ('data.npz', 'shared_gamma_selection.json'):
            shutil.copyfile(GF / name, destination / name)
        for seed in range(4):
            for source, name in ((FULL, f'ssv_{seed}.pt'), (GF, f'ssv_gf_{seed}.pt'),
                                 (GF, f'ssv_gf_{seed}_test.npz')):
                shutil.copyfile(source / name, destination / name)
        actual = build_diagnostics(destination, methods=METHODS, device='cpu')
    assert actual['completed_fits'] == expected['completed_fits'] == 8
    for method in METHODS:
        for variant in ('normal', 'frozen'):
            reproduced = actual['summary'][method][variant]['t10']
            original = expected['summary'][method][variant]['t10']
            np.testing.assert_allclose(reproduced['values'], original['values'], rtol=2e-5, atol=0.001)
            print(f"{method:6s} {variant:6s}: {reproduced['mean']:.3f} +/- {reproduced['std_ddof0']:.3f}%")


if __name__ == '__main__':
    replay()
