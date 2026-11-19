# Two-Gaussian heat experiment

This experiment uses shared γ=4, a horizon of 10, six observed frames through
1.25, and matched optimizer seeds 0–3. The four methods cross SSV and GF.
Table 2 also reports the counterfactual that holds the trained SSV latent state
at 1.25 while continuing the inverse SSV transform.

From the repository root, verify the archived terminal predictions and
the per-seed time curves and regenerate the archived figures.

```sh
cd experiments/two_gaussian_gamma4
python -m pip install -r requirements.txt
python verify_archived_results.py
python deliveries/overleaf_non_gaussian_2026-11-10/plot_appendix.py
```

The plotter uses the archived seed-0 SSV+GF prediction on `test_001`.
Figure 4 crops the panels at t=0.5, 2, and 10 from
`deliveries/overleaf_non_gaussian_2026-11-10/non_gaussian_profiles.pdf`.
The plotter saves all six profile panels and the error curves under its
`project/figures` directory.

## Checkpoint replay and retraining

Use the separate `non_gaussian_gamma4_t10_repro.zip` supplemental archive
for checkpoint replay and training. It contains the exact archived inputs,
checkpoint files, provenance snapshots and training README. The SHA-256 of
the ZIP supplied with the paper is
`5464c3a10ea2f3e9f2983b2fc842b4ec4afb1c719eb6795ede382a33a49191db`.
From the extracted archive root, run:

```sh
sha256sum -c SHA256SUMS
python -m pip install -r requirements.txt
python replay_frozen_latent.py
```

The archive's README gives commands for production training with fresh output
directories. Use Python 3.12 and the original CUDA environment for the paired
initialization checks. The original trainer snapshot and license are also in
this directory, alongside the core source for the final protocol. The archive
carries the additional files hashed by the historical training entry points.

Both GF arms use the detached pre-update GF loss at step 101 as a fixed
normalizer with a floor of 1e-12. The weight is zero through step 100, ramps
linearly from step 101, and reaches 1e-4 at step 200. Collocation uses 24 spatial
points per trajectory at each of 32 predicted rollout states. The times are
0, 0.25, 0.5 and 29 uniformly spaced targets through 10, rounded up to the
evaluation grid. The physical-coordinate GF arm uses the physical time
derivative and the field and spatial derivative in physical coordinates.
