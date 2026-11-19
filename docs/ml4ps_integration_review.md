# ML4PS 2026 reproducibility review

## Experiment provenance

| Protocol | Observations | Evaluation grid | Seeds and SD | Conditioning and scale |
| --- | --- | --- | --- | --- |
| Single-Gaussian final3000 | Three labels through 0.5 | 256 spatial points, 201 times through 10 | Five matched seeds, population SD | Gaussian parameters; learned trajectory-specific or oracle SSV |
| Two-Gaussian camera-ready candidate | Six labels through 1.25 | 192 points on [−16,16), 81 times through 10 | Four matched seeds, population SD | Sampled initial field; shared γ=4, α=β=0.5 |
| November 8 collaborator ZIP, weighted γ run | Six labels through 1.25 | 192 points on [−24,24), 49 times through 6 | Five matched seeds, sample SD | Sampled initial field; γ=2 selected separately with and without GF, plus a fixed γ=1 GF control |

The camera-ready candidate uses 36 training, 12 validation and 24 test fields
with independent mixture draws. Its data fingerprint starts with
`195d7ad6`; the validated collaborator run starts with `649a604f`. Both use
seed 71 for the initial draws. The newer run selects one rate using seven
candidates and three calibration seeds, then reuses it for both SSV arms.
The controlled-IC γ=4 archive is a historical integrity reference used by the
training entry point. Its data and scores have their own provenance.
The earlier ZIP's validation recorded 599 manifest files, 20 recomputed final
predictions and 42 calibration records.

## Validation

All 142 distributed checksum entries passed. The archived verifier recomputed
terminal errors from prediction arrays and window averages from the 81-point
curves. Entries below are percent mean ± population SD.

| Two-Gaussian arm | Observed through 1.25 | Extrapolation after 1.25 | Final at 10 |
| --- | ---: | ---: | ---: |
| LDNet | 5.918 ± 0.197 | 244.727 ± 30.566 | 508.612 ± 83.792 |
| LDNet + GF | 5.898 ± 0.233 | 183.365 ± 43.104 | 362.336 ± 89.888 |
| LDNet + SSV | 4.814 ± 0.202 | 6.161 ± 0.237 | 7.623 ± 0.406 |
| LDNet + SSV + GF | 4.691 ± 0.193 | 5.824 ± 0.138 | 7.209 ± 0.261 |

The eight CPU checkpoint replays reproduced final frozen-latent errors of
33.705 ± 0.188% (SSV) and 33.736 ± 0.180% (SSV + GF). Frozen Latent keeps the
trained state at 1.25 and continues decoding at the time-dependent scaled
coordinates. A Frozen Profile would instead hold an observed transformed field
fixed and needs its own interpolation, extrapolation and boundary conventions.
The November 8 validator contains an implementation for γ=1 and γ=2 on the
older grid. A γ=4 implementation and its camera-ready reference must be checked
before claiming a γ=4 Frozen Profile result.

The delivered plotting script regenerated the error curves and the six-panel
profile figure. The final manuscript's Table 2 matched all six mean/SD pairs
recomputed from the saved terminal predictions. Figure 4 uses the 0.5, 2 and 10
panels for `test_001`, seed 0, in the full-rollout SSV + GF arm. Its checkpoint
was selected at step 3000. CPU replay matched the archived prediction field
within 1.79e-7 maximum absolute difference.

Single-Gaussian CPU inference regenerated the data and matched the complete
201-point seed-0 curve with a maximum absolute difference of 3.21e-8 in
fractional RelRMSE. Observed, extrapolation and terminal errors were 1.9758%,
2.7504% and 3.7990%. Reaggregation of all 25 archived curves matched the supplied
Table 1 at its printed precision, and the two-panel plot regenerated.
A one-step CPU training smoke test also passed.
The transform test passed when called directly because pytest was absent
from the local environment. The public checkpoint covers one of the 25 runs. Exact scale pretraining remains limited by the
missing original optimizer and initialization source, as documented in the
single-Gaussian audit.

## Proposed integration

1. Keep the tracked single-Gaussian trainer, data generator, scale and
   seed-0 checkpoints, summaries and figures. Preserve the original inference
   and training entry points. Link the final paper when its public URL is ready.
2. Add a dedicated two-Gaussian workflow under `experiments/non_gaussian_heat/`.
   Start from `ordinary_shared_gamma_domain16.py`,
   `ordinary_full_comparison.py`, the data generator and their import
   dependencies. Keep the upstream trainer snapshot and its license because
   runtime provenance checks use its hash. The current snapshot logic also
   hashes the Windows launchers. Make that metadata portable and verify
   the resulting source checks before excluding them. Preserve the frozen-latent replay,
   archive verifier and plotter. Review the dependency closure before removing
   exploratory scripts from the proposed public file set.
3. Track the production configs, compact per-seed curves, final slices,
   frozen-latent summary and figure input. Keep the tested archive available
   as a versioned release asset with a SHA-256 manifest. It carries the
   checkpoint and dataset inputs and historical reference needed by the current
   replay and training entry points. Preserve original archives and checksum
   histories. Document any new manifests for a curated release.
4. Keep the two GF comparisons paired by seed. For the two-Gaussian ablation,
   use LDNet/LDNet + GF and SSV/SSV + GF at fixed γ. If the manuscript includes
   Frozen Profile, port the observed-profile calculation to the final grid and
   compare its output directly with the appendix before shipping it.
5. Verify both protocols from a clean checkout and extracted release asset.
   Compare the regenerated plots and rounded table against the accepted PDF.
   Confirm the public scope and release URLs before pushing.

## Release audit

The compressed delivery is about 42 MB. The run outputs occupy about 44 MiB
uncompressed, including two roughly 18 MiB directories for the production GF
arms and controlled reference. Repeated full prediction arrays, histories,
energy grids, calibration outputs and the 249 MB collaborator ZIP belong
outside Git. Retain the source references and required input assets in the
versioned download. Run outputs should go to fresh scratch directories
because entry points support resuming and write progress on failures.

Keep the single-Gaussian root requirements. Preserve the two-Gaussian pins in
an experiment-specific environment (NumPy 2.3.5, PyTorch 2.11.0, Matplotlib
3.11.2; original Python 3.12 and CUDA 12.8). CPU validation here used NumPy
2.2.6, PyTorch 2.5.1+cu121 and Matplotlib 3.10.6. Full production retraining,
including the CUDA calibration, was outside this validation. A CPU
comparison smoke run with the archived CUDA SSV + GF source reached the
first fit, then failed the paired initialization SHA-256 assertion. The
cause of the mismatch was not isolated in this environment. A separate CPU calibration smoke run passed input checks, completed two reduced dictionary fits and reached final GF training. It was stopped during the 3,000-step fit. Confirm the match
in the original environment or regenerate all paired arms together in a
single environment and record the new run separately.

The targeted text scan found local-path references in legacy exploratory
scripts and no matching common credential signatures. The training/report
workflow also emits absolute paths in newly generated provenance. Check the
exported files, binary metadata and repository history before publication.
Exclude exploratory launchers, logs and unpublished workspace changes from
the proposed release. Keep archive input paths portable and verify relative
path handling independently of any historical provenance rewrite.
