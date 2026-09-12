# Reproduction audit

Before editing, the complete working tree excluding Git internals was preserved
outside the repository with a per-file byte count and SHA-256 inventory.
This includes modified tracked files, untracked scripts, all experiments,
final data, scale selection records, 25 final checkpoints/metrics, and figures.
The private preservation directory is not a release artifact.

Dependency chain:
- generate_data.py → data/single_gaussian_heat_fixed_domain_v1.npz
- original scale.pt (or reconstructed pretrain_scale.py) → frozen scale
- train.py + data + frozen scale → five methods × seeds 0–4
- per-run full_relrmse_vs_time.npy + time.npy + summary.json → summarize.py
- results/summary/aggregate.json → plot.py
- data + learned_ssv_energy/seed0/model.pt + train.py → plot_latest_model_shapes.py
- 25 history.json files → plot_final_training_diagnostics.py

The old tracked README/trainer/plots described initial-field encoding, RBF,
expanding-window metrics and a 150-run matrix. They did not describe final3000.
The final trainer was untracked. The run launcher contained an absolute private
Python path, now replaced with PYTHON or python.
Historical scripts, metrics, and artifacts are preserved in the private backup.

The paper's conceptual overview image SS_LDNet_demo.png is absent from this
repository. It is not needed to reproduce numerical results.
The original scale fitting source is missing; refitting is explicitly documented
as reconstructed. The GF normalization omitted in the paper is documented in README.

Git history/remotes may contain identity information. This working tree is intended
for a fresh anonymous repository; its existing .git directory is not anonymized.

Inference addendum: artifacts/pretrained/learned_ssv_energy_seed0.pt and inference.json
provide a final model for scripts/predict.py. The supplied scale is embedded in
this model. A standalone source copy generated data and verified the full test
error curve without training; see inference_validation.json.
