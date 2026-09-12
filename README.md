# Self-Similar LDNet with Local Energy

Anonymous reproduction code for sparse-frame heat-equation extrapolation.
The release follows the final3000 experiment: five methods, five matched seeds,
32 training trajectories, and three observed frames at 0, 0.25, 0.5.

Run commands from this directory with Python 3.10+:
```bash
python -m pip install -r requirements.txt
python scripts/generate_data.py --out data/single_gaussian_heat_fixed_domain_v1.npz
python scripts/plot.py
```
The compact `results/summary/aggregate.json` contains all 25 runs, 201-point
error curves, selected epochs, and mean/population standard deviations.
Plotting this file requires neither data nor trained LDNet checkpoints.

To run CPU inference without training:
```bash
python scripts/generate_data.py --out data/single_gaussian_heat_fixed_domain_v1.npz
python scripts/predict.py
```
This loads the supplied final learned-SSV + local-energy seed-0 checkpoint,
including its learned scale, predicts all 32 test trajectories at all 201 times,
and checks the entire error curve against the published seed-0 results.
Outputs in `outputs/inference/` are `metrics.json`, `predictions.npz`, and
`prediction.png`. No separate scale fitting or training is performed.
Expected seed-0 RelRMSE: observed 1.9758%, extrapolation 2.7504%, final 3.7990%.
These are one seed's scores, not the five-seed averages in the paper.
The other 24 final models are not included in this source release.

To repeat training on GPU 0 (each command trains five seeds):
```bash
bash scripts/run_final3000_protocol.sh 0 physical
bash scripts/run_final3000_protocol.sh 0 learned_ssv
bash scripts/run_final3000_protocol.sh 0 learned_ssv_energy
bash scripts/run_final3000_protocol.sh 0 oracle_ssv
bash scripts/run_final3000_protocol.sh 0 oracle_ssv_energy
python scripts/summarize.py
python scripts/plot.py
python scripts/plot_latest_model_shapes.py
python scripts/plot_final_training_diagnostics.py
```
Set `PYTHON` to select a Python executable. Data and full training outputs are
ignored by Git. The supplied `artifacts/pretrained/scale.pt` is the original
selected scale checkpoint; it is frozen in both learned-SSV methods.
All 25 final checkpoints are retained locally under
`outputs/ssv_gamma_frozen_ldnet_v5_final3000_cosine`.

Optional scale refitting:
```bash
python scripts/pretrain_scale.py
python scripts/train.py --data data/single_gaussian_heat_fixed_domain_v1.npz --out_dir outputs/refit_run --mode learned_ssv --scale_checkpoint outputs/scale_refit/selected_model.pt
```
The scale pretraining entry point is reconstructed from the reported loss.
The original pretraining source was not located; its exact optimizer and
initialization cannot be established from the retained selection records.
Refitting therefore does not promise the original selected checkpoint.
The supplied checkpoint has alpha=0.5001377463 and beta=0.4996300638.

The analytic data have shape (160, 201, 256) on [-16,16] × [0,10].
A permutation with seed 123 defines 96 training candidates, 32 validation
and 32 test trajectories; training uses the first 32 candidates for every seed.
Inputs are Gaussian parameters (A,c,sigma). Dynamics use shifted initialization,
Forward Euler with physical grid spacing 0.05 (or its transformed clock),
latent dimension 16 and width 128. The continuous Fourier decoder has 16
frequencies plus its constant coefficient. Training uses AdamW, weight decay
1e-6, 3000 steps, cosine learning rate 1e-3 to 1e-5, and gradient clipping at 5.

Implementation details needed alongside the manuscript:
- Data loss divides each trajectory's squared error by its initial squared norm.
- Validation selects the lowest three-frame mean relative initial-norm error,
  evaluated at step 1 and every 10 steps; reported metrics use that checkpoint.
- GF uses 32 fixed rollout times (three observed, 29 future times rounded up
  onto the data grid) and 24 random spatial points per state.
- GF residual is divided by its detached value at step 101. Its weight ramps
  from step 101 to 200 up to 1e-4. This normalization is absent from the paper's
  displayed loss formula and should be clarified there.
- Reported full-domain RelRMSE aggregates squared norms over test trajectories
  before taking the square root. Window averages include all saved times,
  not just the three supervised frames. Standard deviations use ddof=0.

Extrapolation RelRMSE (%): LDNet 238.34 ± 81.92; learned SSV 4.48 ± 0.84;
learned SSV + energy 4.11 ± 0.90; oracle SSV 4.48 ± 0.88;
oracle SSV + energy 4.09 ± 0.90.

See `docs/reproduction_audit.md` for preservation and dependency details.
