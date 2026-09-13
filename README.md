# Self-Similar Latent Dynamics Network with Local Energy

Anonymous reproduction code for sparse-frame heat-equation extrapolation. This
release supports the final 3000-step experiment: five methods, five matched
seeds, 32 training trajectories, and three observed frames at
`t = {0, 0.25, 0.5}`.

## Quick start

Run commands from this directory with Python 3.10+:

```bash
python -m pip install -r requirements.txt
python scripts/generate_data.py --out data/single_gaussian_heat_fixed_domain_v1.npz
python scripts/plot.py
```

The compact `results/summary/aggregate.json` contains all 25 runs, 201-point
error curves, selected epochs, and mean/population standard deviations. Plotting
this file requires neither data nor trained checkpoints.

## CPU inference without training

```bash
python scripts/generate_data.py --out data/single_gaussian_heat_fixed_domain_v1.npz
python scripts/predict.py
```

This loads the supplied learned-SSV + local-energy seed-0 checkpoint, predicts
all 32 test trajectories at all 201 times, and checks the error curve against
the reported seed-0 results. Outputs in `outputs/inference/` are `metrics.json`,
`predictions.npz`, and `prediction.png`.

Expected seed-0 RelRMSE: observed 1.9758%, extrapolation 2.7504%, final 3.7990%.
These are one seed's scores, not the five-seed averages in the paper.

## Repeating training

To repeat training on GPU 0, run:

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
ignored by Git. The supplied `artifacts/pretrained/scale.pt` is the selected
scale checkpoint used in both learned-SSV methods.

## Protocol details

The analytic data have shape `(160, 201, 256)` on `[-16,16] x [0,10]`. A
permutation with seed 123 defines 96 training candidates, 32 validation
trajectories, and 32 test trajectories. Training uses the first 32 candidates
for every seed. Inputs are Gaussian parameters `(A,c,sigma)`.

Training details:

- LDNet uses shifted initialization and Forward Euler.
- The latent dimension is 16 and the hidden width is 128.
- The Fourier decoder uses 16 frequencies plus the constant coefficient.
- Training uses AdamW, weight decay `1e-6`, 3000 steps, cosine learning-rate
  decay from `1e-3` to `1e-5`, and gradient clipping at 5.
- Data loss and validation use only the three observed frames
  `t = {0, 0.25, 0.5}`.
- Validation selects the checkpoint with the lowest three-frame mean relative
  initial-norm error, evaluated at step 1 and every 10 steps.
- The GF loss uses 32 fixed rollout times: the three observed frames and
  29 unlabeled future times in `(0.5,10]`, rounded to the data grid.
- The GF loss uses 24 random spatial points per rollout state.
- The GF residual is normalized by its detached value when the GF loss is first
  activated; its weight ramps from step 101 to 200 up to `1e-4`.
- Reported full-domain RelRMSE aggregates squared norms over test trajectories
  before taking the square root. Window averages include all saved times.
  Standard deviations use `ddof=0`.

Optional scale refitting is provided for convenience and may produce slightly
different selected checkpoints due to restart initialization. The supplied scale
checkpoint has `alpha=0.5001377463` and `beta=0.4996300638`.

## Main results

Extrapolation RelRMSE (%):

- LDNet: 238.34 ± 81.92
- LDNet + learned SSV: 4.48 ± 0.84
- LDNet + learned SSV + local energy: 4.11 ± 0.90
- LDNet + oracle SSV: 4.48 ± 0.88
- LDNet + oracle SSV + local energy: 4.09 ± 0.90

See `docs/reproduction_audit.md` for preservation and dependency details.
